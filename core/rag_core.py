# core/rag_core.py
import contextlib
import logging
from typing import List, Optional, AsyncIterator

import httpx
from langchain_core.documents import Document

from config.rag_config import RAGConfig
from core.orchestration.answer_generator import AnswerGenerator
from core.storage.chat_history import RedisChatHistory
from core.orchestration.collection_manager import CollectionManager
from core.ingestion.document_indexer import DocumentIndexer
from core.storage.embedding_cache import RedisEmbeddingCache
from core.storage.milvus_manager import MilvusManager
from core.models.model_provider import ModelProvider
from core.orchestration.prompt_builder import PromptBuilder
from core.models.rate_limiter import LLMRateLimiter
from core.ingestion.splitter import SplitterManager
from core.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)


class RAGCore:
    """
    Фасад RAG-пайплайна. Собирает компоненты и делегирует им работу.

    Компоненты:
      - ModelProvider     — embeddings / LLM / LLM+tools / tool_check_llm
      - DocumentIndexer   — загрузка и индексация в Milvus
      - PromptBuilder     — контекст и сообщения
      - ToolExecutor      — выполнение tool_calls и валидация
      - AnswerGenerator   — стриминг ответа (tool-flow + RAG-flow)
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()

        self.history = RedisChatHistory(
            session_id="default",
            host=self.config.REDIS_HOST,
            port=self.config.REDIS_PORT,
            ttl_days=self.config.HISTORY_TTL_DAYS,
            max_messages=self.config.MAX_HISTORY_MESSAGES,
        )

        self.milvus = MilvusManager(
            uri=self.config.MILVUS_URI,
            collection_name=self.config.COLLECTION_NAME,
            recreate=self.config.RECREATE_COLLECTION,
        )

        self.splitters = SplitterManager(
            self.config.CHUNK_SIZE,
            self.config.CHUNK_OVERLAP,
        )

        self.models = ModelProvider(self.config)
        self.prompts = PromptBuilder(self.config)
        self.tool_executor = ToolExecutor()
        self.indexer = DocumentIndexer(
            config=self.config,
            splitters=self.splitters,
            models=self.models,
            milvus=self.milvus,
        )

        self.embedding_cache = RedisEmbeddingCache(
            host=self.config.REDIS_HOST,
            port=self.config.REDIS_PORT,
            ttl=self.config.REDIS_TTL,
        )

        self.collection_manager = CollectionManager(
            config=self.config,
            milvus=self.milvus,
            indexer=self.indexer,
            models=self.models,
            load_documents_fn=self.indexer.load_documents,
            get_embedding_cached_fn=self._get_embedding_cached,
        )

        self.llm_limiter = LLMRateLimiter(
            max_concurrent=self.config.LLM_MAX_CONCURRENT,
            max_queue_wait=self.config.LLM_MAX_QUEUE_WAIT,
            max_per_user_per_min=self.config.LLM_MAX_PER_USER_PER_MIN,
        )

        self.retriever = None
        self.answer_generator: Optional[AnswerGenerator] = None
        # self.qa_chain_with_history = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def __repr__(self):
        s = self.llm_limiter.stats()
        return (
            f"<RAGCore mode={self.config.MODE}, "
            f"collection={self.config.COLLECTION_NAME}, "
            f"llm_slots={s['active']}/{s['max_concurrent']}>"
        )

    def __str__(self):
        return self.__repr__()

    # ── История / embeddings ────────────────────────────────────

    def get_history(self, session_id: str) -> RedisChatHistory:
        return RedisChatHistory(
            session_id=session_id,
            host=self.config.REDIS_HOST,
            port=self.config.REDIS_PORT,
            ttl_days=self.config.HISTORY_TTL_DAYS,
            max_messages=self.config.MAX_HISTORY_MESSAGES,
        )

    async def _get_embedding_cached(self, query: str) -> Optional[List[float]]:
        cached = self.embedding_cache.get(query)
        if cached is not None:
            return cached
        try:
            emb = await self.models.embeddings.embed_query(query)
        except (httpx.HTTPError, ValueError, RuntimeError) as e:
            logger.warning("Ошибка эмбеддинга запроса: %s", e)
            return None

        try:
            self.embedding_cache.set(query, emb)
        except Exception as e:  # ← остаётся, но с явной причиной
            # Redis не критичен для корректности ответа — только для latency
            logger.warning("Не удалось закэшировать эмбеддинг: %s", e)
        return emb

    # ── Загрузка / индексация ───────────────────────────────────

    def load_documents(self, directory: Optional[str] = None) -> List[Document]:
        return self.indexer.load_documents(directory)

    async def setup_vectorstore(self, documents: List[Document]) -> None:
        await self.indexer.setup_vectorstore(documents)

    # ── Retriever ───────────────────────────────────────────────

    def create_retriever(self, k: Optional[int] = None, fetch_k: Optional[int] = None) -> None:
        k = k or self.config.K
        fetch_k = fetch_k or self.config.FETCH_K

        async def routed_retrieve(query: str, session_id: str | None = None) -> List[Document]:
            if not query.strip():
                return []

            sid = session_id or "global"

            try:
                docs = await self.collection_manager.routed_search_with_fallback(
                    query, conversation_id=sid
                )
            except Exception as e:
                logger.exception("[%s] Ошибка routed_search: %s", sid, e)
                return []

            logger.info(
                "[%s] retrieval: docs=%d, collections=%s, scores=%s",
                sid, len(docs),
                [d.metadata.get("collection") for d in docs],
                [round(d.metadata.get("score", 0), 3) for d in docs],
            )
            return docs

        self.retriever = routed_retrieve
        logger.info("Ретривер с маршрутизацией создан (k=%d, fetch_k=%d)", k, fetch_k)


    # ── QA-генератор ────────────────────────────────────────────

    async def qa_chain_with_history(
            self, question: str, user_id: str = "default"
    ) -> AsyncIterator[dict]:
        """
        Делегат в AnswerGenerator.stream — стабильный публичный API для UI/API.
        Позволяет не зависеть от того, что `answer_generator` — атрибут,
        который меняется при повторной инициализации.
        """
        if self.answer_generator is None:
            raise RuntimeError(
                "QA-генератор не создан. Вызовите create_qa_generator()."
            )
        async for chunk in self.answer_generator.stream(question, user_id=user_id):
            yield chunk

    def create_qa_generator(self) -> None:
        if not self.retriever:
            raise ValueError("Ретривер не создан. Вызовите create_retriever().")

        self.answer_generator = AnswerGenerator(
            config=self.config,
            retriever=self.retriever,
            history_factory=self.get_history,
            models=self.models,
            prompts=self.prompts,
            tool_executor=self.tool_executor,
            limiter=self.llm_limiter,
        )
        logger.info("QA-генератор со стримингом и историей настроен")

    # ── Warmup / close ──────────────────────────────────────────

    async def warmup(self) -> None:
        import time as _t
        t0 = _t.monotonic()

        try:
            from core.common.text import get_tokenizer, count_tokens
            get_tokenizer()
            count_tokens("warmup")
            logger.info("✅ Токенизатор прогрет")
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            logger.warning("⚠️ Токенизатор warmup failed: %s", e)

        await self.models.warmup()

        logger.info("🎯 Warmup завершён за %.2fс", _t.monotonic() - t0)

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await self.milvus.close()
        with contextlib.suppress(Exception):
            await self.models.aclose()
        with contextlib.suppress(Exception):
            self.embedding_cache.client.close()
        with contextlib.suppress(Exception):
            await self.collection_manager.redis.close()
        with contextlib.suppress(Exception):
            self.history.redis_client.close()

        logger.info("RAGCore: все ресурсы закрыты")