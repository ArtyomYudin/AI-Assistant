import logging
from typing import AsyncGenerator, List, Optional

from langchain_core.documents import Document

from config.rag_config import RAGConfig
from core.chat_history import ChatHistoryManager
from core.document_loader import load_documents_from_directory
from core.milvus_manager import MilvusManager
from core.splitters import SplitterManager
from core.utils import count_tokens, truncate_text_by_tokens
from core.redis_embedding_cache import RedisEmbeddingCache

logger = logging.getLogger(__name__)

class RAGCore:
    """
      Основной класс, объединяющий все компоненты RAG-пайплайна:
      - загрузка и индексация документов
      - поиск по Milvus (векторное хранилище)
      - генерация эмбеддингов
      - кэширование эмбеддингов в Redis
      - управление историей диалога
      - генерация ответов с использованием LLM
    """
    def __init__(self, config: Optional[RAGConfig] = None):
        # Загружаем конфиг или используем значения по умолчанию
        self.config = config or RAGConfig()

        # Менеджер истории диалога (хранит переписку)
        self.history = ChatHistoryManager(self.config.MAX_HISTORY_MESSAGES)

        # Подключение к Milvus (векторное хранилище)
        self.milvus = MilvusManager(
            uri=self.config.MILVUS_URI,
            collection_name=self.config.COLLECTION_NAME,
            recreate=self.config.RECREATE_COLLECTION
        )

        # Управление сплиттингом текста (по заголовкам, чанкам)
        self.splitters = SplitterManager(
            self.config.CHUNK_SIZE,
            self.config.CHUNK_OVERLAP
        )

        # Лениво инициализируемые объекты (создаются при первом вызове)
        self._embeddings = None
        self._llm = None

        # Ретривер (поиск документов по запросу)
        self.retriever = None

        # Цепочка вопрос-ответ с историей и стримингом
        self.qa_chain_with_history = None

        # Кэш эмбеддингов запросов в Redis
        self.embedding_cache = RedisEmbeddingCache(
            host=self.config.REDIS_HOST,
            port=self.config.REDIS_PORT,
            ttl=self.config.REDIS_TTL
        )

    @property
    # def embeddings(self):
    #     if self._embeddings is None:
    #         from langchain_openai import OpenAIEmbeddings
    #         self._embeddings = OpenAIEmbeddings(
    #             model=self.config.EMBEDDING_NAME,
    #             api_key="EMPTY",
    #             base_url=self.config.EMBEDDING_BASE_URL,
    #             embedding_ctx_length=512,
    #             timeout=60,
    #         )
    #     return self._embeddings
    def embeddings(self):
        """
        Локальная или внешняя модель эмбеддингов.
        При первом вызове создаётся объект.
        """
        if self._embeddings is None:
            from core.local_embeddings import LocalEmbeddings
            self._embeddings = LocalEmbeddings(
                model=self.config.EMBEDDING_NAME,
                base_url=self.config.EMBEDDING_BASE_URL,
                timeout=60,
            )
        return self._embeddings

    @property
    def llm(self):
        """
        LLM-клиент (ChatOpenAI API-совместимый).
        Используется для генерации ответов.
        """
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self.config.LLM_NAME,
                api_key="EMPTY",
                base_url=self.config.LLM_BASE_URL,
                max_tokens=self.config.LLM_MAX_TOKEN,
                temperature=self.config.LLM_TEMPERATURE,
                streaming=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
        return self._llm

    def load_documents(self, directory: Optional[str] = None) -> List[Document]:
        """
        Загружает документы из указанной директории.
        По умолчанию — директория config.DATA_DIR.
        """
        directory = directory or self.config.DATA_DIR
        return load_documents_from_directory(directory)

    async def setup_vectorstore(self, documents: List[Document]) -> None:
        """
        Индексация документов в Milvus:
        - нарезка на чанки
        - генерация эмбеддингов
        - вставка в Milvus
        """
        if not documents:
            logger.warning("Нет документов для индексации")
            return

        # Сплитим документы по заголовкам / чанкам
        processed = self.splitters.split_by_headers(documents)
        if not processed:
            logger.warning("Нет документов после обработки")
            return
        texts = [d.page_content for d in processed]
        sources = [d.metadata.get("source","N/A") for d in processed]
        hashes = [d.metadata.get("hash") for d in processed]

        # Генерация эмбеддингов для каждого чанка
        try:
            # texts = [d.page_content for d in processed]
            # print("DEBUG types:", [type(t) for t in texts[:5]])
            # print("DEBUG sample:", texts[:2])
            dense_vectors = self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.exception("Ошибка при эмбеддингах: %s", e)
            raise
        dense_dim = len(dense_vectors[0]) if dense_vectors else 0
        if dense_dim <= 0:
            logger.error("Не удалось определить размерность эмбеддинга")
            return

        # Создаём коллекцию в Milvus (если нет)
        await self.milvus.create_collection_if_needed(dense_dim=dense_dim)

        # Подготавливаем данные для вставки
        rows = [{"text": t, "source": s, "hash": h or "", "dense_vector": dv} for t, s, dv, h in zip(texts, sources, dense_vectors, hashes)]

        # Убираем дубликаты
        unique = await self.milvus.ensure_not_duplicate_rows(rows)
        if not unique:
            logger.info("Все фрагменты уже в индексе")
            return

        # Вставка чанками (batch insert)
        bs = self.config.INDEX_BATCH_SIZE
        total = 0
        for i in range(0, len(unique), bs):
            total += await self.milvus.insert_records(unique[i:i+bs])
        logger.info("Проиндексировано фрагментов: %d", total)

        # Загружаем коллекцию в память сразу после вставки
        await self.milvus.client.load_collection(self.milvus.collection_name)
        logger.info("Коллекция %s загружена в память", self.milvus.collection_name)

    def create_retriever(self, k: Optional[int] = None, fetch_k: Optional[int] = None) -> None:
        """
        Создаёт асинхронный ретривер:
        - ищет документы в Milvus
        - использует кэширование эмбеддингов запросов
        """
        k = k or self.config.K
        fetch_k = fetch_k or self.config.FETCH_K

        async def retrieve(query: str) -> List[Document]:
            if not query.strip():
                return []

            # Проверяем, что коллекция существует
            if not await self.milvus.client.has_collection(self.milvus.collection_name):
                logger.warning("Коллекция %s не найдена", self.milvus.collection_name)
                return []

            # Подстраховка: загружаем коллекцию перед поиском
            if not await self.milvus.client.is_collection_loaded(self.milvus.collection_name):
                await self.milvus.client.load_collection(self.milvus.collection_name)
                logger.info("Коллекция %s загружена", self.milvus.collection_name)

            try:
                # проверка кэша из Redis
                cached = self.embedding_cache.get(query)
                if cached is not None:
                    dense = cached
                else:
                    # Генерируем эмбеддинг для запроса
                    dense = self.embeddings.embed_query(query)
                    self.embedding_cache.set(query, dense)
            except Exception as e:
                logger.exception("Ошибка при эмбеддинге запроса: %s", e)
                return []

            # Делаем гибридный поиск (по вектору + тексту)
            return await self.milvus.hybrid_search(
                query_text=query,
                query_dense=dense,
                fetch_k=fetch_k,
                top_k=k,
                reranker_endpoint=self.config.RERANKER_BASE_URL
            )

        self.retriever = retrieve
        logger.info("Ретривер создан (k=%d, fetch_k=%d)", k, fetch_k)

    def _build_context(self, docs: List[Document], session_id: str) -> str:
        """
        Формирует контекст для LLM:
        - учитывает лимит токенов
        - добавляет историю чата
        - включает top-k документов
        """
        # Сколько токенов можно занять под документы
        available = self.config.MAX_CONTEXT_TOKENS - self.config.RESERVED_FOR_COMPLETION

        # История чата
        hist = self.history.get(session_id)
        history_text = "\n".join(f"{m.type.capitalize()}: {m.content}" for m in hist.messages)
        available -= count_tokens(history_text) + self.config.RESERVED_FOR_OVERHEAD
        available = max(0, available)
        if available <= 0: return ""

        # Добавляем документы до исчерпания лимита
        parts, used = [], 0
        for d in sorted(docs, key=lambda x: x.metadata.get("score", 0), reverse=True):
            text = f"[Источник: {d.metadata.get('source','N/A')}] {d.page_content.strip()}\n"
            tks = count_tokens(text)
            if used + tks > available:
                remain = available - used
                if remain > 50:
                    parts.append(truncate_text_by_tokens(text, remain))
                break
            parts.append(text)
            used += tks
        return "".join(parts).strip()

    def create_qa_generator(self) -> None:
        """
        Создаёт генератор Q&A:
        - ищет документы
        - строит контекст
        - вызывает LLM потоково
        - сохраняет историю чата
        """
        if not self.retriever:
            raise ValueError("Ретривер не создан. Вызовите create_retriever().")
        async def generate_answer_stream(question: str, session_id: str = "default"):
            if not question.strip():
                yield "Пожалуйста, задайте вопрос."
                return

            # Поиск документов
            docs = await self.retriever(question)
            if not docs:
                yield "Информация не найдена."
                return

            # Формируем контекст для LLM
            context = self._build_context(docs, session_id)
            if not context:
                yield "Информация не найдена."
                return

            # История диалога
            hist = self.history.get(session_id)
            history_text = "\n".join(f"{m.type.capitalize()}: {m.content}" for m in hist.messages)

            # Финальный промпт
            prompt = self.config.QA_PROMPT_TEMPLATE.format(context=context, chat_history=history_text, question=question)

            # Добавляем вопрос в историю
            hist.add_user_message(question)

            full = ""
            try:
                # Потоковый вызов LLM
                async for chunk in self.llm.astream([{ "role": "user", "content": prompt }]):
                    if content := chunk.content:
                        full += content
                        yield content

                # Сохраняем ответ в историю
                hist.add_ai_message(full)
            except Exception as e:
                logger.exception("Ошибка при генерации ответа: %s", e)
                yield "Произошла ошибка при генерации ответа."

        self.qa_chain_with_history = generate_answer_stream
        logger.info("QA-генератор со стримингом и историей настроен")

    async def close(self) -> None:
        # Корректно закрываем соединение с Milvus
        if hasattr(self.milvus.client, "close"):
            try: await self.milvus.client.close()
            except Exception: pass
