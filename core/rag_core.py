# core/rag_core.py

import asyncio
import logging
import time
import json

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage

from config.rag_config import RAGConfig
from core.chat_history import RedisChatHistory
from core.document_loader import load_documents_from_directory
from core.milvus_manager import MilvusManager
from core.polite_detector import PoliteDetector
from core.splitters import SplitterManager
from core.utils import count_tokens, truncate_text_by_tokens
from core.embedding_cache import RedisEmbeddingCache
from core.collection_manager import CollectionManager
from core.rate_limiter import LLMRateLimiter, RateLimitExceeded

from core.tools.registry import ALL_TOOLS, TOOL_REGISTRY

logger = logging.getLogger(__name__)


class RAGCore:
    """
    Основной класс RAG-пайплайна:
      - загрузка и индексация документов
      - поиск по Milvus (гибридный dense + BM25 + reranker)
      - маршрутизация по коллекциям (multi-collection)
      - генерация ответов с LLM (стриминг + история)
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()

        # История диалога
        self.history = RedisChatHistory(
            session_id="default",
            host=self.config.REDIS_HOST,
            port=self.config.REDIS_PORT,
            ttl_days=self.config.HISTORY_TTL_DAYS,
            max_messages=self.config.MAX_HISTORY_MESSAGES,
        )

        # Milvus
        self.milvus = MilvusManager(
            uri=self.config.MILVUS_URI,
            collection_name=self.config.COLLECTION_NAME,
            recreate=self.config.RECREATE_COLLECTION,
        )

        # Сплиттер
        self.splitters = SplitterManager(
            self.config.CHUNK_SIZE,
            self.config.CHUNK_OVERLAP,
        )

        # Лениво инициализируемые объекты
        self._embeddings = None
        self._llm = None
        self._llm_with_tools = None
        self._tool_check_llm = None

        # Ретривер
        self.retriever = None

        # QA-генератор
        self.qa_chain_with_history = None

        # Кэш эмбеддингов
        self.embedding_cache = RedisEmbeddingCache(
            host=self.config.REDIS_HOST,
            port=self.config.REDIS_PORT,
            ttl=self.config.REDIS_TTL,
        )

        # Менеджер коллекций (роутер)
        self.collection_manager = CollectionManager(core=self)

        # Rate-limit на LLM: защита vLLM от перегрузки + per-session лимит
        self.llm_limiter = LLMRateLimiter(
            max_concurrent=self.config.LLM_MAX_CONCURRENT,
            max_queue_wait=self.config.LLM_MAX_QUEUE_WAIT,
            max_per_user_per_min=self.config.LLM_MAX_PER_USER_PER_MIN,
        )

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

    # ─────────────────────────────────────────────────────────────
    # Ленивая инициализация моделей
    # ─────────────────────────────────────────────────────────────

    @property
    def embeddings(self):
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
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self.config.LLM_NAME,
                api_key="EMPTY",
                base_url=self.config.LLM_BASE_URL,
                max_tokens=self.config.LLM_MAX_TOKEN,
                temperature=self.config.LLM_TEMPERATURE,
                top_p=self.config.LLM_TOP_P,
                presence_penalty=self.config.LLM_PRESENCE_PENALTY,
                streaming=True,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "top_k": self.config.LLM_TOP_K,
                    "bad_words": [
                        "Если нужно — могу",
                        "Если нужно, могу",
                        "Если нужно – могу",
                        "Если нужно, я могу",
                        "Если нужно — я могу",
                        "Если потребуется — могу",
                        "Если потребуется, могу",
                        "Если необходимо — могу",
                        "Если необходимо, могу",
                        "Если понадобится — могу",
                        "Если понадобится, могу",
                        "Дайте знать, если",
                        "Дайте знать если",
                        "Обращайтесь, если",
                        "Обращайтесь если",
                        "Готов помочь",
                        "Готова помочь",
                        "Рад помочь",
                        "Рада помочь",
                        "могу экспортировать",
                        "могу прислать",
                        "могу отфильтровать",
                        "могу создать",
                        "могу сделать",
                        "могу предоставить",
                        "могу уточнить",
                        "могу показать",
                        "могу предложить",
                    ],
                },
            )
        return self._llm


    @property
    def llm_with_tools(self):
        """LLM с привязанными инструментами (ленивая инициализация)."""
        if self._llm_with_tools is None:
            self._llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
            logger.info(
                "LLM bind_tools: %d инструментов → %s",
                len(ALL_TOOLS),
                [t["function"]["name"] for t in ALL_TOOLS],
            )
        return self._llm_with_tools


    @property
    def tool_check_llm(self):
        """Лёгкий LLM только для решения «нужен ли инструмент»."""
        if self._tool_check_llm is None:
            # ★ bind к llm_with_tools (не к llm) — переиспользует уже
            # сделанный bind_tools, не строит его заново
            self._tool_check_llm = self.llm_with_tools.bind(
                max_tokens=64,
                temperature=0.0,
            )
        return self._tool_check_llm

    async def _exec_tool(self, tool_call: dict) -> dict:
        name = tool_call["name"]
        args = tool_call.get("args") or {}
        tool_id = tool_call.get("id", "?")

        func = TOOL_REGISTRY.get(name)
        if func is None:
            logger.warning("Неизвестный инструмент: %s (id=%s)", name, tool_id)
            return {"error": f"Инструмент '{name}' не найден"}

        try:
            logger.info("🔧 tool_call: %s(%s) id=%s", name, args, tool_id)

            # ★ Универсальный потолок — ни один инструмент не занимает >5 сек
            # ★ Поддержка StructuredTool (если кто-то обернёт в @tool)
            coro = func.ainvoke(args) if hasattr(func, "ainvoke") else func(**args)
            result = await asyncio.wait_for(coro, timeout=5.0)

            if not isinstance(result, dict):
                result = {"result": result}
            return result

        except asyncio.TimeoutError:
            logger.warning("⏱️ tool %s превысил 5 сек — прерываю", name)
            return {"error": f"Инструмент {name} не ответил вовремя"}
        except TypeError as e:
            logger.exception("Неверные аргументы для %s(%s): %s", name, args, e)
            return {"error": f"Неверные аргументы для {name}: {e}"}
        except Exception as e:
            logger.exception("Ошибка выполнения %s: %s", name, e)
            return {"error": f"Ошибка выполнения {name}: {e}"}


    async def _index_documents(
            self,
            collection_name: str,
            documents: List[Document],
    ) -> int:
        """
        Каноничный путь индексации в коллекцию.
        Возвращает число вставленных фрагментов.

        Правила:
          - text       = тело чанка БЕЗ title (для LLM-контекста и reranker'а)
          - text_full  = title + body (формируется в MilvusManager.insert_records)
          - dense      = эмбеддинг ТОЛЬКО от body
          - bm25_text  = title + global_bm25 + body[:2500] (из SplitterManager)
        """
        if not documents:
            return 0

        processed = self.splitters.split_by_headers(documents)
        if not processed:
            logger.warning("Нет чанков после сплиттинга для %s", collection_name)
            return 0

        # ★ Ключевое: text = body без title, эмбеддинг от body
        texts = [d.page_content for d in processed]
        titles = [d.metadata.get("title", "") for d in processed]
        bm25_texts = [d.metadata.get("bm25_text", "") for d in processed]
        sources = [d.metadata.get("source", "N/A") for d in processed]
        hashes = [d.metadata.get("hash", "") for d in processed]

        # Эмбеддинги — от тела чанка
        try:
            dense_vectors = await self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.exception("Ошибка эмбеддинга для %s: %s", collection_name, e)
            raise

        if not dense_vectors:
            logger.error("Пустой список эмбеддингов для %s", collection_name)
            return 0
        dense_dim = len(dense_vectors[0])

        # Milvus-менеджер на эту коллекцию
        milvus = MilvusManager(
            uri=self.config.MILVUS_URI,
            collection_name=collection_name,
            recreate=self.config.RECREATE_COLLECTION,
        )
        await milvus.create_collection_if_needed(dense_dim=dense_dim)

        rows = [
            {
                "text": t,
                "title": tit,
                "bm25_text": bm25,
                "source": s,
                "hash": h,
                "dense_vector": dv,
            }
            for t, tit, bm25, s, h, dv in zip(
                texts, titles, bm25_texts, sources, hashes, dense_vectors
            )
        ]

        unique = await milvus.ensure_not_duplicate_rows(rows)
        if not unique:
            logger.info("Все фрагменты уже в %s", collection_name)
            return 0

        bs = self.config.INDEX_BATCH_SIZE
        total = 0
        for i in range(0, len(unique), bs):
            total += await milvus.insert_records(unique[i:i + bs])

        await milvus.client.load_collection(collection_name)
        logger.info(
            "Коллекция %s проиндексирована: %d фрагментов (dim=%d)",
            collection_name, total, dense_dim,
        )
        return total

    # ─────────────────────────────────────────────────────────────
    # Работа с историей
    # ─────────────────────────────────────────────────────────────

    def get_history(self, session_id: str) -> RedisChatHistory:
        return RedisChatHistory(
            session_id=session_id,
            host=self.config.REDIS_HOST,
            port=self.config.REDIS_PORT,
            ttl_days=self.config.HISTORY_TTL_DAYS,
            max_messages=self.config.MAX_HISTORY_MESSAGES,
        )

    # ─────────────────────────────────────────────────────────────
    # Загрузка и индексация
    # ─────────────────────────────────────────────────────────────

    def load_documents(self, directory: Optional[str] = None) -> List[Document]:
        directory = directory or self.config.DATA_DIR
        return load_documents_from_directory(directory)


    async def setup_vectorstore(self, documents: List[Document]) -> None:
        """Индексирует документы в основную коллекцию (config.COLLECTION_NAME)."""
        total = await self._index_documents(self.config.COLLECTION_NAME, documents)
        if total == 0:
            logger.warning("setup_vectorstore: вставлено 0 фрагментов")

    # ─────────────────────────────────────────────────────────────
    # Эмбеддинги запроса (с кэшем)
    # ─────────────────────────────────────────────────────────────

    async def _get_embedding_cached(self, query: str) -> Optional[List[float]]:
        cached = self.embedding_cache.get(query)
        if cached is not None:
            return cached
        try:
            emb = await self.embeddings.embed_query(query)
            self.embedding_cache.set(query, emb)
            return emb
        except Exception as e:
            logger.warning("Ошибка эмбеддинга запроса: %s", e)
            return None

    # ─────────────────────────────────────────────────────────────
    # Ретривер (multi-collection + per-collection threshold + fallback)
    # ─────────────────────────────────────────────────────────────

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

    # ─────────────────────────────────────────────────────────────
    # Контекст и сообщения
    # ─────────────────────────────────────────────────────────────

    def _build_context(
        self,
        docs: List[Document],
        history: List[BaseMessage],
    ) -> tuple[str, List[BaseMessage], str]:
        """
        Формирует контекст для LLM с учётом:
          - истории чата (обрезанной по токенам + резюме старых)
          - контекста документов
          - лимита токенов MAX_CONTEXT_TOKENS
          - детерминированного порядка документов

        Возвращает (context, truncated_history, history_summary).
        """
        # История: обрезаем с конца, сохраняя роли
        truncated_rev: List[BaseMessage] = []
        history_tokens = 0

        for m in reversed(history):
            msg_text = f"{m.type.capitalize()}: {m.content}"
            tks = count_tokens(msg_text)
            if history_tokens + tks > self.config.MAX_HISTORY_TOKENS:
                break
            truncated_rev.insert(0, m)
            history_tokens += tks

        truncated_history: List[BaseMessage] = truncated_rev

        # Резюме отброшенных сообщений
        history_summary = ""
        num_skipped = len(history) - len(truncated_history)
        if num_skipped > 0:
            old_msgs = history[:num_skipped]
            old_text = "\n".join(
                f"{m.type.capitalize()}: {truncate_text_by_tokens(m.content, 200)}"
                for m in old_msgs
            )
            history_summary = f"[Резюме предыдущих сообщений]: {old_text}"
            history_tokens += count_tokens(history_summary)

        # Бюджет на документы
        available = (
            self.config.MAX_CONTEXT_TOKENS
            - self.config.RESERVED_FOR_COMPLETION
            - history_tokens
            - self.config.RESERVED_FOR_OVERHEAD
        )
        available = max(0, available)
        if available <= 0:
            return "", truncated_history, history_summary

        # Детерминированная сортировка документов
        docs_sorted = sorted(
            docs,
            key=lambda d: (-d.metadata.get("score", 0), d.metadata.get("source", "")),
        )

        # Формирование контекста
        parts, used = [], 0
        for d in docs_sorted:
            text = f"[Источник: {d.metadata.get('source', 'N/A')}] {d.page_content.strip()}\n"
            tks = count_tokens(text)
            if used + tks > available:
                remain = available - used
                if remain > self.config.MIN_DOC_TOKEN:
                    parts.append(truncate_text_by_tokens(text, remain))
                break
            parts.append(text)
            used += tks

        context = "".join(parts).strip()
        return context, truncated_history, history_summary

    def _build_messages(
        self,
        mode: str,
        question: str,
        context: str,
        truncated_history: List[BaseMessage],
        history_summary: str = "",
    ) -> List[BaseMessage]:
        """
        Структура:
          1) SystemMessage: правила (+ резюме истории)
          2) truncated_history: user/assistant в исходных ролях
          3) HumanMessage: контекст + текущий вопрос
        """
        system_content = self.config.QA_PROMPT_HYBRID_RU_SYSTEM
        if mode == "rag":
            system_content = getattr(self.config, "QA_PROMPT_RAG_SYSTEM", system_content)
        elif mode == "llm_only":
            system_content = getattr(self.config, "QA_PROMPT_LLM_ONLY_SYSTEM", system_content)

        if history_summary:
            system_content = (
                f"{system_content}\n\n"
                f"---\n"
                f"КОНТЕКСТ ПРЕДЫДУЩЕГО ДИАЛОГА (не факты, только для связности):\n"
                f"{history_summary}"
            )

        messages: List[BaseMessage] = [SystemMessage(content=system_content)]
        messages.extend(truncated_history)

        if context.strip():
            user_content = f"Контекст:\n{context}\n\nВопрос: {question}"
        else:
            user_content = question

        messages.append(HumanMessage(content=user_content))
        return messages


    async def warmup(self) -> None:
        """
        Единая точка прогрева всех «ленивых» ресурсов.
        Вызывать из lifespan (api) и из старта gradio — один раз.
        Прогрев токенизатора — синхронный, embeddings — async.
        """
        import time as _t
        t0 = _t.monotonic()

        # Токенизатор (native HF или tiktoken fallback)
        try:
            from core.utils import get_tokenizer, count_tokens
            get_tokenizer()
            count_tokens("warmup")
            logger.info("✅ Токенизатор прогрет")
        except Exception as e:
            logger.warning("⚠️ Токенизатор warmup failed: %s", e)

        # Embeddings (FRIDA на GPU1 — первый вызов ~500мс)
        try:
            await self.embeddings.embed_query("warmup")
            logger.info("✅ Embeddings прогреты")
        except Exception as e:
            logger.warning("⚠️ Embeddings warmup failed: %s", e)

        # LLM: создаёт ChatOpenAI + httpx-клиент + bind_tools
        try:
            _ = self.llm
            _ = self.llm_with_tools
            _ = self.tool_check_llm
            # Реальный ping: устанавливает TCP и греет планировщик vLLM
            await self.llm.bind(max_tokens=1).ainvoke([HumanMessage(content=".")])
            logger.info("✅ LLM прогреты: %d инструментов", len(ALL_TOOLS))
        except Exception as e:
            logger.warning("⚠️ LLM warmup failed: %s", e)

        logger.info("🎯 Warmup завершён за %.2fс", _t.monotonic() - t0)


    # ─────────────────────────────────────────────────────────────
    # QA-генератор
    # ─────────────────────────────────────────────────────────────

    def create_qa_generator(self) -> None:
        """
        Создаёт генератор Q&A:
          - проверяет вежливые фразы
          - параллельно: поиск документов + загрузка истории
          - строит контекст по токен-бюджету
          - стримит ответ LLM
          - сохраняет в историю
        """
        if not self.retriever:
            raise ValueError("Ретривер не создан. Вызовите create_retriever().")

        async def generate_answer_stream(question: str, user_id: str = "default"):
            start_time = time.time()
            logger.debug(f"[{user_id}] 🚀 Начал обработку: {question[:100]}...")

            # 1. Пустой вопрос
            if not question.strip():
                yield {"type": "response", "text": "Пожалуйста, задайте вопрос."}
                return

            # 2. Вежливые фразы — без LLM
            if PoliteDetector.is_polite(question):
                yield {"type": "response", "text": PoliteDetector.reply(question)}
                return

            # yield {"type": "status", "text": "🤔 Определяю, что нужно сделать...\n"}
            yield {"type": "status", "text": "🔍 Ищу информацию...\n"}

            # ═══════════════════════════════════════════════════════════════
            # ПАРАЛЛЕЛЬНЫЙ СТАРТ: retrieval + история
            # retrieval понадобится только если инструмент не нужен.
            # Запускаем заранее — latency скроется за tool-check.
            # ═══════════════════════════════════════════════════════════════
            retrieval_task = asyncio.create_task(self.retriever(question, user_id))
            history_task = asyncio.create_task(
                asyncio.to_thread(self.get_history(user_id).get_messages)
            )

            try:
                hist = await history_task
                logger.debug(f"[{user_id}] ✅ История: {len(hist)} сообщений")
            except Exception as e:
                logger.exception(f"[{user_id}] ❌ Ошибка истории: {e}")
                hist = []

            # ═══════════════════════════════════════════════════════════════
            # TOOL-CHECK: модель решает, нужен ли инструмент
            # ═══════════════════════════════════════════════════════════════
            # tool_messages: List[BaseMessage] = [
            #     SystemMessage(content=self.config.QA_PROMPT_HYBRID_RU_SYSTEM),
            #     *hist,
            #     HumanMessage(content=question),
            # ]
            recent_hist = hist[-4:] if len(hist) > 4 else hist
            tool_messages: List[BaseMessage] = [
                SystemMessage(content=self.config.QA_PROMPT_TOOL_CHECK_SYSTEM),
                *recent_hist,
                HumanMessage(content=question),
            ]

            first_response = None
            try:
                # first_response = await self.llm_with_tools.ainvoke(tool_messages)
                first_response = await self.tool_check_llm.ainvoke(tool_messages)
            except Exception as e:
                # Не падаем — проваливаемся в обычный RAG
                logger.exception(f"[{user_id}] ❌ Ошибка tool-check: {e}")

            # ═══════════════════════════════════════════════════════════════
            # ВЕТКА A: МОДЕЛЬ ЗАПРОСИЛА ИНСТРУМЕНТ(Ы)
            # ═══════════════════════════════════════════════════════════════
            if first_response and getattr(first_response, "tool_calls", None):
                # retrieval больше не нужен — отменяем чисто
                if not retrieval_task.done():
                    retrieval_task.cancel()
                    try:
                        await retrieval_task
                    except (asyncio.CancelledError, Exception):
                        pass

                yield {"type": "status", "text": "🛠️ Выполняю внешний запрос...\n"}

                tool_messages.append(first_response)

                # Параллельно выполняем все запрошенные инструменты
                results = await asyncio.gather(
                    *(self._exec_tool(tc) for tc in first_response.tool_calls)
                )

                for tc, result in zip(first_response.tool_calls, results):
                    tool_messages.append(
                        ToolMessage(
                            content=json.dumps(result, ensure_ascii=False),
                            tool_call_id=tc["id"],
                        )
                    )

                yield {"type": "status", "text": "🧠 Формирую ответ...\n"}

                # Заменяем system-промпт роутера на боевой QA-промпт
                # — иначе модель отвечает без правил стиля («не предлагай услуг», «кратко»)
                tool_messages[0] = SystemMessage(content=self.config.QA_PROMPT_HYBRID_RU_SYSTEM)

                full = ""
                buffer = ""

                if not self.llm_limiter.is_available():
                    yield {"type": "status", "text": "⏳ Ожидаю свободный слот LLM...\n"}

                try:
                    async with self.llm_limiter.acquire(user_id=user_id):
                        async for chunk in self.llm.astream(tool_messages):
                            if content := chunk.content:
                                full += content
                                buffer += content
                                if len(buffer) > 50 or content in ".!?\n":
                                    yield {"type": "response", "text": buffer}
                                    buffer = ""
                    if buffer:
                        yield {"type": "response", "text": buffer}
                except RateLimitExceeded as e:
                    logger.warning(f"[{user_id}] Rate limit (tool-flow): {e}")
                    yield {"type": "error", "text": str(e)}
                    return
                except Exception as e:
                    logger.exception(f"[{user_id}] ❌ Генерация (tool-flow): {e}")
                    yield {"type": "error", "text": "Произошла ошибка при генерации ответа."}
                    return

                # История: только вопрос + финальный ответ (без ToolMessage и AIMessage-with-tool_calls)
                if full.strip():
                    try:
                        h = self.get_history(user_id)
                        h.add_message(HumanMessage(content=question))
                        h.add_message(AIMessage(content=full))
                    except Exception as e:
                        logger.exception(f"[{user_id}] ❌ Ошибка сохранения: {e}")

                logger.debug(f"[{user_id}] 🎯 Tool-flow: {time.time() - start_time:.2f}с")
                return

            # ═══════════════════════════════════════════════════════════════
            # ВЕТКА B: ОБЫЧНЫЙ RAG-ПОТОК (retrieval уже запущен)
            # ═══════════════════════════════════════════════════════════════
            yield {"type": "status", "text": "📚 Ищу документы...\n"}

            try:
                docs = await retrieval_task
                logger.debug(f"[{user_id}] ✅ Найдено {len(docs)} документов")
            except Exception as e:
                logger.exception(f"[{user_id}] ❌ Ошибка поиска: {e}")
                yield {"type": "error", "text": "Ошибка при поиске документов."}
                return

            if not docs:
                if self.config.MODE == "rag":
                    yield {"type": "info", "text": "Информация не найдена."}
                    return
                logger.info(f"[{user_id}] Нет документов — режим {self.config.MODE}")
                docs = []

            yield {"type": "status", "text": "✅ Документы найдены. Формирую ответ...\n"}

            try:
                context, truncated_history, history_summary = self._build_context(docs, hist)
                logger.info(
                    f"[{user_id}] retrieval: docs={len(docs)}, "
                    f"context_chars={len(context)}, "
                    f"history_msgs={len(truncated_history)}, "
                    f"has_summary={bool(history_summary)}, "
                    f"sources={[d.metadata.get('source') for d in docs]}"
                )
            except Exception as e:
                logger.exception(f"[{user_id}] ❌ Ошибка контекста: {e}")
                yield {"type": "error", "text": "Ошибка при подготовке контекста."}
                return

            if not context and self.config.MODE == "rag":
                yield {"type": "error", "text": "Информация не найдена."}
                return

            logger.info(f"[{user_id}] ПОЛНЫЙ КОНТЕКСТ В LLM:\n{context}")

            messages = self._build_messages(
                mode=self.config.MODE,
                question=question,
                context=context,
                truncated_history=truncated_history,
                history_summary=history_summary,
            )

            full = ""
            buffer = ""
            llm_start = time.time()
            first_token_received = False

            if not self.llm_limiter.is_available():
                yield {"type": "status", "text": "⏳ Ожидаю свободный слот LLM...\n"}

            try:
                async with self.llm_limiter.acquire(user_id=user_id):
                    async for chunk in self.llm.astream(messages):
                        if content := chunk.content:
                            if not first_token_received:
                                logger.debug(
                                    f"[{user_id}] ⚡ Первый токен за "
                                    f"{time.time() - llm_start:.2f}с"
                                )
                                first_token_received = True
                                yield {"type": "status", "text": "🧠 Генерирую ответ...\n"}
                            full += content
                            buffer += content
                            if len(buffer) > 50 or content in ".!?\n":
                                yield {"type": "response", "text": buffer}
                                buffer = ""
                if buffer:
                    yield {"type": "response", "text": buffer}
            except RateLimitExceeded as e:
                logger.warning(f"[{user_id}] Rate limit: {e}")
                yield {"type": "error", "text": str(e)}
                return
            except Exception as e:
                logger.exception(f"[{user_id}] ❌ Ошибка генерации: {e}")
                yield {"type": "error", "text": "Произошла ошибка при генерации ответа."}
                return

            if full.strip():
                generated_tokens = count_tokens(full)
                gen_time = time.time() - llm_start
                logger.info(
                    f"[{user_id}] 📊 генерация: {generated_tokens} токенов за {gen_time:.2f}с "
                    f"(~{generated_tokens / gen_time:.0f} tok/s), chars={len(full)}"
                )
                try:
                    h = self.get_history(user_id)
                    h.add_message(HumanMessage(content=question))
                    h.add_message(AIMessage(content=full))
                except Exception as e:
                    logger.exception(f"[{user_id}] ❌ Ошибка сохранения: {e}")

            logger.debug(f"[{user_id}] 🎯 RAG-flow: {time.time() - start_time:.2f}с")


        self.qa_chain_with_history = generate_answer_stream
        logger.info("QA-генератор со стримингом и историей настроен")


    async def close(self) -> None:
        await self.milvus.close()

        if self._embeddings is not None and hasattr(self._embeddings, "client"):
            try:
                await self._embeddings.client.aclose()
            except Exception as e:
                logger.warning("Ошибка при закрытии embeddings: %s", e)

        if self._llm is not None:
            # ChatOpenAI использует httpx.AsyncClient внутри _async_client
            for attr in ("_async_client", "client"):
                c = getattr(self._llm, attr, None)
                if c is not None and hasattr(c, "aclose"):
                    try:
                        await c.aclose()
                    except Exception as e:
                        logger.warning("Ошибка при закрытии LLM-клиента: %s", e)
                    break

        if self.embedding_cache is not None:
            try:
                self.embedding_cache.client.close()
            except Exception:
                pass

        if self.collection_manager is not None:
            try:
                self.collection_manager.redis.close()
            except Exception:
                pass

        if self.history is not None:
            try:
                self.history.redis_client.close()
            except Exception:
                pass

        logger.info("RAGCore: все ресурсы закрыты")