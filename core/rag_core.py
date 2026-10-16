import asyncio
import logging
import time
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from transformers import TFOpenAIGPTDoubleHeadsModel

from config.rag_config import RAGConfig
from core.chat_history import RedisChatHistory
from core.document_loader import load_documents_from_directory
from core.milvus_manager import MilvusManager
from core.polite_detector import PoliteDetector
from core.splitters import SplitterManager
from core.utils import count_tokens, truncate_text_by_tokens
from core.embedding_cache import RedisEmbeddingCache
from core.collection_manager import CollectionManager

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
        # self.config = config or RAGConfig()
        self.config = config or RAGConfig()

        # Менеджер истории диалога (хранит переписку)
        self.history = RedisChatHistory(
            session_id = "default",  # потом можно передавать разный session_id от клиента
            host = self.config.REDIS_HOST,
            port = self.config.REDIS_PORT,
            ttl_days = self.config.HISTORY_TTL_DAYS,
            max_messages = self.config.MAX_HISTORY_MESSAGES
        )

        # Подключение к Milvus (векторное хранилище)
        self.milvus = MilvusManager(
            uri = self.config.MILVUS_URI,
            collection_name = self.config.COLLECTION_NAME,
            recreate = self.config.RECREATE_COLLECTION
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
            host = self.config.REDIS_HOST,
            port = self.config.REDIS_PORT,
            ttl = self.config.REDIS_TTL
        )

        # менеджер коллекций
        self.collection_manager = CollectionManager(core=self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def __repr__(self):
        return f"<RAGConfig mode={self.config.MODE}, collection={self.config.COLLECTION_NAME}>"

    def __str__(self):
        return self.__repr__()

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
                model = self.config.EMBEDDING_NAME,
                base_url = self.config.EMBEDDING_BASE_URL,
                timeout = 60,
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
                model = self.config.LLM_NAME,
                api_key = "EMPTY",
                base_url =self.config.LLM_BASE_URL,
                max_tokens = self.config.LLM_MAX_TOKEN,
                temperature = self.config.LLM_TEMPERATURE,
                streaming = True,
                extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
            )
        return self._llm

    def get_history(self, session_id: str) -> RedisChatHistory:
        """
        Выдаем новый объект под конкретный session_id.
        """
        return RedisChatHistory(
            session_id = session_id,
            host = self.config.REDIS_HOST,
            port = self.config.REDIS_PORT,
            ttl_days = self.config.HISTORY_TTL_DAYS,
            max_messages=self.config.MAX_HISTORY_MESSAGES
        )

    def load_documents(self, directory: Optional[str] = None) -> List[Document]:
        """
        Загружает документы из указанной директории.
        По умолчанию — директория config.DATA_DIR.
        """
        directory = directory or self.config.DATA_DIR
        return load_documents_from_directory(directory)

    async def setup_collection(self, collection_name: str, documents: List[Document]):
        """
        Создаёт коллекцию Milvus с именем collection_name
        и индексирует туда документы.
        """

        # Создаём отдельный экземпляр менеджера Milvus под коллекцию
        milvus = MilvusManager(
            uri=self.config.MILVUS_URI,
            collection_name=collection_name,
            recreate=self.config.RECREATE_COLLECTION
        )

        # Сплитинг по чанкам
        processed = self.splitters.split_by_headers(documents)
        if not processed:
            return

        texts = [d.page_content for d in processed]
        titles = [d.metadata.get("title") for d in processed]
        bm25_t = [d.metadata.get("bm25_text") for d in processed]
        sources = [d.metadata.get("source", "N/A") for d in processed]
        hashes = [d.metadata.get("hash") for d in processed]

        # Эмбеддинги
        dense_vectors = await self.embeddings.embed_documents(texts)
        dim = len(dense_vectors[0])

        await milvus.create_collection_if_needed(dense_dim=dim)

        # Формируем записи
        rows = [
            {
                "text": t,
                "title": tit,
                "bm25_text": b25,
                "source": s,
                "hash": h,
                "dense_vector": dv,
            }
            for t, tit, b25, s, dv, h in zip(texts, titles, bm25_t, sources, dense_vectors, hashes)
        ]

        unique = await milvus.ensure_not_duplicate_rows(rows)
        for i in range(0, len(unique), self.config.INDEX_BATCH_SIZE):
            await milvus.insert_records(unique[i:i + self.config.INDEX_BATCH_SIZE])

        await milvus.client.load_collection(collection_name)
        logger.info("Коллекция %s загружена в память", collection_name)

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
        title = [d.metadata.get("title") for d in processed]
        bm25_text = [d.metadata.get("bm25_text") for d in processed]
        sources = [d.metadata.get("source","N/A") for d in processed]
        hashes = [d.metadata.get("hash") for d in processed]

        # Генерация эмбеддингов для каждого чанка
        try:
            dense_vectors = await self.embeddings.embed_documents(texts)
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
        rows = [
            {"text": t,
             "title": tit,
             "bm25_text": bm25,
             "source": s,
             "hash": h or "",
             "dense_vector": dv} for t, tit, bm25, s, dv, h in zip(texts, title, bm25_text, sources, dense_vectors, hashes)]

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

    async def _get_embedding_cached(self, query: str) -> Optional[List[float]]:
        """проверка кэша из Redis"""
        cached = self.embedding_cache.get(query)
        if cached is not None:
            return cached
        try:
            # Генерируем эмбеддинг для запроса
            emb = await self.embeddings.embed_query(query)
            self.embedding_cache.set(query, emb)
            return emb
        except Exception as e:
            logger.warning("Ошибка эмбеддинга запроса: %s", e)
            return None

    def create_retriever(self, k: Optional[int] = None, fetch_k: Optional[int] = None) -> None:
        """
        Создаёт асинхронный ретривер:
        - ищет документы в Milvus
        - использует кэширование эмбеддингов запросов
        """
        k = k or self.config.K
        fetch_k = fetch_k or self.config.FETCH_K

        async def routed_retrieve(query: str, session_id: str | None = None) -> List[Document]:
            if not query.strip():
                return []

            sid = session_id or "global"

            # Загружаем последнюю коллекцию из Redis (по conversation_id)
            last = self.collection_manager.load_last_used_collection(session_id=sid)

            # Если follow-up и есть последняя коллекция — используем её
            if last and self.collection_manager.is_followup_query(query):
                collection_name = last
                logger.info("Follow-up запрос (sid=%s) — использую прошлую коллекцию: %s",  sid, collection_name)
            else:
                # Иначе обычный routing
                collection_name = await self.collection_manager.route_query(query)
                logger.info("RAW QUERY: %s", query)
                logger.info("Роутер выбрал коллекцию: %s", collection_name)
                # Сохраняем выбор коллекции в Redis (по conversation_id)
                self.collection_manager.save_last_used_collection(collection_name, session_id=sid)


            # # Определяем коллекцию
            # collection_name = await self.collection_manager.route_query(query)
            # if not collection_name:
            #     logger.warning("Не удалось определить коллекцию для запроса")
            #     return []
            # logger.info("Роутер выбрал коллекцию: %s", collection_name)

            # Проверяем и загружаем коллекцию
            if not await self.milvus.client.has_collection(collection_name):
                logger.warning("Коллекция %s не найдена", collection_name)
                return []
            await self.milvus.ensure_collection_loaded(collection_name)

            # Эмбеддинг
            dense = await self._get_embedding_cached(query)
            if dense is None:
                return []

            # Поиск
            return await self.milvus.hybrid_search(
                query_text=query,
                query_dense=dense,
                fetch_k=fetch_k,
                top_k=k,
                collection_name=collection_name,
                reranker_endpoint=self.config.RERANKER_BASE_URL
            )

        async def retrieve(query: str) -> List[Document]:
            if not query.strip():
                return []

            # Проверяем, что коллекция существует
            if not await self.milvus.client.has_collection(self.milvus.collection_name):
                logger.warning("Коллекция %s не найдена", self.milvus.collection_name)
                return []

            # Подстраховка: загружаем коллекцию перед поиском
            await self.milvus.ensure_collection_loaded(self.milvus.collection_name)

            dense = await self._get_embedding_cached(query)

            # Делаем гибридный поиск (по вектору + тексту)
            return await self.milvus.hybrid_search(
                query_text = query,
                query_dense = dense,
                fetch_k = fetch_k,
                top_k = k,
                reranker_endpoint = self.config.RERANKER_BASE_URL
            )

        # self.retriever = retrieve
        # logger.info("Ретривер создан (k=%d, fetch_k=%d)", k, fetch_k)
        self.retriever = routed_retrieve
        logger.info("Ретривер с маршрутизацией создан (k=%d, fetch_k=%d)", k, fetch_k)

    def _build_context(self, docs: List[Document], session_id: str) -> tuple[str, str]:
        """
        Формирует контекст для LLM с учетом:
          - истории чата (суммаризация старых сообщений при необходимости)
          - контекста документов
          - лимита токенов MAX_CONTEXT_TOKENS
          - детерминированного порядка документов
        Возвращает (context_documents, history_text)
        """

        # --- История чата ---
        hist = self.get_history(session_id).get_messages()
        truncated_history = []
        history_tokens = 0

        # Берём последние сообщения (с конца), чтобы последние важные сообщения вошли
        for m in reversed(hist):
            msg_text = f"{m.type.capitalize()}: {m.content}"
            tks = count_tokens(msg_text)
            if history_tokens + tks > self.config.MAX_HISTORY_TOKENS:
                break
            truncated_history.insert(0, msg_text)  # вставляем в начало
            history_tokens += tks

        # Старые сообщения, которые не вошли, суммируем в одно резюме
        num_skipped = len(hist) - len(truncated_history)
        if num_skipped > 0:
            old_msgs = hist[:num_skipped]
            old_text = "\n".join(
                f"{m.type.capitalize()}: {truncate_text_by_tokens(m.content, 200)}"
                for m in old_msgs
            )
            truncated_history.insert(0, f"[Резюме предыдущих сообщений]: {old_text}")
            history_tokens += count_tokens(truncated_history[0])

        history_text = "\n".join(truncated_history)

        # --- Доступные токены для документов ---
        available = self.config.MAX_CONTEXT_TOKENS - self.config.RESERVED_FOR_COMPLETION \
                    - history_tokens - self.config.RESERVED_FOR_OVERHEAD
        available = max(0, available)
        if available <= 0:
            return "", history_text

        # --- Детерминированная сортировка документов ---
        # Сначала по score (убывание), затем по source (алфавит) для стабильного порядка
        docs_sorted = sorted(
            docs,
            key=lambda d: (-d.metadata.get("score", 0), d.metadata.get("source", ""))
        )

        # --- Формирование контекста документов ---
        parts, used = [], 0
        for d in docs_sorted:
            text = f"[Источник: {d.metadata.get('source','N/A')}] {d.page_content.strip()}\n"
            tks = count_tokens(text)
            if used + tks > available:
                remain = available - used
                if remain > self.config.MIN_DOC_TOKEN:
                    parts.append(truncate_text_by_tokens(text, remain))
                break
            parts.append(text)
            used += tks

        context = "".join(parts).strip()
        return context, history_text

    def _build_prompt(self, mode: str, question: str, context: str, history_text: str) -> str:
        """
        конструктор промптов в зависимости от режима работы RAG
        """
        if mode == "rag":
            return self.config.QA_PROMPT_RAG_EN.format(context=context, chat_history=history_text, question=question)
        elif mode == "hybrid":
            return self.config.QA_PROMPT_HYBRID_RU.format(context=context, chat_history=history_text, question=question)
        elif mode == "llm_only":
            return self.config.QA_PROMPT_LLM_ONLY_EN.format(chat_history=history_text, question=question)
        else:
            raise ValueError(f"Неизвестный режим: {mode}")

    def _build_messages(self, mode: str, question: str, context: str, history: List[BaseMessage]) -> List[BaseMessage]:
        """
        Собирает структурированный список сообщений для ChatModel.
        """
        #TODO
        # добавить зависимость от режима работы RAG

        # Системное сообщение — ТОЛЬКО правила (без {context}!)

        messages: List[BaseMessage] = [SystemMessage(content=self.config.QA_PROMPT_HYBRID_RU_SYSTEM)]

        # Добавляем контекст как первое user-сообщение
        if context.strip():
            messages.append(HumanMessage(content=f"Контекст:\n{context}"))

        # Добавляем историю
        messages.extend(history)

        # Добавляем текущий вопрос
        messages.append(HumanMessage(content=question))

        return messages

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
            start_time = time.time()
            logger.debug(f"[{session_id}] 🚀 Начал обработку вопроса: {question[:100]}...")

            if not question.strip():
                logger.warning(f"[{session_id}] Получен пустой вопрос")
                yield {
                    "type": "response",  # или "message"
                    "text": "Пожалуйста, задайте вопрос."
                }
                return

            # Проверяем вежливые/служебные фразы
            if PoliteDetector.is_polite(question):
                yield {
                    "type": "response",
                    "text": PoliteDetector.reply(question)
                }
                return

                # Сразу отдаём первый токен — "отклик"
            yield {
                "type": "status",  # или "message"
                "text": "📚⏳ Думаю над вашим вопросом...\n"
            }

            # Запускаем асинхронно генерацию эмбеддинга и загрузку истории
            embedding_start = time.time()
            embedding_task = asyncio.create_task(self._get_embedding_cached(question))
            history_task = asyncio.create_task(
                asyncio.to_thread(self.get_history(session_id).get_messages)
            )
            logger.debug(f"[{session_id}] Запустил параллельные задачи: эмбеддинг + история")

            # Пока ждём эмбеддинг — можно начать формировать часть промпта без контекста
            yield {
                "type": "status",  # или "message"
                "text": "📚 Ищу документы...\n"
            }

            # Ждём эмбеддинг
            try:
                dense = await embedding_task
                embedding_time = time.time() - embedding_start
                logger.debug(f"[{session_id}] ✅ Эмбеддинг получен за {embedding_time:.2f} сек")
            except Exception as e:
                logger.exception(f"[{session_id}] ❌ Ошибка при получении эмбеддинга: {e}")
                yield {
                    "type": "error",  # или "message"
                    "text": "Ошибка при обработке запроса."
                }
                return

            if not dense:
                logger.warning(f"[{session_id}] Эмбеддинг пустой или None")
                yield {
                    "type": "error",  # или "message"
                    "text": "Ошибка при обработке запроса."
                }
                return

            # Поиск документов (можно тоже обернуть в таск, если хочешь параллелить с чем-то ещё)
            search_start = time.time()
            try:
                docs = await self.retriever(question, session_id)
                search_time = time.time() - search_start
                logger.debug(f"[{session_id}] ✅ Найдено {len(docs)} документов за {search_time:.2f} сек")
                # logger.info(f"BLA BLA BLA {docs}")
            except Exception as e:
                logger.exception(f"[{session_id}] ❌ Ошибка поиска: {e}")
                yield {
                    "type": "error",  # или "message"
                    "text": "Ошибка при поиске документов."
                }
                return

            if not docs:
                logger.debug(f"[{session_id}] ❗ Документы не найдены")
                yield {
                    "type": "info",  # или "message"
                    "text": "Информация не найдена."
                }
                return

            yield {
                "type": "status",  # или "message"
                "text": "✅ Документы найдены. Формирую ответ...\n"
            }

            # Ждём историю
            try:
                hist = await history_task
                hist_time = time.time() - embedding_start  # с момента запуска задачи
                logger.debug(f"[{session_id}] ✅ История загружена за {hist_time:.2f} сек, сообщений: {len(hist)}")
            except Exception as e:
                logger.exception(f"[{session_id}] ❌ Ошибка загрузки истории: {e}")
                hist = []

            history_text = "\n".join(f"{m.type.capitalize()}: {m.content}" for m in hist)

            # Формируем контекст
            context_start = time.time()
            try:
                context, _ = self._build_context(docs, session_id)
                context_time = time.time() - context_start
                context_len = len(context)
                token_count = count_tokens(context) if context else 0
                logger.debug(
                    f"[{session_id}] ✅ Контекст сформирован за {context_time:.2f} сек, токенов: {token_count}, символов: {context_len}")
                # logger.debug(
                #     f"[{session_id}] ✅ Контекст: {context}")

            except Exception as e:
                logger.exception(f"[{session_id}] ❌ Ошибка формирования контекста: {e}")
                yield {
                    "type": "error",  # или "message"
                    "text": "Ошибка при подготовке контекста."
                }
                return

            if not context:
                logger.warning(f"[{session_id}] Контекст пуст после построения")
                yield {
                    "type": "error",  # или "message"
                    "text": "Информация не найдена."
                }
                return

            # Генерация ответа LLM
            history = self.get_history(session_id).get_messages()  # это список BaseMessage
            messages = self._build_messages(
                mode=self.config.MODE,
                question=question,
                context=context,
                history=history
            )

            # prompt = self._build_prompt(
            #     mode=self.config.MODE,
            #     question=question,
            #     context=context,
            #     history_text=history_text
            # )


            full = ""
            buffer = ""
            llm_start = time.time()

            try:
                logger.debug(f"[{session_id}] 🧠 Начинаю генерацию LLM...")
                first_token_received = False
                token_count = 0

                async for chunk in self.llm.astream(messages):
                # async for chunk in self.llm.astream([{"role": "user", "content": prompt}]):
                    if content := chunk.content:
                        if not first_token_received:
                            first_token_time = time.time() - llm_start
                            logger.debug(f"[{session_id}] ⚡ Первый токен LLM получен за {first_token_time:.2f} сек")
                            first_token_received = True
                            yield {
                                "type": "status",  # или "message"
                                "text": "🧠 Генерирую ответ...\n"
                            }

                        full += content
                        token_count += 1

                        # Отправляем, только если накопилось достаточно или встретили знак препинания
                        buffer += content
                        if len(buffer) > 50 or content in ".!?\n":
                            yield {
                                "type": "response",  # или "message"
                                "text": buffer
                            }
                            buffer = ""

                # Скидываем остаток
                if buffer:
                    yield {
                        "type": "response",  # или "message"
                        "text": buffer
                    }

                total_llm_time = time.time() - llm_start
                logger.debug(f"[{session_id}] ✅ LLM сгенерировал {token_count} токенов за {total_llm_time:.2f} сек")

                # --- Этап 7: Сохранение в историю ---
                save_start = time.time()
                try:
                    history = self.get_history(session_id)
                    history.add_message(HumanMessage(content=question))
                    history.add_message(AIMessage(content=full))
                    save_time = time.time() - save_start
                    logger.debug(f"[{session_id}] 💾 Ответ сохранён в историю за {save_time:.2f} сек")
                except Exception as e:
                    logger.exception(f"[{session_id}] ❌ Ошибка сохранения в историю: {e}")

            except Exception as e:
                logger.exception(f"[{session_id}] ❌ Ошибка при генерации ответа: {e}")
                yield {
                    "type": "error",  # или "message"
                    "text": "Произошла ошибка при генерации ответа."
                }

            total_time = time.time() - start_time
            logger.debug(f"[{session_id}] 🎯 Полное время обработки: {total_time:.2f} сек")

        self.qa_chain_with_history = generate_answer_stream
        logger.info("QA-генератор со стримингом и историей настроен")

    async def close(self) -> None:
        # Корректно закрываем соединение с Milvus
        await self.milvus.close()