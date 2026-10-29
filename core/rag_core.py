# core/rag_core.py

# Copyright 2025 Artyom Yudin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
                frequency_penalty=self.config.LLM_FREQUENCY_PENALTY,
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


    def _count_msg_tokens(self, m: BaseMessage) -> int:
        """Токены сообщения с учётом tool_calls (для AIMessage)."""
        base = count_tokens(m.content or "")
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            base += count_tokens(json.dumps(tool_calls, ensure_ascii=False))
        return base


    def _log_prompt_stats(self, user_id: str, messages: List[BaseMessage], branch: str):
        """Раскладка промпта по ролям: system / history / tool / question."""
        if not messages:
            return

        system_tok = 0
        history_tok = 0
        tool_tok = 0
        question_tok = 0

        # Находим последний HumanMessage — это текущий вопрос
        last_human_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].type == "human":
                last_human_idx = i
                break

        for i, m in enumerate(messages):
            n = self._count_msg_tokens(m)
            if m.type == "system":
                system_tok += n
            elif m.type == "tool":
                tool_tok += n
            elif i == last_human_idx:
                question_tok += n
            elif m.type == "ai" and getattr(m, "tool_calls", None):
                tool_tok += n
            else:
                history_tok += n

        total = system_tok + history_tok + tool_tok + question_tok

        logger.info(
            f"[{user_id}] 📥 ПРОМПТ ({branch}): total={total} tok | "
            f"system={system_tok}, history={history_tok}, "
            f"tool={tool_tok}, question={question_tok} | msgs={len(messages)}"
        )


    def _log_full_messages(self, user_id: str, messages: List[BaseMessage], branch: str):
        """Полный дамп всех сообщений — для отладки."""
        if not logger.isEnabledFor(logging.DEBUG):
            return  # ★ дампим только в DEBUG
        if not messages:
            return
        lines = [f"[{user_id}] 📜 ПОЛНЫЙ ПРОМПТ ({branch}) — {len(messages)} сообщений:"]
        for i, m in enumerate(messages):
            role = m.type.upper()
            content = (m.content or "").strip()
            tool_calls = getattr(m, "tool_calls", None)
            tool_call_id = getattr(m, "tool_call_id", None)

            if tool_calls:
                tc_str = ", ".join(
                    f"{tc['name']}({tc.get('args', {})})" for tc in tool_calls
                )
                lines.append(f"  [{i}] {role} (tool_calls: {tc_str}): {content[:150]!r}")
            elif tool_call_id:
                preview = content[:300] if len(content) > 300 else content
                lines.append(f"  [{i}] {role} (id={tool_call_id[:20]}...): {preview}")
            else:
                preview = content[:500] if len(content) > 500 else content
                suffix = "..." if len(content) > 500 else ""
                lines.append(f"  [{i}] {role}: {preview}{suffix}")
        logger.info("\n".join(lines))


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
                "chunk_index": idx,  # позиция в документе
            }
            for idx, (t, tit, bm25, s, h, dv) in enumerate(zip(
                texts, titles, bm25_texts, sources, hashes, dense_vectors
            ))
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

    def _truncate_history_safe(
            self,
            history: List[BaseMessage],
            max_tokens: int,
    ) -> List[BaseMessage]:
        """
        Обрезает историю по токенам, НИКОГДА не разрывая tool-flow.
        Точки обрезки — только перед HumanMessage.

        Уровни (по убыванию):
          1. Максимальная обрезка: с самой ранней safe-point.
          2. Меньшая обрезка: с более поздних safe-point.
          3. Fallback: последний ход пользователя (Human + AI), даже если
             слегка превышает бюджет — лучше чуть больше, чем потерять контекст.
          4. Последний резерв: только последний HumanMessage.
        """
        if not history:
            return []

        safe_points = [i for i, m in enumerate(history) if m.type == "human"]
        if not safe_points:
            return history

        def _total(msgs):
            return sum(
                count_tokens(f"{m.type.capitalize()}: {m.content}")
                for m in msgs
            )

        # ── УРОВНИ 1–2: пробуем обрезать по safe-points (от ранней к поздней) ──
        # ★ БЕЗ [len(history)] в конце — иначе цикл всегда вернёт []
        for idx in safe_points:
            candidate = history[idx:]
            if _total(candidate) <= max_tokens:
                return candidate

        # ── УРОВЕНЬ 3: FALLBACK — последний ход пользователя ──
        # Ни одна safe-point не влезла. Возвращаем хотя бы последний Human
        # + следующий за ним AI (один полный ход), даже если слегка
        # превышает max_tokens — иначе модель потеряет весь контекст.
        last_human_idx = safe_points[-1]
        tail = history[last_human_idx:]

        if len(tail) >= 2:
            one_turn = tail[:2]  # Human + AI (ответ ассистента)
            if _total(one_turn) <= max_tokens:
                return one_turn
            # Даже один ход не влезает — вернём как есть.
            # Превышение бюджета лучше, чем пустая история для tool-check.
            logger.warning(
                "История переполнена: последний ход (%d tok) > бюджет (%d tok). "
                "Возвращаю как есть.",
                _total(one_turn), max_tokens,
            )
            return one_turn

        # ── УРОВЕНЬ 4: совсем крайний случай — только Human ──
        return tail[:1]

    # ─────────────────────────────────────────────────────────────
    # Контекст и сообщения
    # ─────────────────────────────────────────────────────────────

    def _build_context(
            self,
            docs: List[Document],
            history: List[BaseMessage],
    ) -> tuple[str, List[BaseMessage], str]:
        """
        Формирует контекст для LLM.

        Отличия от старой версии:
          - документы группируются по source (документ = источник = логическая единица)
          - источник включается ЦЕЛИКОМ или пропускается (никогда не режется посередине)
          - источники упорядочены по top-score своих чанков
          - fallback: если ни один источник не влез — старое поведение с truncate
            на top-чанках, чтобы модель не осталась с пустым контекстом
        """
        # ── 1. История (без изменений) ───────────────────────────────
        truncated_history = self._truncate_history_safe(
            history, self.config.MAX_HISTORY_TOKENS
        )
        history_tokens = sum(
            count_tokens(f"{m.type.capitalize()}: {m.content}")
            for m in truncated_history
        )

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

        # ── 2. Бюджет на документы ───────────────────────────────────
        available = (
                self.config.MAX_CONTEXT_TOKENS
                - self.config.RESERVED_FOR_COMPLETION
                - history_tokens
                - self.config.RESERVED_FOR_OVERHEAD
        )
        available = max(0, available)
        if available <= 0:
            return "", truncated_history, history_summary

        # ── 3. Группировка по source ─────────────────────────────────
        by_source: dict[str, list[Document]] = {}
        for d in docs:
            src = d.metadata.get("source", "N/A")
            by_source.setdefault(src, []).append(d)

        # ── 4. Сортировка источников по их top-score ─────────────────
        source_groups = sorted(
            by_source.items(),
            key=lambda kv: max(d.metadata.get("score", 0.0) for d in kv[1]),
            reverse=True,
        )
        # Внутри источника — чанки тоже по score, для детерминизма
        for _, chunks in source_groups:
            # Expanded-чанки (включая top-chunk, у которого теперь есть chunk_index)
            #   сортируются по позиции в документе. Для обычных (не-expanded) —
            #   по score, чтобы top-чанки были сверху.
            any_expanded = any(d.metadata.get("expanded") for d in chunks)
            if any_expanded:
                chunks.sort(key=lambda d: d.metadata.get("chunk_index", 999))
            else:
                chunks.sort(key=lambda d: -d.metadata.get("score", 0.0))

        # ── 5. Сборка: источник целиком или пропуск ──────────────────
        parts: list[str] = []
        used = 0
        included: list[tuple[str, int, int]] = []  # (source, chunks, tokens)
        skipped: list[tuple[str, int]] = []  # (source, tokens)

        for src, chunks in source_groups:
            # Заголовок один раз + все чанки подряд
            body = "\n".join(d.page_content.strip() for d in chunks)
            src_text = f"[Источник: {src}]\n{body}\n"
            src_tokens = count_tokens(src_text)

            if used + src_tokens > available:
                skipped.append((src, src_tokens))
                continue

            parts.append(src_text)
            used += src_tokens
            included.append((src, len(chunks), src_tokens))

        # ── 6. FALLBACK: ни один источник не влез целиком ────────────
        #    Старое поведение: top-чанки с truncate.
        #    Лучше частичный контекст, чем пустой.
        if not parts and docs:
            logger.warning(
                "Context: ни один источник не влез целиком (available=%d tok) — "
                "фолбэк на top-чанки с truncate",
                available,
            )
            docs_sorted = sorted(docs, key=lambda d: -d.metadata.get("score", 0.0))
            for d in docs_sorted:
                text = (
                    f"[Источник: {d.metadata.get('source', 'N/A')}] "
                    f"{d.page_content.strip()}\n"
                )
                tks = count_tokens(text)
                if used + tks > available:
                    remain = available - used
                    if remain > self.config.MIN_DOC_TOKEN:
                        parts.append(truncate_text_by_tokens(text, remain))
                    break
                parts.append(text)
                used += tks

        context = "".join(parts).strip()

        # ── 7. Диагностика ───────────────────────────────────────────
        if included or skipped:
            logger.info(
                "Context: included=%d [%s] | skipped=%d [%s] | used=%d/%d tok",
                len(included),
                ", ".join(f"{s}:{n}ch/{t}t" for s, n, t in included) or "—",
                len(skipped),
                ", ".join(f"{s}:{t}t" for s, t in skipped) or "—",
                used,
                available,
            )

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
            # Динамический token-aware срез истории диалога
            # ═══════════════════════════════════════════════════════════════

            max_tool_history_tokens = self.config.MAX_TOOL_HISTORY_TOKENS
            truncated_tool_hist = self._truncate_history_safe(hist, max_tool_history_tokens)
            tool_hist_tokens = sum(
                count_tokens(f"{m.type.capitalize()}: {m.content}")
                for m in truncated_tool_hist
            )

            logger.debug(
                f"[{user_id}] 🧠 Tool-check история сжата: {len(truncated_tool_hist)}/"
                f"{len(hist)} сообщений ({tool_hist_tokens} токенов)"
            )

            tool_messages: List[BaseMessage] = [
                SystemMessage(content=self.config.QA_PROMPT_TOOL_CHECK_SYSTEM),
                *truncated_tool_hist,
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
                # Ждём retrieval (он уже стартовал параллельно). Если запрос
                # смешанный (RAG + tool) — его результат пойдёт в контекст LLM.
                # Если чистый tool — потеряем 0–500мс, что приемлемо.
                retrieval_docs: List[Document] = []
                try:
                    retrieval_docs = await asyncio.wait_for(
                        retrieval_task, timeout=5.0
                    )
                    logger.info(
                        "[%s] tool-flow: retrieval вернул %d docs (для RAG-контекста)",
                        user_id, len(retrieval_docs),
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "[%s] tool-flow: retrieval не успел за 5с — работаю без RAG",
                        user_id,
                    )
                    if not retrieval_task.done():
                        retrieval_task.cancel()
                except Exception as e:
                    logger.warning(
                        "[%s] tool-flow: retrieval ошибка (%s) — работаю без RAG",
                        user_id, e,
                    )

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
                # Если RAG что-то нашёл — добавляем контекст в tool-flow.
                # Без этого смешанные запросы («камера + погода») теряют RAG-часть.
                if retrieval_docs:
                    try:
                        rag_context, _, _ = self._build_context(retrieval_docs, hist)
                        if rag_context.strip():
                            # Вставляем как отдельный system сразу после основного —
                            # модель видит и правила, и факты из документов
                            tool_messages.insert(
                                1,
                                SystemMessage(content=(
                                    "Дополнительный контекст из внутренних документов "
                                    "(относится к части вопроса про инфраструктуру):\n"
                                    f"{rag_context}"
                                )),
                            )
                            logger.info(
                                "[%s] tool-flow: RAG-контекст добавлен (%d chars)",
                                user_id, len(rag_context),
                            )
                        else:
                            logger.info(
                                "[%s] tool-flow: RAG-контекст пуст (нет релевантных документов)",
                                user_id,
                            )
                    except Exception as e:
                        logger.warning(
                            "[%s] tool-flow: не удалось построить RAG-контекст: %s",
                            user_id, e,
                        )

                # ═══════════════════════════════════════════════════════════════
                # ДИАГНОСТИКА ПРОМПТА (до вызова LLM)
                # ═══════════════════════════════════════════════════════════════
                self._log_prompt_stats(user_id, tool_messages, branch="tool-flow")
                self._log_full_messages(user_id, tool_messages, branch="tool-flow")

                # Валидация tool-flow: нет висячих tool_call_id
                _open_ids = set()
                for m in tool_messages:
                    if m.type == "ai" and getattr(m, "tool_calls", None):
                        for tc in m.tool_calls:
                            _open_ids.add(tc["id"])
                    elif m.type == "tool" and hasattr(m, "tool_call_id"):
                        _open_ids.discard(m.tool_call_id)
                if _open_ids:
                    logger.error(
                        f"[{user_id}] ⚠️ ВАЛИДАЦИЯ: {len(_open_ids)} tool_call_id без ответа — "
                        f"API вернёт 400. IDs: {_open_ids}"
                    )

                full = ""
                buffer = ""
                llm_start = time.time()  # ★
                ttft_seconds = 0.0  # ★
                first_token_received = False  # ★

                if not self.llm_limiter.is_available():
                    yield {"type": "status", "text": "⏳ Ожидаю свободный слот LLM...\n"}

                try:
                    async with self.llm_limiter.acquire(user_id=user_id):
                        async for chunk in self.llm.astream(tool_messages):
                            if content := chunk.content:
                                if not first_token_received:
                                    ttft_seconds = time.time() - llm_start  # ★ сохраняем
                                    first_token_received = True
                                    logger.debug(
                                        f"[{user_id}] ⚡ [tool-flow] Первый токен за {ttft_seconds:.2f}с"
                                    )
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

                # ═══════════════════════════════════════════════════════════════
                # МЕТРИКИ TOOL-FLOW (после LLM)
                # ═══════════════════════════════════════════════════════════════
                if full.strip():
                    generated_tokens = count_tokens(full)
                    total_time = time.time() - start_time
                    gen_time = time.time() - llm_start
                    decode_time = max(gen_time - ttft_seconds, 0.001)
                    decode_tps = generated_tokens / decode_time

                    logger.info(
                        f"[{user_id}] 📊 TOOL-FLOW: {generated_tokens} tok | "
                        f"tools={[tc['name'] for tc in first_response.tool_calls]} | "
                        f"TTFT={ttft_seconds:.2f}s | decode={decode_time:.2f}s | "
                        f"decode_tok/s={decode_tps:.0f} | tool-flow={total_time:.2f}s"
                    )

                    try:
                        h = self.get_history(user_id)
                        h.add_message(HumanMessage(content=question))
                        h.add_message(first_response)
                        for tc, result in zip(first_response.tool_calls, results):
                            h.add_message(ToolMessage(
                                content=json.dumps(result, ensure_ascii=False),
                                tool_call_id=tc["id"],
                            ))
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

            # ═══════════════════════════════════════════════════════════════
            # ДИАГНОСТИКА ПРОМПТА (до вызова LLM)
            # ═══════════════════════════════════════════════════════════════
            self._log_prompt_stats(user_id, messages, branch="RAG")
            self._log_full_messages(user_id, messages, branch="RAG")

            full = ""
            buffer = ""
            llm_start = time.time()  # ★
            ttft_seconds = 0.0  # ★ — это инициализация, а не «сохранение в цикле»
            first_token_received = False

            if not self.llm_limiter.is_available():
                yield {"type": "status", "text": "⏳ Ожидаю свободный слот LLM...\n"}

            try:
                async with self.llm_limiter.acquire(user_id=user_id):
                    async for chunk in self.llm.astream(messages):
                        if content := chunk.content:
                            if not first_token_received:
                                ttft_seconds = time.time() - llm_start  # ★ сохраняем
                                first_token_received = True
                                logger.debug(
                                    f"[{user_id}] ⚡ Первый токен за {ttft_seconds:.2f}с"
                                )
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

            # ═══════════════════════════════════════════════════════════════
            # МЕТРИКИ RAG-FLOW (после LLM)
            # ═══════════════════════════════════════════════════════════════
            if full.strip():
                generated_tokens = count_tokens(full)
                total_time = time.time() - start_time
                gen_time = time.time() - llm_start
                decode_time = max(gen_time - ttft_seconds, 0.001)
                decode_tps = generated_tokens / decode_time
                total_tps = generated_tokens / gen_time

                logger.info(
                    f"[{user_id}] 📊 RAG-FLOW: {generated_tokens} tok | "
                    f"TTFT={ttft_seconds:.2f}s | decode={decode_time:.2f}s | "
                    f"decode_tok/s={decode_tps:.0f} | total_tok/s={total_tps:.0f} | "
                    f"RAG-flow={total_time:.2f}s | chars={len(full)}"
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
                await self.collection_manager.redis.close()
            except Exception:
                pass

        if self.history is not None:
            try:
                self.history.redis_client.close()
            except Exception:
                pass

        logger.info("RAGCore: все ресурсы закрыты")