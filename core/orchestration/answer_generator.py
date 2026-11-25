# core/orchestration/answer_generator.py

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
import json
import logging
import time
from typing import AsyncIterator, Awaitable, Callable, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)

from config.rag_config import RAGConfig
from core.storage.chat_history import RedisChatHistory
from core.models.model_provider import ModelProvider
from core.common.polite_detector import PoliteDetector
from core.orchestration.prompt_builder import PromptBuilder
from core.orchestration.prompt_diagnostics import log_full_messages, log_prompt_stats
from core.models.rate_limiter import LLMRateLimiter, RateLimitExceeded
from core.tools.context import set_current_tool_call_id, pop_attachments_for_calls, run_tool_calls_with_attachments
from core.tools.executor import ToolExecutor
from core.common.text import count_tokens, count_msg_tokens, truncate_text_by_tokens


logger = logging.getLogger(__name__)

Retriever = Callable[[str, Optional[str]], Awaitable[List[Document]]]
HistoryFactory = Callable[[str], RedisChatHistory]


class AnswerGenerator:
    """
    Стримит ответ пользователю.

    Диспетчеризация:
      1) пустой вопрос / вежливая фраза → мгновенный ответ
      2) параллельно: retrieval + история
      3) tool-check
      4) ветка A (tool) или ветка B (RAG) — независимые корутины
    """

    def __init__(
        self,
        *,
        config: RAGConfig,
        retriever: Retriever,
        history_factory: HistoryFactory,
        models: ModelProvider,
        prompts: PromptBuilder,
        tool_executor: ToolExecutor,
        limiter: LLMRateLimiter,
    ):
        self.config = config
        self.retriever = retriever
        self.history_factory = history_factory
        self.models = models
        self.prompts = prompts
        self.tool_executor = tool_executor
        self.llm_limiter = limiter

    # ── Публичная точка входа ────────────────────────────────────

    async def stream(self, question: str, user_id: str = "default") -> AsyncIterator[dict]:
        start_time = time.time()
        logger.debug(f"[{user_id}] 🚀 Начал обработку: {question[:100]}...")

        if not question.strip():
            yield {"type": "response", "text": "Пожалуйста, задайте вопрос."}
            return

        if PoliteDetector.is_polite(question):
            yield {"type": "response", "text": PoliteDetector.reply(question)}
            return

        # обрезаем для LLM сразу на входе. Retrieval получает
        # оригинал — ему важна полная семантика. Всё, что идёт в LLM
        # (tool-check, tool-flow, rag-flow), использует обрезанную версию.
        question_for_llm = truncate_text_by_tokens(
            question, self.config.MAX_QUESTION_TOKENS
        )
        if len(question_for_llm) < len(question):
            logger.warning(
                "[%s] Question обрезан для LLM: %d → %d символов",
                user_id, len(question), len(question_for_llm),
            )

        yield {"type": "status", "text": "🔍 Ищу информацию...\n"}

        # Параллельный старт: retrieval + история
        retrieval_task = asyncio.create_task(self.retriever(question, user_id))
        history_task = asyncio.create_task(
            asyncio.to_thread(self.history_factory(user_id).get_messages)
        )

        try:
            hist = await history_task
            logger.debug(f"[{user_id}] ✅ История: {len(hist)} сообщений")
        except Exception as e:
            logger.exception(f"[{user_id}] ❌ Ошибка истории: {e}")
            hist = []

        # Tool-check
        # передаём обрезанный question дальше
        first_response, tool_messages = await self._tool_check(
            question_for_llm, hist, user_id
        )

        if first_response and getattr(first_response, "tool_calls", None):
            async for chunk in self._tool_flow(
                question, user_id, hist, tool_messages, first_response,
                retrieval_task, start_time,
            ):
                yield chunk
        else:
            # tool-check мог сгенерировать ответ, но без tool_calls —
            # в этом случае content надо передать в RAG как preliminary,
            # а не выбрасывать (иначе 8B-модель «забывает», что уже знает).
            preanswer = ""
            if first_response and getattr(first_response, "content", None):
                preanswer = first_response.content.strip()
            async for chunk in self._rag_flow(
                question, user_id, hist, retrieval_task, start_time, preliminary_answer=preanswer,
            ):
                yield chunk

    # ── Tool-check ───────────────────────────────────────────────

    async def _tool_check(
        self,
        question: str,
        hist: List[BaseMessage],
        user_id: str,
    ) -> tuple[Optional[BaseMessage], List[BaseMessage]]:
        truncated_tool_hist = self.prompts.truncate_history_safe(
            hist, self.config.MAX_TOOL_HISTORY_TOKENS
        )
        # Используем единый счётчик с tool_calls — иначе AIMessage с
        # tool_calls занижает бюджет и попадает в LLM с переполнением.
        tool_hist_tokens = sum(
            count_msg_tokens(m, include_role=True) for m in truncated_tool_hist
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

        try:
            first_response = await self.models.tool_check_llm.ainvoke(tool_messages)
            return first_response, tool_messages
        except Exception as e:
            logger.exception(f"[{user_id}] ❌ Ошибка tool-check: {e}")
            return None, tool_messages

    # ── Ветка A: tool-flow ──────────────────────────────────────

    async def _tool_flow(
        self,
        question: str,
        user_id: str,
        hist: List[BaseMessage],
        tool_messages: List[BaseMessage],
        first_response: BaseMessage,
        retrieval_task: "asyncio.Task[List[Document]]",
        start_time: float,
    ) -> AsyncIterator[dict]:
        # Дожидаемся retrieval (он уже стартовал параллельно)
        retrieval_docs: List[Document] = []
        try:
            retrieval_docs = await asyncio.wait_for(retrieval_task, timeout=5.0)
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


        # results = await asyncio.gather(
        #     *(self.tool_executor.exec_tool(tc) for tc in first_response.tool_calls)
        # )

        # Хелпер инкапсулирует set_current_tool_call_id + exec + pop
        # внутри одной таски. ContextVar изолирован по asyncio-таске,
        # поэтому pop должен происходить там же, где set — иначе
        # attachments теряются (см. core/tools/context.py).
        results, attached_markdown = await run_tool_calls_with_attachments(
            first_response.tool_calls,
            self.tool_executor.exec_tool,
        )

        for tc, result in zip(first_response.tool_calls, results):
            tool_messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=tc["id"],
                )
            )

        if attached_markdown:
            logger.info(
                "[%s] tool-flow: attachments collected (%d chars)",
                user_id, len(attached_markdown),
            )

        yield {"type": "status", "text": "🧠 Формирую ответ...\n"}

        # Заменяем system-промпт роутера на боевой QA-промпт
        tool_messages[0] = SystemMessage(
            content=self.prompts.get_system_prompt(self.config.MODE)
        )

        # Добавляем RAG-контекст, если retrieval что-то нашёл

        # считаем уже занятый бюджет ДО вставки RAG-контекста ──
        # tool_messages на этой строке содержит:
        #   QA-system + truncated_tool_hist + Human(question)
        #   + AIMessage(tool_calls) + ToolMessage(JSON result)
        already_used_tokens = sum(
            count_msg_tokens(m, include_role=True) for m in tool_messages
        )
        rag_budget = (
                self.config.MAX_CONTEXT_TOKENS
                - already_used_tokens
                - self.config.RESERVED_FOR_COMPLETION
        )
        logger.info(
            "[%s] tool-flow budget: already_used=%d tok, rag_budget=%d tok "
            "(max_context=%d, completion_reserve=%d)",
            user_id, already_used_tokens, rag_budget,
            self.config.MAX_CONTEXT_TOKENS, self.config.RESERVED_FOR_COMPLETION,
        )
        log_prompt_stats(user_id, tool_messages, branch="tool-flow/pre-rag")

        if retrieval_docs and rag_budget > self.config.MIN_DOC_TOKEN:
            try:
                rag_context, _, _ = self.prompts.build_context(
                    retrieval_docs, hist, question,
                    available_override=rag_budget,
                )
                if rag_context.strip():
                    tool_messages.insert(
                        1,
                        SystemMessage(content=(
                            "Дополнительный контекст из внутренних документов "
                            "(относится к части вопроса про инфраструктуру):\n"
                            f"{rag_context}"
                        )),
                    )
                    logger.info(
                        "[%s] tool-flow: RAG-контекст добавлен (%d chars, ~%d tok)",
                        user_id, len(rag_context), count_tokens(rag_context),
                    )
                else:
                    logger.info(
                        "[%s] tool-flow: RAG-контекст пуст (build_context вернул '')",
                        user_id,
                    )
            except Exception as e:
                logger.warning(
                    "[%s] tool-flow: не удалось построить RAG-контекст: %s",
                    user_id, e,
                )
        elif retrieval_docs:
            # Бюджета нет — вставка RAG-контекста гарантированно приведёт к 400.
            logger.warning(
                "[%s] tool-flow: ⚠️ RAG-контекст пропущен — rag_budget=%d tok "
                "≤ MIN_DOC_TOKEN=%d. already_used=%d + completion=%d > "
                "MAX_CONTEXT_TOKENS=%d",
                user_id, rag_budget, self.config.MIN_DOC_TOKEN,
                already_used_tokens, self.config.RESERVED_FOR_COMPLETION,
                self.config.MAX_CONTEXT_TOKENS,
            )

        log_prompt_stats(user_id, tool_messages, branch="tool-flow")
        log_full_messages(user_id, tool_messages, branch="tool-flow")
        self.tool_executor.validate_tool_calls(tool_messages, user_id)

        llm_start = time.time()
        full = ""
        buffer = ""
        ttft_seconds = 0.0
        first_token_received = False

        if not self.llm_limiter.is_available():
            yield {"type": "status", "text": "⏳ Ожидаю свободный слот LLM...\n"}

        try:
            async with self.llm_limiter.acquire(user_id=user_id):
                async for chunk in self.models.llm.astream(tool_messages):
                    content = chunk.content
                    if content:
                        if not first_token_received:
                            ttft_seconds = time.time() - llm_start
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

            # Сохраняем чистый ответ модели ДО прикрепления таблиц
            llm_only_reply = full

            # ── прикрепляем таблицы, собранные из тасок ДО стриминга ──
            # attached_markdown получен в _run_with_tc_context (см. выше),
            # pop_attachments_for_calls здесь не нужен.
            if attached_markdown:
                suffix = f"\n\n{attached_markdown}"
                full += suffix
                yield {"type": "response", "text": suffix}

        except RateLimitExceeded as e:
            if attached_markdown:
                yield {"type": "response", "text": attached_markdown}
            logger.warning(f"[{user_id}] Rate limit (tool-flow): {e}")
            yield {"type": "error", "text": str(e)}
            return
        except Exception as e:
            if attached_markdown:
                yield {"type": "response", "text": attached_markdown}
            logger.exception(f"[{user_id}] ❌ Генерация (tool-flow): {e}")
            yield {"type": "error", "text": "Произошла ошибка при генерации ответа."}
            return

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

            # try:
            #     h = self.history_factory(user_id)
            #     h.add_message(HumanMessage(content=question))
            #     h.add_message(first_response)
            #     for tc, result in zip(first_response.tool_calls, results):
            #         h.add_message(ToolMessage(
            #             content=json.dumps(result, ensure_ascii=False),
            #             tool_call_id=tc["id"],
            #         ))
            #     h.add_message(AIMessage(content=full))
            # except Exception as e:
            #     logger.exception(f"[{user_id}] ❌ Ошибка сохранения: {e}")
            try:
                h = self.history_factory(user_id)
                h.add_message(HumanMessage(content=question))
                # Не сохраняем ToolMessage в историю: полный JSON (до 2000 токенов)
                # устаревает, дублирует AI-ответ и раздувает контекст.
                # Если пользователь задаст follow-up — модель вызовет инструмент
                # снова (0.5–1.7с), это дешевле, чем хранить 2000 токенов stale JSON.
                h.add_message(AIMessage(content=llm_only_reply))
            except Exception as e:
                logger.exception(f"[{user_id}] ❌ Ошибка сохранения: {e}")

        logger.debug(f"[{user_id}] 🎯 Tool-flow: {time.time() - start_time:.2f}с")

    # ── Ветка B: RAG-flow ───────────────────────────────────────

    async def _rag_flow(
        self,
        question: str,
        user_id: str,
        hist: List[BaseMessage],
        retrieval_task: "asyncio.Task[List[Document]]",
        start_time: float,
        preliminary_answer: str = "",
    ) -> AsyncIterator[dict]:
        yield {"type": "status", "text": "📚 Ищу документы...\n"}

        try:
            docs = await retrieval_task
            logger.debug(f"[{user_id}] ✅ Найдено {len(docs)} документов")
        except Exception as e:
            logger.exception(f"[{user_id}] ❌ Ошибка поиска: {e}")
            yield {"type": "error", "text": "Ошибка при поиске документов."}
            return

        # Follow-up retry: пустой retrieval + есть история
        if not docs and hist:
            last_human = next(
                (m.content for m in reversed(hist) if m.type == "human"),
                None,
            )
            if last_human and last_human.strip() != question.strip():
                combined = f"{last_human} {question}"
                logger.info(
                    f"[{user_id}] 🔁 follow-up retry: "
                    f"'{last_human[:50]}…' + '{question[:50]}…'"
                )
                try:
                    docs = await self.retriever(combined, user_id)
                    logger.info(
                        f"[{user_id}] 🔁 follow-up retry → docs={len(docs)}"
                    )
                except Exception as e:
                    logger.warning(
                        f"[{user_id}] follow-up retry failed: {e}"
                    )

        if not docs:
            if self.config.MODE == "rag":
                yield {"type": "info", "text": "Информация не найдена."}
                return
            logger.info(f"[{user_id}] Нет документов — режим {self.config.MODE}")
            docs = []

        yield {"type": "status", "text": "✅ Документы найдены. Формирую ответ...\n"}

        try:
            context, truncated_history, history_summary = self.prompts.build_context(
                docs, hist, question
            )
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

        messages = self.prompts.build_messages(
            mode=self.config.MODE,
            question=question,
            context=context,
            truncated_history=truncated_history,
            history_summary=history_summary,
        )

        # Если tool-check сгенерировал осмысленный ответ, но без tool_calls —
        # используем его как «черновик», на который опирается основная LLM.

        if preliminary_answer and len(preliminary_answer) >= 30:
            messages.insert(-1, SystemMessage(content=(
                "Черновик ответа от роутера (может быть неполным, но содержит "
                f"факты из истории):\n{preliminary_answer}\n\n"
                "Используй эти факты, если они релевантны вопросу. Если нет — "
                "игнорируй."
            )))
            logger.info(
                "[%s] rag-flow: preliminary_answer добавлен (%d chars)",
                user_id, len(preliminary_answer),
            )

        log_prompt_stats(user_id, messages, branch="RAG")
        log_full_messages(user_id, messages, branch="RAG")

        llm_start = time.time()
        full = ""
        buffer = ""
        ttft_seconds = 0.0
        first_token_received = False

        if not self.llm_limiter.is_available():
            yield {"type": "status", "text": "⏳ Ожидаю свободный слот LLM...\n"}

        try:
            async with self.llm_limiter.acquire(user_id=user_id):
                async for chunk in self.models.llm.astream(messages):
                    if content := chunk.content:
                        if not first_token_received:
                            ttft_seconds = time.time() - llm_start
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
                h = self.history_factory(user_id)
                h.add_message(HumanMessage(content=question))
                h.add_message(AIMessage(content=full))
            except Exception as e:
                logger.exception(f"[{user_id}] ❌ Ошибка сохранения: {e}")

        logger.debug(f"[{user_id}] 🎯 RAG-flow: {time.time() - start_time:.2f}с")