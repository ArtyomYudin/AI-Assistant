# core/orchestration/prompt_builder.py

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

import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from config.rag_config import RAGConfig
from core.common.text import count_tokens, truncate_text_by_tokens, count_msg_tokens

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Чистая логика построения промпта:
      - выбор system-промпта по режиму
      - token-aware обрезка истории (никогда не рвёт tool-flow)
      - сборка контекста из документов (по источникам целиком)
      - сборка финального списка сообщений

    Не хранит состояния и ничего не пишет наружу — только читает config.
    Легко покрывается юнит-тестами без моков.
    """

    def __init__(self, config: RAGConfig):
        self.config = config

    # ── System prompt ────────────────────────────────────────────

    def get_system_prompt(self, mode: str) -> str:
        if mode == "rag":
            return self.config.QA_PROMPT_RAG_SYSTEM
        if mode == "llm_only":
            return self.config.QA_PROMPT_LLM_ONLY_SYSTEM
        return self.config.QA_PROMPT_HYBRID_RU_SYSTEM

    # ── Truncation ───────────────────────────────────────────────

    def truncate_history_safe(
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
             слегка превышает бюджет.
          4. Последний резерв: только последний HumanMessage.
        """
        if not history:
            return []

        safe_points = [i for i, m in enumerate(history) if m.type == "human"]
        if not safe_points:
            return history

        def _total(msgs):
            # Учитываем tool_calls и tool_call_id — иначе AIMessage с
            # tool_calls занижает бюджет и tool-check уходит с переполнением.
            return sum(count_msg_tokens(m, include_role=True) for m in msgs)

        for idx in safe_points:
            candidate = history[idx:]
            if _total(candidate) <= max_tokens:
                return candidate

        last_human_idx = safe_points[-1]
        tail = history[last_human_idx:]

        if len(tail) >= 2:
            one_turn = tail[:2]
            if _total(one_turn) <= max_tokens:
                return one_turn
            # fallback не должен возвращать over-budget turn.
            # Обрезаем содержимое HumanMessage, оставляя AI как есть.
            # Считаем «бюджет на контент» как max_tokens − роль-маркеры.
            logger.warning(
                "История переполнена: последний ход (%d tok) > бюджет (%d tok). "
                "Обрезаю HumanMessage.",
                _total(one_turn), max_tokens,
            )

            human_msg = one_turn[0]
            ai_msg = one_turn[1] if len(one_turn) > 1 else None

            ai_tokens = count_msg_tokens(ai_msg, include_role=True) if ai_msg else 0
            human_overhead = count_msg_tokens(
                HumanMessage(content=""), include_role=True
            )
            human_budget = max(0, max_tokens - ai_tokens - human_overhead)

            new_human = HumanMessage(
                content=truncate_text_by_tokens(
                    human_msg.content or "", human_budget
                )
            )
            result = [new_human]
            if ai_msg is not None:
                result.append(ai_msg)
            return result

        # один Human — тоже обрезаем, если не влезает
        only_human = tail[:1]
        if _total(only_human) > max_tokens:
            logger.warning(
                "История: единственный Human (%d tok) > бюджета (%d tok) — обрезаю.",
                _total(only_human), max_tokens,
            )
            human_overhead = count_msg_tokens(
                HumanMessage(content=""), include_role=True
            )
            return [HumanMessage(content=truncate_text_by_tokens(
                only_human[0].content or "",
                max(0, max_tokens - human_overhead),
            ))]
        return only_human

    # ── Context ──────────────────────────────────────────────────

    def build_context(
        self,
        docs: List[Document],
        history: List[BaseMessage],
        question: str = "",
        available_override: Optional[int] = None,
    ) -> Tuple[str, List[BaseMessage], str]:
        """
        Формирует контекст для LLM:
          - документы группируются по source
          - источник включается ЦЕЛИКОМ или пропускается
          - источники упорядочены по top-score своих чанков
          - fallback: если ни один источник не влез — top-чанки с truncate
        available_override — если caller уже посчитал израсходованный бюджет
        (например, tool-flow, где в messages уже лежат system + history +
        question + tool_calls + ToolMessage), он передаёт готовый лимит на
        документы. В этом случае внутренний расчёт available пропускается.
        """
        question_for_budget = truncate_text_by_tokens(
            question or "", self.config.MAX_QUESTION_TOKENS
        )
        question_tokens = count_tokens(question_for_budget)

        # 1. История
        truncated_history = self.truncate_history_safe(
            history, self.config.MAX_HISTORY_TOKENS
        )
        history_tokens = sum(
            count_msg_tokens(m, include_role=True) for m in truncated_history
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

        # 2. Бюджет на документы
        if available_override is not None:
            available = max(0, available_override)
            logger.info(
                "Context: budget override=%d tok "
                "(history_summary=%d, question=%d — уже учтены caller'ом)",
                available, history_tokens, question_tokens,
            )
        else:
            available = (
                self.config.MAX_CONTEXT_TOKENS
                - self.config.RESERVED_FOR_COMPLETION
                - history_tokens
                - self.config.RESERVED_FOR_OVERHEAD
                - question_tokens
            )
            available = max(0, available)

        if available <= 0:
            return "", truncated_history, history_summary

        # Группировка по source
        by_source: dict[str, list[Document]] = {}
        for d in docs:
            src = d.metadata.get("source", "N/A")
            by_source.setdefault(src, []).append(d)

        source_groups = sorted(
            by_source.items(),
            key=lambda kv: max(d.metadata.get("score", 0.0) for d in kv[1]),
            reverse=True,
        )
        for _, chunks in source_groups:
            any_expanded = any(d.metadata.get("expanded") for d in chunks)
            if any_expanded:
                chunks.sort(key=lambda d: d.metadata.get("chunk_index", 999))
            else:
                chunks.sort(key=lambda d: -d.metadata.get("score", 0.0))

        # Сборка
        parts: list[str] = []
        used = 0
        included: list[tuple[str, int, int]] = []
        skipped: list[tuple[str, int]] = []

        for src, chunks in source_groups:
            body = "\n".join(d.page_content.strip() for d in chunks)
            src_text = f"[Источник: {src}]\n{body}\n"
            src_tokens = count_tokens(src_text)

            if used + src_tokens > available:
                skipped.append((src, src_tokens))
                continue

            parts.append(src_text)
            used += src_tokens
            included.append((src, len(chunks), src_tokens))

        # Fallback
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

    # ── Messages ─────────────────────────────────────────────────

    def build_messages(
        self,
        mode: str,
        question: str,
        context: str,
        truncated_history: List[BaseMessage],
        history_summary: str = "",
    ) -> List[BaseMessage]:
        """
        Структура промпта:
          1) SystemMessage — правила режима (СТАТИЧНЫЙ между ходами)
          2) SystemMessage — резюме обрезанных сообщений (только при truncation)
          3) truncated_history — пары user/assistant из истории
          4) HumanMessage — «Контекст:\\n{...}\\n\\nВопрос: {question}»
        обрезаем question до MAX_QUESTION_TOKENS здесь — это
        последний барьер перед LLM. Вызывающий код может передать оригинал,
        но в промпт уйдёт не больше лимита.
        """
        question_for_llm = truncate_text_by_tokens(
            question or "", self.config.MAX_QUESTION_TOKENS
        )
        if len(question_for_llm) < len(question or ""):
            logger.warning(
                "Question обрезан: %d → %d символов (MAX_QUESTION_TOKENS=%d)",
                len(question or ""), len(question_for_llm),
                self.config.MAX_QUESTION_TOKENS,
            )

        system_content = self.get_system_prompt(mode)
        messages: List[BaseMessage] = [SystemMessage(content=system_content)]

        if history_summary:
            messages.append(SystemMessage(content=history_summary))

        messages.extend(truncated_history)

        if context.strip():
            user_content = f"Контекст:\n{context}\n\nВопрос: {question_for_llm}"
        else:
            user_content = question_for_llm

        messages.append(HumanMessage(content=user_content))
        return messages