# core/orchestration/prompt_diagnostics.py

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

import json
import logging
from typing import List

from langchain_core.messages import BaseMessage

from core.common.text import count_tokens

logger = logging.getLogger(__name__)


def count_msg_tokens(m: BaseMessage) -> int:
    """Токены сообщения с учётом tool_calls (для AIMessage)."""
    base = count_tokens(m.content or "")
    tool_calls = getattr(m, "tool_calls", None)
    if tool_calls:
        base += count_tokens(json.dumps(tool_calls, ensure_ascii=False))
    return base


def log_prompt_stats(user_id: str, messages: List[BaseMessage], branch: str) -> None:
    """Раскладка промпта по ролям: system / history / tool / question."""
    if not messages:
        return

    system_tok = 0
    history_tok = 0
    tool_tok = 0
    question_tok = 0

    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].type == "human":
            last_human_idx = i
            break

    for i, m in enumerate(messages):
        n = count_msg_tokens(m)
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


def log_full_messages(user_id: str, messages: List[BaseMessage], branch: str) -> None:
    """Полный дамп всех сообщений — только в DEBUG."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
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