# core/tools/context.py
import asyncio
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


# core/tools/context.py
from contextvars import ContextVar
from typing import Any, Optional, Callable, Awaitable


_current_tc_id: ContextVar[Optional[str]] = ContextVar("current_tc_id", default=None)

# Словарь привязан к контексту выполнения (корутине/таске).
# У каждого запроса свой изолированный словарь — никаких пересечений
# между параллельными пользователями.
# ВАЖНО: ContextVar изолирован по asyncio-таске. Если инструмент
# выполняется через asyncio.gather, attach_table внутри таски НЕ
# будет виден в pop_attachments_for_calls родителя. Забирайте
# attachments внутри той же таски, где вызывался инструмент
# (см. answer_generator.py::_tool_flow, _run_with_tc_context).
_attachments_by_tc: ContextVar[dict[str, list[str]]] = ContextVar(
    "attachments_by_tc", default=None,
)


def _get_attachments_store() -> dict[str, list[str]]:
    store = _attachments_by_tc.get()
    if store is None:
        store = {}
        _attachments_by_tc.set(store)
    return store


def set_current_tool_call_id(tc_id: str) -> None:
    """Вызывается перед запуском конкретного tool_call."""
    _current_tc_id.set(tc_id)
    _get_attachments_store().setdefault(tc_id, [])


def add_tool_attachment(markdown_block: str, tc_id: Optional[str] = None) -> None:
    """Сохраняет Markdown-блок с привязкой к tool_call_id."""
    key = tc_id or _current_tc_id.get()
    if not key:
        return
    _get_attachments_store().setdefault(key, []).append(markdown_block)


def attach_table(
    title: str,
    headers: list[str],
    rows: list[list[Any]],
    max_rows: int = 200,
) -> None:
    """Универсальный рендер Markdown-таблицы для любых больших списков."""
    total_count = len(rows)
    if total_count == 0:
        return

    truncated_note = ""
    display_rows = rows
    if total_count > max_rows:
        truncated_note = f"\n\n*Показаны первые {max_rows} записей из {total_count}.*"
        display_rows = rows[:max_rows]

    def _esc(val: Any) -> str:
        return str(val).replace("|", "\\|").replace("\n", " ").strip()

    lines = [
        f"### {title} ({total_count} шт.)",
        "",
        "| " + " | ".join(_esc(h) for h in headers) + " |",
        "| " + " | ".join([":---"] * len(headers)) + " |",
    ]
    for row in display_rows:
        lines.append("| " + " | ".join(_esc(cell) for cell in row) + " |")

    add_tool_attachment("\n".join(lines) + truncated_note)


def pop_attachments_for_calls(tc_ids: list[str]) -> str:
    """Забирает и удаляет из памяти все вложения для списка tool_call_id."""
    store = _get_attachments_store()
    blocks: list[str] = []
    for tc_id in tc_ids:
        blocks.extend(store.pop(tc_id, []))
    return "\n\n".join(blocks)


async def run_tool_calls_with_attachments(
    tool_calls: list[dict],
    exec_fn: Callable[[dict], Awaitable[dict]],
) -> tuple[list[dict], str]:
    """
    Выполняет tool_calls параллельно, собирает результаты и attachments.

    Единственная точка, где правильно «переносится» ContextVar через
    asyncio.gather: set_current_tool_call_id и pop_attachments_for_calls
    выполняются внутри одной и той же таски. Любой orchestration-модуль
    (answer_generator, будущий agent.py) вызывает этот хелпер — и не
    думает про границы контекста.

    Возвращает:
      - results: список результатов в порядке tool_calls
      - attached_markdown: конкатенация всех attach_table из инструментов,
        готовая к отправке в UI
    """
    async def _run(tc: dict):
        set_current_tool_call_id(tc["id"])
        try:
            result = await exec_fn(tc)
        finally:
            attachment = pop_attachments_for_calls([tc["id"]])
        return result, attachment

    pairs = await asyncio.gather(*(_run(tc) for tc in tool_calls))
    results = [r for r, _ in pairs]
    attached = "\n\n".join(a for _, a in pairs if a).strip()
    return results, attached