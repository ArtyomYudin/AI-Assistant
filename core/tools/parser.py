# core/tools/parser.py

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
import re

logger = logging.getLogger(__name__)


def parse_tool_calls_fallback(content: str) -> list[dict] | None:
    """
    Парсит сырой <tool_call>...</tool_call> из content, если vLLM-парсер
    не справился (например, из-за лишней '}' в JSON).
    Возвращает [{name, args, id}] или None.
    """
    if not content or "<tool_call>" not in content:
        return None

    results: list[dict] = []
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"

    for match in re.finditer(pattern, content, re.DOTALL):
        raw = match.group(1).strip()
        parsed = _try_parse_json_robust(raw)
        if not parsed or not isinstance(parsed.get("name"), str):
            continue

        args = parsed.get("arguments") or parsed.get("args") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}

        results.append({
            "name": parsed["name"],
            "args": args,
            "id": f"fallback_{len(results)}",
        })

    return results or None


def _try_parse_json_robust(raw: str) -> dict | None:
    """Пробует распарсить JSON, исправляя типичные артефакты."""
    candidates = [raw]
    trimmed = raw.rstrip()

    # Лишние '}' или ']' в конце (артефакт T-lite)
    for _ in range(3):
        if not trimmed or trimmed[-1] not in "}]":
            break
        trimmed = trimmed[:-1].rstrip()
        if trimmed:
            candidates.append(trimmed)

    # Пробуем разные «фиксеры»
    fixers = (
        lambda x: x,
        lambda x: x.replace("'", '"'),
        lambda x: re.sub(r",\s*([}\]])", r"\1", x),
    )

    for candidate in candidates:
        for fixer in fixers:
            try:
                return json.loads(fixer(candidate))
            except json.JSONDecodeError:
                continue

    logger.warning("Fallback не смог распарсить: %r", raw[:200])
    return None