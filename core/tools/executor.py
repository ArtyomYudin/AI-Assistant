# core/rools/executor.py

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
from typing import List

from langchain_core.messages import BaseMessage

from core.tools.registry import TOOL_REGISTRY

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Выполнение tool_call'ов и валидация tool-flow."""

    async def exec_tool(self, tool_call: dict) -> dict:
        name = tool_call["name"]
        args = tool_call.get("args") or {}
        tool_id = tool_call.get("id", "?")

        func = TOOL_REGISTRY.get(name)
        if func is None:
            logger.warning("Неизвестный инструмент: %s (id=%s)", name, tool_id)
            return {"error": f"Инструмент '{name}' не найден"}

        try:
            logger.info("🔧 tool_call: %s(%s) id=%s", name, args, tool_id)

            if asyncio.iscoroutinefunction(func):
                coro = func(**args)
            elif hasattr(func, "ainvoke"):
                coro = func.ainvoke(args)
            else:
                coro = asyncio.to_thread(func, **args)
            result = await asyncio.wait_for(coro, timeout=5.0)

            if not isinstance(result, dict):
                result = {"result": result}
            return result

        except asyncio.TimeoutError:
            logger.warning("⏱️ tool %s превысил 5 сек — прерываю", name)
            return {"error": f"Инструмент {name} не ответил вовремя"}
        except TypeError as e:
            logger.warning("Неверные аргументы для %s(%s): %s", name, args, e)
            return {"error": f"Неверные аргументы для {name}: {e}"}
        except Exception as e:
            logger.exception("Ошибка выполнения %s: %s", name, e)
            return {"error": f"Ошибка выполнения {name}: {e}"}

    @staticmethod
    def validate_tool_calls(messages: List[BaseMessage], user_id: str = "?") -> None:
        """Проверяет, что у каждого tool_call_id есть ответный ToolMessage."""
        open_ids = set()
        for m in messages:
            if m.type == "ai" and getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    open_ids.add(tc["id"])
            elif m.type == "tool" and hasattr(m, "tool_call_id"):
                open_ids.discard(m.tool_call_id)
        if open_ids:
            logger.error(
                f"[{user_id}] ⚠️ ВАЛИДАЦИЯ: {len(open_ids)} tool_call_id без ответа — "
                f"API вернёт 400. IDs: {open_ids}"
            )