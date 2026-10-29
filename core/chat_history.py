# core/chat_history.py

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
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
import redis
from datetime import timedelta

from pydantic import BaseModel

logger = logging.getLogger(__name__)

class MergeRequest(BaseModel):
    session_id: str
    user_id: str


class RedisChatHistory(BaseChatMessageHistory):
    """
    Хранит историю чата в Redis с TTL.
    - Ключ: chat_history:{session_id}
    - Автоматически создаётся при первом обращении
    - Хранит максимум 100 сообщений
    - TTL = 7 дней (настраиваемо)
    """

    def __init__(
        self,
        session_id: str,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        ttl_days: int = 7,
        max_messages: int = 50
    ):
        self.session_id = session_id
        self.redis_client = redis.StrictRedis(
            host = host,
            port = port,
            db = db,
            decode_responses = True,
            socket_connect_timeout = 5,
            socket_timeout = 5,
        )
        self.key = f"chat_history:{session_id}"
        self.ttl = timedelta(days=ttl_days)  # TTL в секундах
        self.max_messages = max_messages  # ограничение

    def add_message(self, message):
        """Добавляет сообщение. Сохраняет tool_calls и tool_call_id."""
        msg_data = {
            "type": message.type,
            "content": message.content,
            "timestamp": getattr(message, "additional_kwargs", {}).get("timestamp"),
        }

        # ★ Сохраняем tool_calls для AIMessage (иначе API вернёт 400)
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            msg_data["tool_calls"] = tool_calls

        # ★ Сохраняем tool_call_id для ToolMessage (иначе API вернёт 400)
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            msg_data["tool_call_id"] = tool_call_id

        try:
            pipe = self.redis_client.pipeline()
            pipe.lpush(self.key, json.dumps(msg_data))
            pipe.ltrim(self.key, 0, self.max_messages - 1)
            pipe.expire(self.key, int(self.ttl.total_seconds()))
            pipe.execute()
        except Exception as e:
            logger.warning("Ошибка при сохранении сообщения в Redis: %s", e)

    def get_messages(self) -> List:
        """Восстанавливает сообщения с tool_calls и tool_call_id."""
        try:
            raw = self.redis_client.lrange(self.key, 0, -1)
            messages = []
            for item in reversed(raw):
                try:
                    data = json.loads(item)
                    msg_type = data.get("type")
                    content = data.get("content", "")

                    if msg_type == "human":
                        messages.append(HumanMessage(content=content))
                    elif msg_type == "system":
                        messages.append(SystemMessage(content=content))
                    elif msg_type == "ai":
                        # ★ Восстанавливаем tool_calls, если были
                        tool_calls = data.get("tool_calls") or []
                        if tool_calls:
                            messages.append(AIMessage(content=content, tool_calls=tool_calls))
                        else:
                            messages.append(AIMessage(content=content))
                    elif msg_type == "tool":
                        # ★ Восстанавливаем ToolMessage с tool_call_id
                        tool_call_id = data.get("tool_call_id", "")
                        if not tool_call_id:
                            logger.warning(
                                "ToolMessage без tool_call_id — пропускаю: %r",
                                content[:80],
                            )
                            continue
                        messages.append(ToolMessage(
                            content=content, tool_call_id=tool_call_id,
                        ))
                    else:
                        logger.warning("Неизвестный тип сообщения: %s", msg_type)
                except Exception as e:
                    logger.warning("Ошибка при разборе сообщения: %s", e)
            return messages
        except Exception as e:
            logger.warning("Ошибка при чтении истории из Redis: %s", e)
            return []

    def clear(self):
        self.redis_client.delete(self.key)

    @classmethod
    def merge(cls, src_session: str, dst_session: str, **kwargs) -> int:
        src = cls(src_session, **kwargs)
        dst = cls(dst_session, **kwargs)
        messages = src.get_messages()
        if not messages:
            return 0
        for msg in messages:
            dst.add_message(msg)
        src.clear()
        return len(messages)

    def __repr__(self):
        return f"RedisChatHistory(session_id={self.session_id}, ttl={self.ttl.days}d)"