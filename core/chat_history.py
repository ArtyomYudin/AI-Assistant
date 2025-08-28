import json
import hashlib
from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
import redis
from datetime import timedelta

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
    ):
        self.session_id = session_id
        self.redis_client = redis.StrictRedis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        self.key = f"chat_history:{session_id}"
        self.ttl = timedelta(days=ttl_days)  # TTL в секундах
        self.max_messages = 100  # ограничение

    def add_message(self, message):
        """Добавляет сообщение в историю."""
        msg_data = {
            "type": message.type,
            "content": message.content,
            "timestamp": getattr(message, "additional_kwargs", {}).get("timestamp"),
        }

        # Сохраняем в Redis
        try:
            # Добавляем в начало списка
            self.redis_client.lpush(self.key, json.dumps(msg_data))
            # Ограничиваем длину списка
            self.redis_client.ltrim(self.key, 0, self.max_messages - 1)
            # Устанавливаем TTL, если ключа ещё нет
            if self.redis_client.ttl(self.key) == -1:  # ключ не имеет TTL
                self.redis_client.expire(self.key, int(self.ttl.total_seconds()))
        except Exception as e:
            print(f"Ошибка при сохранении сообщения в Redis: {e}")

    def get_messages(self) -> List:
        """Возвращает список сообщений (в обратном порядке)."""
        try:
            raw = self.redis_client.lrange(self.key, 0, -1)
            messages = []
            for item in reversed(raw):  # последние — первые
                try:
                    data = json.loads(item)
                    cls = HumanMessage if data["type"] == "human" else AIMessage
                    messages.append(cls(content=data["content"]))
                except Exception as e:
                    print(f"Ошибка при разборе сообщения: {e}")
            return messages
        except Exception as e:
            print(f"Ошибка при чтении истории из Redis: {e}")
            return []

    def clear(self):
        """Очищает историю."""
        self.redis_client.delete(self.key)

    def __repr__(self):
        return f"RedisChatHistory(session_id={self.session_id}, ttl={self.ttl.days}d)"
