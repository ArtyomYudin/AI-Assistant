# core/embedding_cache.py

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

import hashlib
import json
import redis
from typing import Optional, List

class RedisEmbeddingCache:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, ttl: int = 3600):
        self.client = redis.StrictRedis(host=host, port=port, db=db, decode_responses=False)
        self.ttl = ttl  # время жизни ключа в секундах

    @staticmethod
    def _make_key(text: str) -> str:
        norm = text.strip().lower()
        return "emb:" + hashlib.sha256(norm.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        key = self._make_key(text)
        data = self.client.get(key)
        if data:
            return json.loads(data.decode("utf-8"))
        return None

    def set(self, text: str, vector: List[float]) -> None:
        key = self._make_key(text)
        self.client.setex(key, self.ttl, json.dumps(vector))
