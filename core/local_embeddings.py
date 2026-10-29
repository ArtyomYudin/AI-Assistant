# core/local_embeddings.py

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

import httpx
from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    """LangChain-совместимый Embeddings для FRIDA."""

    def __init__(self, model: str, base_url: str, timeout: int = 60):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def aclose(self):
        await self.client.aclose()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self.client.post(
            "/embeddings",
            json={"model": self.model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"]

    async def embed_query(self, text: str) -> list[float]:
        resp = await self.client.post(
            "/embeddings",
            json={"model": self.model, "input": [text]},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]
