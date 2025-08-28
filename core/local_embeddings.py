import httpx
from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    """LangChain-совместимый Embeddings для FRIDA."""

    def __init__(self, model: str, base_url: str, timeout: int = 60):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

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
