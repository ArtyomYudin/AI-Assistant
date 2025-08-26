from dataclasses import dataclass
import os
from pathlib import Path

@dataclass
class RAGConfig:
    # PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
    PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()
    DATA_DIR: str = str(PROJECT_ROOT / "scraped_data")

    # LLM/Embeddings (OpenAI-compatible)
    LLM_NAME: str = os.getenv("RAG_LLM_NAME", "Qwen3-8B-AWQ")
    LLM_BASE_URL: str = os.getenv("RAG_LLM_BASE_URL", "http://172.20.4.50:8001/v1")
    EMBEDDING_NAME: str = os.getenv("RAG_EMBEDDING_NAME", "ai-forever/FRIDA")
    EMBEDDING_BASE_URL: str = os.getenv("RAG_EMBEDDING_BASE_URL", "http://172.20.4.50:8000/v1")

    # Milvus
    MILVUS_URI: str = os.getenv("RAG_MILVUS_URI", "http://172.20.4.50:19530")
    COLLECTION_NAME: str = os.getenv("RAG_COLLECTION", "centrinform_rag")
    RECREATE_COLLECTION: bool = os.getenv("RAG_RECREATE_COLLECTION", "true").lower() == "true"
    CHECK_DUPLICATES_IN_MILVUS: bool = os.getenv("RAG_CHECK_DUPLICATES", "true").lower() == "true"

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))

    # History
    MAX_HISTORY_MESSAGES: int = int(os.getenv("RAG_MAX_HISTORY_MESSAGES", "5"))

    # Indexing
    INDEX_BATCH_SIZE: int = int(os.getenv("RAG_INDEX_BATCH_SIZE", "100"))

    # Retrieval
    K: int = int(os.getenv("RAG_K", "7"))
    FETCH_K: int = int(os.getenv("RAG_FETCH_K", "15"))

    # Reranker
    RERANKER_BASE_URL: str = os.getenv("RAG_RERANKER_BASE_URL", "http://172.20.4.50:8002/v1/rerank")

    # Context control
    MAX_CONTEXT_TOKENS: int = int(os.getenv("RAG_MAX_CONTEXT_TOKENS", "8192"))
    RESERVED_FOR_COMPLETION: int = int(os.getenv("RAG_RESERVED_FOR_COMPLETION", "2048"))
    RESERVED_FOR_OVERHEAD: int = int(os.getenv("RAG_RESERVED_FOR_OVERHEAD", "512"))

    # Redis
    REDIS_HOST: str = os.getenv("RAG_REDIS_HOST", "172.20.4.50")
    REDIS_PORT: int = int(os.getenv("RAG_REDIS_PORT", "6379"))
    REDIS_TTL: int = 24 * 3600  # сутки