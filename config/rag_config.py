from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

class RAGConfig(BaseSettings):
    # Общие
    PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()
    DATA_DIR: str = "scraped_data"
    MODE: str = "rag"  # rag, llm_only, hybrid

    # LLM / Embedding
    LLM_NAME: str = "Qwen3-8B-AWQ"
    LLM_BASE_URL: str = "http://172.20.4.50:8001/v1"
    LLM_MAX_TOKEN: int = 4096
    LLM_TEMPERATURE: float = 0.3
    EMBEDDING_NAME: str = "ai-forever/FRIDA"
    EMBEDDING_BASE_URL: str = "http://172.20.4.50:8000/v1"

    # Milvus
    MILVUS_URI: str = "http://172.20.4.50:19530"
    COLLECTION_NAME: str = "centrinform_rag"
    RECREATE_COLLECTION: bool = True
    CHECK_DUPLICATES_IN_MILVUS: bool = True

    # Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    # History
    USE_REDIS_HISTORY: bool = True
    MAX_HISTORY_MESSAGES: int = 5
    HISTORY_TTL_DAYS: int = 7

    # Indexing
    INDEX_BATCH_SIZE: int = 100

    # Retrieval
    K: int = 7
    FETCH_K: int = 20

    # Reranker
    RERANKER_BASE_URL: str = "http://172.20.4.50:8002/v1/rerank"

    # Context control
    MAX_CONTEXT_TOKENS: int = 8192
    RESERVED_FOR_COMPLETION: int = 2048
    RESERVED_FOR_OVERHEAD: int = 512

    # Redis
    REDIS_HOST: str = "172.20.4.50"
    REDIS_PORT: int = 6379
    REDIS_TTL: int = 24 * 3600

    # PROMPTS
    QA_PROMPT_TEMPLATE: str = (
        "Вы — экспертный ассистент АО ЦентрИнформ. Отвечайте ТОЛЬКО на основе контекста.\n"
        "Контекст:\n{context}\n\n"
        "История диалога:\n{chat_history}\n\n"
        "Текущий вопрос: {question}\n\n"
        "Правила:\n"
        "1. Только на основе контекста.\n"
        "2. На русском языке.\n"
        "3. Если нет — 'Информация не найдена'.\n"
        "4. Без предположений.\n"
        "Ответ:"
    )
    QA_PROMPT_RAG_EN: str = (
        "You are an expert assistant for АО ЦентрИнформ. "
        "Answer STRICTLY based only on the provided context.\n\n"
        "Context:\n{context}\n\n"
        "Conversation history:\n{chat_history}\n\n"
        "Current question: {question}\n\n"
        "Step 1: Extract facts from the context if they exist.\n"
        "Step 2: Formulate the answer strictly from these facts.\n"
        "Rules:\n"
        "1. Use ONLY the context.\n"
        "2. Respond in Russian.\n"
        "3. If the answer is missing — reply: 'Информация не найдена'.\n"
        "4. No assumptions or speculation.\n"
        "Answer:"
    )
    QA_PROMPT_HYBRID: str = (
        "Ты ассистент АО ЦентрИнформ.\n"
        "Используй документы, если они есть. "
        "Если документов недостаточно — дополни ответ своими знаниями но в рамках вопроса (пометь как [по памяти]).\n\n"
        "Контекст:\n{context}\n\n"
        "История диалога:\n{chat_history}\n\n"
        "Вопрос: {question}\n\n"
        "Шаг 1: Извлеки факты из контекста, если они есть.\n"
        "Шаг 2: Составь ответ строго на основе этих фактов.\n"
        "Ответ:"
    )
    QA_PROMPT_HYBRID_EN: str = (
        "You are an assistant for АО ЦентрИнформ.\n"
        "Use the provided documents if available. "
        "If the documents are insufficient — supplement the answer with your own knowledge "
        "(mark such parts as [по памяти]).\n\n"
        "Context:\n{context}\n\n"
        "Conversation history:\n{chat_history}\n\n"
        "Question: {question}\n\n"
        "Step 1: Extract facts from the context if they exist.\n"
        "Step 2: Formulate the answer based primarily on these facts, "
        "and only if necessary add [по памяти] parts.\n"
        "Answer in Russian:\n"
    )
    QA_PROMPT_LLM_ONLY: str = (
        "Ты экспертный ассистент АО ЦентрИнформ.\n"
        "Используй свои знания для ответа.\n\n"
        "История диалога:\n{chat_history}\n\n"
        "Вопрос: {question}\n\n"
        "Ответ:"
    )
    QA_PROMPT_LLM_ONLY_EN: str = (
        "You are an expert assistant for АО ЦентрИнформ.\n"
        "Answer using your own knowledge, without relying on external documents.\n\n"
        "Conversation history:\n{chat_history}\n\n"
        "Question: {question}\n\n"
        "Answer in Russian:\n"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # игнорировать неизвестные переменные