from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

class RAGConfig(BaseSettings):
    # Общие
    PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()
    DATA_DIR: str = "scraped_data"
    MODE: str = "hybrid"  # rag, llm_only, hybrid

    # LLM / Embedding
    LLM_NAME: str = "Qwen3-8B-AWQ"
    LLM_BASE_URL: str = "http://172.20.4.50:8001/v1"
    LLM_MAX_TOKEN: int = 4096
    LLM_TEMPERATURE: float = 0.1
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
    MAX_HISTORY_MESSAGES: int = 3
    HISTORY_TTL_DAYS: int = 7
    MAX_HISTORY_TOKENS:int = 1024  # Бюджет токенов на историю
    MIN_DOC_TOKEN:int  = 50  # Минимальный токен для документа

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

    # Chat Session
    JWT_SECRET: str =  "IOH7aLvm5j4EbKvsSjmx3v3PaY1yKss"  # возьми из настроек
    JWT_ALGO: str = "HS256"

    # PROMPTS
    QA_PROMPT_TEMPLATE: str = (
        "Вы — экспертный ассистент отдела ИТО. Отвечайте ТОЛЬКО на основе контекста.\n"
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
        "You are an expert assistant in the ITO department. Your name is Elsa.\n"
        "Use the provided documents in the Context as the ONLY source of factual information. "
        "Conversation history is given only to maintain continuity of dialogue, "
        "but it MUST NOT be used as a source of facts.\n\n"

        "Very important: Context may include multiple unrelated documents. "
        "When answering, use ONLY the fragments that are directly relevant to the Question. "
        "Completely ignore unrelated fragments, even if they contain technical details.\n\n"

        "If the Context does not contain enough information to answer, "
        "you may supplement the answer with your own knowledge "
        "(mark such parts explicitly as [по памяти]). "
        "Do not add irrelevant information. "
        "If the answer is unknown — reply: \"Недостаточно данных для ответа.\"\n\n"

        "Context:\n{context}\n\n"
        "Conversation history (for continuity only, not for facts):\n{chat_history}\n\n"
        "Question: {question}\n\n"

        "Step 0: Identify which parts of Context are directly relevant to the Question.\n"
        "Step 1: Extract facts strictly from the relevant parts of Context.\n"
        "Step 2: Formulate the answer based only on these extracted facts.\n"
        "Step 3: If Context lacks sufficient data — add a short supplement [по памяти], "
        "but only if it directly relates to the Question.\n"
        "Step 4: If no answer can be given even [по памяти] — reply: "
        "\"Недостаточно данных для ответа.\"\n\n"
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