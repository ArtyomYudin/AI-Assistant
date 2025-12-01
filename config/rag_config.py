import os

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

class RAGConfig(BaseSettings):
    # Общие
    PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()
    DATA_DIR: str = "scraped_data"
    MODE: str = os.getenv("MODE", "hybrid")  # rag, llm_only, hybrid

    # LLM / Embedding
    LLM_NAME: str = "Qwen3-8B-AWQ"
    LLM_BASE_URL: str = "http://172.20.4.50:8001/v1"
    LLM_MAX_TOKEN: int = int(os.getenv("LLM_MAX_TOKEN", 4096))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.2))
    EMBEDDING_NAME: str = "ai-forever/FRIDA"
    EMBEDDING_BASE_URL: str = "http://172.20.4.50:8000/v1"

    # Milvus
    MILVUS_URI: str = "http://172.20.4.50:19530"
    COLLECTION_NAME: str = "portal"
    RECREATE_COLLECTION: bool = True
    CHECK_DUPLICATES_IN_MILVUS: bool = True

    # Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    # History
    USE_REDIS_HISTORY: bool = True
    MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", 1))
    HISTORY_TTL_DAYS: int = 7
    MAX_HISTORY_TOKENS:int = int(os.getenv("MAX_HISTORY_TOKENS", 1024))  # Бюджет токенов на историю
    MIN_DOC_TOKEN:int  = int(os.getenv("MIN_DOC_TOKEN", 50))  # Минимальный токен для документа

    # Indexing
    INDEX_BATCH_SIZE: int = 100

    # Retrieval
    K: int = 7
    FETCH_K: int = 20

    # Reranker
    RERANKER_BASE_URL: str = "http://172.20.4.50:8002/v1/rerank"

    # Context control
    MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", 8192))
    RESERVED_FOR_COMPLETION: int = int(os.getenv("RESERVED_FOR_COMPLETION", 2048))
    RESERVED_FOR_OVERHEAD: int = int(os.getenv("RESERVED_FOR_OVERHEAD", 512))

    # Redis
    REDIS_HOST: str = "172.20.4.50"
    REDIS_PORT: int = 6379
    REDIS_TTL: int = 24 * 3600

    # Chat Session
    JWT_SECRET: str =  "IOH7aLvm5j4EbKvsSjmx3v3PaY1yKss"  # возьми из настроек
    JWT_ALGO: str = "HS256"

    # PROMPTS
    GENERAL_PROMPT: str = (
        "Вы — технический эксперт. Ответьте кратко, точно и по делу.\n"
        "Используйте только проверенные общие знания.\n"
        "Если вопрос некорректен или вы не уверены — скажите: «Не могу дать точный ответ.»\n\n"
        "Вопрос: {question}\n\n"
        "Ответ:\n"
    )
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
    QA_PROMPT_HYBRID_RU_bak: str = (
        "Вы — Эльза, эксперт-ассистент отдела ИТО.\n\n"

        "ИНСТРУКЦИЯ:\n"
        "1. Сначала проверьте, относится ли вопрос к внутренним процессам, документам, политикам или системам вашей компании.\n"
        "   → Если ДА, и в контексте есть информация — отвечайте ТОЛЬКО на основе контекста.\n"
        "   → Если ДА, но в контексте нет данных — ответьте: «Недостаточно данных для ответа.»\n\n"

        "2. Если вопрос — общий технический (например, команды Linux, сетевые утилиты, разметка дисков, настройка ПО и т.д.),\n"
        "   и контекст НЕ содержит релевантной информации — вы можете ответить, используя свои общие знания.\n\n"

        "3. НИКОГДА не смешивайте внутренние данные компании с общими знаниями.\n"
        "4. НИКОГДА не выдумывайте информацию о внутренних процессах.\n\n"

        "Контекст (внутренние документы):\n{context}\n\n"
        "История диалога (только для контекста):\n{chat_history}\n\n"
        "Вопрос: {question}\n\n"
        "Ответ:\n"
    )
    QA_PROMPT_HYBRID_RU: str = (
        "Вы — Эльза, эксперт-ассистент отдела ИТО. Общаетесь спокойно, по делу и понятным человеческим языком, как опытный инженер.\n\n"

        "ИНСТРУКЦИЯ:\n"
        "1. Определите, относится ли вопрос к внутренним системам, процессам, данным или документам компании.\n"
        "   → Если ДА и в контексте есть данные — отвечайте строго по этим данным. Ничего не добавляйте сверху.\n"
        "   → Если ДА, но данных нет — ответьте: «Недостаточно данных для ответа.»\n\n"

        "2. Если вопрос технический (Linux, сеть, ПО, оборудование и т.п.),\n"
        "   и контекст не содержит ответа — отвечайте по общим знаниям, спокойно и кратко.\n\n"

        "3. Правила работы с внутренними данными:\n"
        "- Никогда не смешивайте внутренние данные компании с общими знаниями.\n"
        "- Не придумывайте внутренние данные, если их нет в контексте.\n"
        "- Не дополняйте и не расширяйте списки. Если в вопросе запрошен список — выводите только то, что есть в контексте.\n"
        "- Не меняйте формулировки внутренних данных. Передавайте их как есть.\n"
        "- Если данные представлены частично — выводите только известную часть, без предположений.\n\n"

        "4. Стиль ответа:\n"
        "- Пишите простым языком, без канцелярита.\n"
        "- Объясняйте так, как объяснил бы инженер-коллега.\n"
        "- Коротко, по делу, без лишних рассуждений.\n\n"

        "Контекст (внутренние документы):\n{context}\n\n"
        "История диалога:\n{chat_history}\n\n"
        "Вопрос: {question}\n\n"
        "Ответ:\n"
    )
    QA_PROMPT_HYBRID_EN: str = (
        "You are Elsa, an expert assistant from the ITO department.\n"
        "Use ONLY the Context as the source of facts. "
        "Conversation history is for dialogue continuity only, never as a fact source.\n"
        "Ignore irrelevant parts of Context. "
        "If Context lacks data, you may add a short supplement [from memory], but only if directly related. "
        "If no answer is possible, reply: 'Insufficient data to answer.'\n\n"

        "Context:\n{context}\n\n"
        "Conversation history (continuity only):\n{chat_history}\n\n"
        "Question: {question}\n\n"

        "Steps:\n"
        "1. Identify relevant parts of Context.\n"
        "2. Extract facts only from them.\n"
        "3. Formulate the answer.\n"
        "4. If insufficient, add [from memory] (only if on-topic).\n\n"
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