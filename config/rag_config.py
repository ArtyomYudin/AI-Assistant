# config/rag_config.py

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

import os

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Set

# Минимальный набор стоп-слов для русского (можно расширить)
RUSSIAN_STOPWORDS: Set[str] = {
    "в", "и", "на", "с", "по", "к", "от", "до", "из", "у", "о", "а", "но", "или",
    "что", "как", "если", "то", "когда", "где", "кто", "этот", "тот", "весь", "все",
    "быть", "иметь", "делать", "сказать", "мочь", "для", "также", "только", "еще",
    "уже", "очень", "самый", "через", "между", "под", "над", "за", "все", "мой",
    "ваш", "его", "её", "их", "мы", "вы", "они", "я", "ты", "он", "она", "оно"
}

class RAGConfig(BaseSettings):
    # Общие
    PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()
    DATA_DIR: str = "scraped_data"
    MODE: str = os.getenv("MODE", "hybrid")  # rag, llm_only, hybrid

    # LLM / Embedding
    LLM_NAME: str = os.getenv("LLM_NAME", "T-lite-it-2.1-AWQ") #"Qwen3-8B-AWQ"
    LLM_MODEL_PATH: str = os.getenv(
        "LLM_MODEL_PATH",
        "/models/pimenovdv/T-lite-it-2.1-AWQ-4bits",
    )
    LLM_BASE_URL: str = "http://172.20.4.50:8001/v1"
    LLM_MAX_TOKEN: int = int(os.getenv("LLM_MAX_TOKEN", 2048))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.7))
    LLM_PRESENCE_PENALTY: float = float(os.getenv("LLM_PRESENCE_PENALTY", 0.4)) # штраф за уже сказанное — борется с повтором фактов
    LLM_FREQUENCY_PENALTY: float = float(os.getenv("LLM_FREQUENCY_PENALTY", 0.2)) # штраф за частые токены
    LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", 0.8))
    LLM_TOP_K: int = int(os.getenv("LLM_TOP_K", 20))
    EMBEDDING_NAME: str = "ai-forever/FRIDA"
    EMBEDDING_BASE_URL: str = "http://172.20.4.50:8000/v1"

    # ─── LLM rate-limit ──────────────────────────────────────────────
    # (Оптимизировано под MI50 16GB / 8k контекст)
    # Разрешаем 2 параллельных потока генерации (оптимально для объема кэша 8k)
    LLM_MAX_CONCURRENT: int = int(os.getenv("LLM_MAX_CONCURRENT", 2))
    # Время ожидания в очереди. При 8k генерация быстрее, 30 сек более чем достаточно
    LLM_MAX_QUEUE_WAIT: int = int(os.getenv("LLM_MAX_QUEUE_WAIT", 30))
    # Лимит запросов в минуту на один user_id. 0 = выключить
    LLM_MAX_PER_USER_PER_MIN: int = int(os.getenv("LLM_MAX_PER_USER_PER_MIN", 10))


    # Milvus
    MILVUS_URI: str = "http://172.20.4.50:19530"
    COLLECTION_NAME: str = "portal"
    # Безопасное значение по умолчанию во избежание случайной очистки прод-базы
    RECREATE_COLLECTION: bool = os.getenv("RECREATE_COLLECTION", "False").lower() == "true"
    CHECK_DUPLICATES_IN_MILVUS: bool = True
    # Полностью удалить все коллекции Milvus и очистить состояние роутера в Redis.
    # ВНИМАНИЕ: после использования рекомендуется выставить обратно в False,
    # иначе при каждом старте коллекции будут удаляться и переиндексироваться.
    DROP_ALL_COLLECTIONS: bool = os.getenv("DROP_ALL_COLLECTIONS", "true").lower() == "true"
    HYBRID_SEARCH_SCORE_THRESHOLD: float = float(os.getenv("HYBRID_SEARCH_SCORE_THRESHOLD", 0.3))

    # RAG-параметры multi-collection retrieval
    ROUTER_MIN_MARGIN: float = float(os.getenv("ROUTER_MIN_MARGIN", 0.1))
    COLLECTION_THRESHOLD_RATIO: float = float(os.getenv("COLLECTION_THRESHOLD_RATIO", 0.3))
    COLLECTION_MIN_ABSOLUTE: float = float(os.getenv("COLLECTION_MIN_ABSOLUTE", 0.03))
    PER_SOURCE_CAP: int = int(os.getenv("PER_SOURCE_CAP", 2))

    # ─── Retrieval: small-to-big expansion ──────────────────────────
    # off    — не разворачивать источники (старое поведение, только per_source_cap)
    # always — разворачивать источник, если top_score >= EXPANSION_THRESHOLD
    # smart  — broad intent → порог 0.0; narrow → EXPANSION_THRESHOLD
    #          (broad intent: «все», «полный список», «перечисли», «какие есть»)
    EXPANSION_MODE: str = os.getenv("EXPANSION_MODE", "smart").lower()
    EXPANSION_THRESHOLD: float = float(os.getenv("EXPANSION_THRESHOLD", 0.6))
    MAX_EXPANDED_SOURCES: int = int(os.getenv("MAX_EXPANDED_SOURCES", 2))
    MAX_CHUNKS_PER_SOURCE: int = int(os.getenv("MAX_CHUNKS_PER_SOURCE", 20))
    # Если не broad-intent, matched чанков <= этого числа,
    # и top_score >= EXPANSION_SKIP_SCORE — не разворачивать.
    # 0 = отключить эвристику (расширять всегда, как сейчас).
    EXPANSION_SKIP_MAX_MATCHED: int = int(os.getenv("EXPANSION_SKIP_MAX_MATCHED", 1))
    EXPANSION_SKIP_SCORE: float = float(os.getenv("EXPANSION_SKIP_SCORE", 0.9))
    # Валидация режима
    if EXPANSION_MODE not in ("off", "always", "smart"):
        raise ValueError(
            f"EXPANSION_MODE={EXPANSION_MODE!r} не поддерживается. "
            f"Допустимо: off | always | smart"
        )

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 100))

    # History
    USE_REDIS_HISTORY: bool = True
    MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", 10))
    HISTORY_TTL_DAYS: int = 7
    MAX_HISTORY_TOKENS: int = int(os.getenv("MAX_HISTORY_TOKENS", 1024))  # для RAG-ветки
    MAX_TOOL_HISTORY_TOKENS: int = int(os.getenv("MAX_TOOL_HISTORY_TOKENS", 2048))  # для tool-check
    MIN_DOC_TOKEN:int  = int(os.getenv("MIN_DOC_TOKEN", 50))  # Минимальный токен для документа

    # Indexing
    INDEX_BATCH_SIZE: int = 100

    # Retrieval
    K: int = 7  # Сколько документов пойдет в Qwen после реранкера
    FETCH_K: int = 20  # Сколько документов Milvus достает изначально для оценки реранкером

    # Reranker
    RERANKER_BASE_URL: str = "http://172.20.4.50:8002"

    # Context control (Сумма жестко бьется в 8192)
    MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", 5120)) # Максимум под документы
    RESERVED_FOR_COMPLETION: int = int(os.getenv("RESERVED_FOR_COMPLETION", 2048)) # Запас на ответ LLM
    RESERVED_FOR_OVERHEAD: int = int(os.getenv("RESERVED_FOR_OVERHEAD", 1024))  # Системный промпт + история


    # Redis
    REDIS_HOST: str = "172.20.4.50"
    REDIS_PORT: int = 6379
    REDIS_TTL: int = 24 * 3600

    # Chat Session
    JWT_SECRET: str = os.getenv("JWT_SECRET", "jdklsfj8ousof4rklckJKjkhRh098jjk")
    JWT_ALGO: str = os.getenv("JWT_ALGO", "HS256")

    # GRADIO
    GRADIO_USER: str = "admin"
    GRADIO_PASSWORD: str = "admin"
    GRADIO_DEBUG: bool = True  # false на «почти-проде», true при активной разработке

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

    QA_PROMPT_RAG_SYSTEM: str = (
        "Ты — ассистент отдела ИТО. Отвечай по-русски, спокойно и по делу.\n\n"

        "КАК ОТВЕЧАТЬ:\n"
        "1. Используй ТОЛЬКО блок «Контекст». Никаких общих знаний.\n"
        "2. Если в контексте нет прямого ответа — скажи:\n"
        "   «К сожалению, у меня нет информации по этому запросу.»\n\n"

        "ЗАПРЕТЫ:\n"
        "- Не выдумывай IP-адреса, имена, названия систем, номера инструкций.\n"
        "- Не переноси IP, модели, названия из одной секции контекста в другую.\n"
        "- Не используй историю диалога как источник фактов.\n"
        "- Не выполняй действия (отправка, блокировка, настройка, экспорт в Excel/CSV).\n"
        "- Не предлагай дополнительных услуг вроде «могу экспортировать», «могу прислать», «могу отфильтровать».\n"
        "- Не используй эмодзи.\n"
        "- Не пиши вступления вроде «На основе контекста…» или «Согласно документу…».\n\n"

        "ПОЛНЫЕ СПИСКИ:\n"
        "- Если просят «все», «полный», «список», «таблицу» — выпиши ВСЕ элементы из контекста.\n"
        "- После вывода посчитай элементы и напиши «Итого: N».\n"
        "- Используй markdown-таблицу, если это уместно.\n\n"

        "СТИЛЬ:\n"
        "- Кратко: 1–5 предложений, если не просят подробнее.\n"
        "- Английские термины — только общепринятые (SSH, DNS, DHCP, AD, VPN).\n"
    )

    QA_PROMPT_LLM_ONLY_SYSTEM: str = (
        "Ты — ассистент отдела ИТО. Отвечай по-русски, спокойно и по делу.\n\n"

        "КАК ОТВЕЧАТЬ:\n"
        "1. Отвечай из своих общих технических знаний (ОС, сети, ПО, БД, DevOps, оборудование).\n"
        "2. Если не уверен — скажи: «Не могу дать точный ответ.»\n"
        "3. Внутренних данных ИТО (IP, имена, системы, инструкции) у тебя нет — не выдумывай их.\n"
        "4. История диалога — только для разрешения местоимений («он», «это», «там»), не для фактов.\n\n"

        "ЗАПРЕТЫ:\n"
        "- Не выдумывай внутренние процессы, IP-адреса, имена сотрудников, названия систем.\n"
        "- Не выполняй действия (отправка, блокировка, настройка, экспорт).\n"
        "- Не предлагай дополнительных услуг вроде «могу экспортировать», «могу прислать».\n"
        "- Не используй эмодзи.\n\n"

        "СТИЛЬ:\n"
        "- Кратко и технически точно.\n"
        "- Английские термины — только общепринятые.\n"
    )

    QA_PROMPT_HYBRID_RU_SYSTEM: str = (
        "Ты — ассистент отдела ИТО. Отвечай по-русски, спокойно, строго по делу.\n\n"

        "КАК ОТВЕЧАТЬ:\n"
        "1. Вопросы по базе знаний (регламенты, серверы, VLAN, инфраструктура ИТО):\n"
        "   Отвечай ТОЛЬКО на основе предоставленного Контекста. Если ответа нет в тексте, скажи:\n"
        "   «К сожалению, у меня нет информации по этому запросу.»\n"
        "2. Вопросы по статусу почты (данные от почтового шлюза):\n"
        "   Проанализируй полученный от инструмента технический ответ (JSON) и объясни пользователю статус понятным языком "
        "   (принято, отклонено, находится в Greylisting и т.д.).\n"
        "3. Общие технические вопросы (ОС, сети, ПО, БД):\n"
        "   Если в контексте нет прямого ответа — отвечай из общих знаний системного администратора.\n\n"

        "ЗАПРЕТЫ:\n"
        "- Не придумывай IP-адреса, имена, названия систем и логи шлюза, которых нет в предоставленных данных.\n"
        "- Используй данные строго локально, не переноси IP и модели из одних систем/логов в другие.\n"
        "- Не предлагай пользователю выполнить действия, которые ты не можешь сделать сам.\n"
        "- Не предлагай дополнительных услуг вроде «могу экспортировать», «могу прислать», «могу отфильтровать».\n"
        "- Не заканчивай ответ предложениями вроде «Если нужно — могу…», «Если потребуется — могу…», «Дайте знать, если…». Заканчивай ответ на последнем факте.\n"
        "- Не используй эмодзи, пиктограммы и извинения.\n"
        "- Не пиши вступления вроде «На основе контекста…» или «Согласно предоставленным данным…».\n\n"

        "ПОЛНЫЕ СПИСКИ:\n"
        "- Если просят «все», «полный список», «таблицу» — выпиши ВСЕ элементы без пропусков. "
        "- Для естественных списков и логов используй markdown-таблицу.\n\n"

        "СТИЛЬ:\n"
        "- Кратко: 1–5 предложений, если не требуется подробный разбор лога или регламента.\n"
        "- Если вопрос или лог неоднозначен — задай один уточняющий вопрос.\n"
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

    QA_PROMPT_TOOL_CHECK_SYSTEM: str = (
        "Ты — интеллектуальный роутер и диспетчер ИТ-отдела поддержки.\n"
        "Тебе доступен динамический список внешних инструментов (функций) для проверки системных логов, "
        "статусов оборудования, учетных записей и выполнения диагностических команд.\n\n"

        "ТВОЯ ЗАДАЧА:\n"
        "1. Проанализируй запрос пользователя и историю диалога.\n"
        "2. Если для ответа на вопрос ТРЕБУЕТСЯ получить актуальные технические данные, выполнить проверку в реальном времени "
        "   или вызвать системную утилиту — выбери подходящий инструмент из списка предоставленных тебе функций и верни его вызов.\n"
        "3. Если запрос пользователя — это общий вопрос по регламентам, архитектуре сети, базам данных, поиску статических инструкций "
        "   или простое общение — НЕ вызывай никакие инструменты. Отвечай обычным текстом, система автоматически перенаправит запрос в RAG.\n\n"

        "КРИТИЧЕСКИЕ ПРАВИЛА:\n"
        "- Вызывай инструмент только в том случае, если переданные пользователем данные (email, IP, имя хоста) "
        "  строго соответствуют обязательным аргументам функции.\n"
        "- Если нужного инструмента нет в списке предоставленных функций — ничего не вызывай, отвечай обычным текстом."
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # игнорировать неизвестные переменные