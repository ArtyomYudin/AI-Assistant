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
    MODE: str = os.getenv("MODE", "rag")  # rag, llm_only, hybrid

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
    DROP_ALL_COLLECTIONS: bool = os.getenv("DROP_ALL_COLLECTIONS", "false").lower() == "true"
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
    K: int = 6  # Сколько документов пойдет в Qwen после реранкера
    FETCH_K: int = 20  # Сколько документов Milvus достает изначально для оценки реранкером
    EXPANSION_MODE: str = os.getenv("EXPANSION_MODE", "smart").lower()
    EXPANSION_THRESHOLD: float = float(os.getenv("EXPANSION_THRESHOLD", 0.6))
    MAX_EXPANDED_SOURCES: int = int(os.getenv("MAX_EXPANDED_SOURCES", 2))
    MAX_CHUNKS_PER_SOURCE: int = int(os.getenv("MAX_CHUNKS_PER_SOURCE", 15))
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
    MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", 12))
    HISTORY_TTL_DAYS: int = 7
    MAX_HISTORY_TOKENS: int = int(os.getenv("MAX_HISTORY_TOKENS", 4000))  # для RAG-ветки
    MAX_TOOL_HISTORY_TOKENS: int = int(os.getenv("MAX_TOOL_HISTORY_TOKENS", 3500))  # для tool-check
    MIN_DOC_TOKEN:int  = int(os.getenv("MIN_DOC_TOKEN", 50))  # Минимальный токен для документа

    # Indexing
    INDEX_BATCH_SIZE: int = 100

    # Reranker
    RERANKER_BASE_URL: str = "http://172.20.4.50:8002"

    # Context control (12288 с slack)
    MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", 11400)) # Максимум под документы
    MAX_SINGLE_TURN_CONTEXT: int = 3500  # потолок контекста на ход
    MAX_QUESTION_TOKENS: int = 2000  # потолок «голого» вопроса/лога
    RESERVED_FOR_COMPLETION: int = int(os.getenv("RESERVED_FOR_COMPLETION", 1024)) # Запас на ответ LLM
    RESERVED_FOR_OVERHEAD: int = int(os.getenv("RESERVED_FOR_OVERHEAD", 300))  # Системный промпт + история


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
    QA_PROMPT_RAG_SYSTEM: str = (
        "Роль: технический ассистент отдела ИТО. Пиши по-русски.\n\n"

        "ИСТОЧНИК ДАННЫХ:\n"
        "- Используй данные только из блока [Контекст].\n"
        "- Если в [Контексте] нет ответа, выведи фразу: "
        "«В предоставленных данных нет информации по этому запросу.»\n\n"

        "ДЕТАЛИ (обязательно):\n"
        "- Приводи все IP-адреса, модели оборудования, имена хостов и числа "
        "из [Контекста] ПОЛНОСТЬЮ.\n"
        "- Для перечислений (камеры, VLAN, IP, модели) — компактный markdown-список "
        "или таблица со всеми элементами, даже если их 2–3.\n"
        "- Пример: «Левое крыло — 172.20.58.11 (E12A), правое крыло — 172.20.58.12 (E12A)».\n\n"

        "ФОРМАТ ОТВЕТА:\n"
        "- Только факты, без вводных («На основе данных…») и заключительных "
        "(«Если нужно — могу…») фраз.\n"
        "- Не предлагай услуг (экспорт, фильтрация, действия).\n"
        "- Без эмодзи.\n"
        "- Отвечай компактно; для полных списков используй markdown-таблицу.\n"
        "- Если вопрос неоднозначен — задай ровно один уточняющий вопрос."
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
        "Роль: технический ассистент отдела ИТО. Пиши по-русски, строго по делу.\n\n"

        "ИСТОЧНИКИ ДАННЫХ:\n"
        "1. Вопросы по инфраструктуре (регламенты, серверы, VLAN): используй данные "
        "только из блока [Контекст]. Если ответа там нет, выведи фразу: "
        "«В предоставленных данных нет информации по этому запросу.»\n"
        "2. JSON в ToolMessage (статус почты, погода, диагностика): переведи "
        "технический ответ на понятный русский язык (принято, отклонено, "
        "Greylisting и т.п.). Это авторитетные данные наравне с [Контекстом].\n"
        "3. Общие ИТ-вопросы (ОС, сети, БД): если в [Контексте] нет данных, "
        "отвечай из общих знаний системного администратора.\n\n"

        "ДЕТАЛИ (обязательно):\n"
        "- Приводи все IP-адреса, модели, имена хостов и числа из [Контекста] ПОЛНОСТЬЮ.\n"
        "- Для перечислений — компактный markdown-список или таблица со всеми элементами.\n\n"

        "ФОРМАТ ОТВЕТА:\n"
        "- Только факты, без вводных и заключительных фраз («Если нужно — могу…»).\n"
        "- Не предлагай услуг (экспорт, фильтрация, действия).\n"
        "- Без эмодзи.\n"
        "- Отвечай компактно; для полных списков используй markdown-таблицу.\n"
        "- Если вопрос неоднозначен — задай ровно один уточняющий вопрос."
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