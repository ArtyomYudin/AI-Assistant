# core/utils.py

import hashlib
import logging
import os
import re
from typing import Optional
import tiktoken
from fastapi import HTTPException, Header
from jose import jwt, JWTError
from functools import lru_cache
from config.rag_config import RAGConfig
import time as _time

import config.rag_config

logger = logging.getLogger(__name__)

# Кэш токенизаторов: родной (HF) и fallback (tiktoken)
_native_tokenizer = None
_native_tokenizer_failed = False
_fallback_tokenizer = None

_last_attempt = 0.0
RETRY_INTERVAL = 60.0


@lru_cache(maxsize=1)
def get_config() -> RAGConfig:
    """Единый экземпляр RAGConfig (BaseSettings читает .env — дорого)."""
    return RAGConfig()

def _resolve_model_path() -> Optional[str]:
    """
    Определяет путь к папке модели по приоритету:
    1. env LLM_MODEL_PATH (проброшено из docker-compose)
    2. config.RAGConfig.LLM_MODEL_PATH (если поле добавлено)
    3. None — тогда используется fallback
    """
    path = os.getenv("LLM_MODEL_PATH")
    if path:
        return path

    try:
        return get_config().LLM_MODEL_PATH
    except Exception:
        return None


def _load_native_tokenizer():
    """
    Ленивая загрузка родного токенизатора модели из LLM_MODEL_PATH.
    При первой ошибке выставляет флаг _native_tokenizer_failed,
    чтобы не пытаться снова на каждый вызов.
    """
    global _native_tokenizer, _native_tokenizer_failed, _last_attempt

    if _native_tokenizer is not None:
        return _native_tokenizer

    now = _time.monotonic()
    if _native_tokenizer_failed and (now - _last_attempt) < RETRY_INTERVAL:
        return None
    _last_attempt = now

    path = _resolve_model_path()
    if not path:
        logger.warning(...)
        _native_tokenizer_failed = True
        return None

    try:
        from transformers import AutoTokenizer
        _native_tokenizer = AutoTokenizer.from_pretrained(...)
        logger.info("Родной токенизатор загружен: %s", path)
        _native_tokenizer_failed = False  # ← сбросить флаг при успехе
        return _native_tokenizer
    except Exception as e:
        logger.warning(...)
        _native_tokenizer_failed = True
        return None


def _get_fallback_tokenizer():
    global _fallback_tokenizer
    if _fallback_tokenizer is None:
        _fallback_tokenizer = tiktoken.get_encoding("cl100k_base")
    return _fallback_tokenizer


def get_tokenizer():
    """
    Возвращает родной токенизатор модели, если он доступен,
    иначе — tiktoken cl100k_base.
    """
    native = _load_native_tokenizer()
    return native if native is not None else _get_fallback_tokenizer()


def _encode(tokenizer, text: str):
    """Универсальная обёртка encode: HF и tiktoken принимают разные kwargs."""
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def _decode(tokenizer, ids):
    """Универсальная обёртка decode."""
    try:
        return tokenizer.decode(ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(ids)


def count_tokens(text: Optional[str], tokenizer=None) -> int:
    """
    Считает токены через родной токенизатор модели.
    Если он недоступен — fallback на tiktoken с калибровкой:
    cl100k завышает русский ~1.33x (2.71 токена/слово у Qwen
    против 1.74 у T-lite), поэтому умножаем на 0.75.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    if not text:
        return 0

    if tokenizer is _native_tokenizer and _native_tokenizer is not None:
        return len(_encode(tokenizer, text))

    # Fallback-ветка (tiktoken или явно переданный чужой токенизатор)
    return int(len(_encode(tokenizer, text)) * 0.75)


def truncate_text_by_tokens(text: str, max_tokens: int, tokenizer=None) -> str:
    if tokenizer is None:
        tokenizer = get_tokenizer()

    if not text:
        return ""

    tokens = _encode(tokenizer, text)
    if len(tokens) <= max_tokens:
        return text

    return _decode(tokenizer, tokens[:max_tokens])


def hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

def clean_md_content(content: str) -> str:
    if not content:
        return ""
    content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
    content = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", content)
    content = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", "", content)
    content = re.sub(r"\n\s*\n", "\n\n", content)
    return content.strip()

def extract_global_bm25_text(md_content: str) -> Optional[str]:
    """Извлекает <!-- bm25: ... --> из всего документа."""
    match = re.search(r"<!--\s*bm25:\s*(.*?)\s*-->", md_content, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None

def remove_bm25_comments(md_content: str) -> str:
    """Удаляет <!-- bm25: ... --> из текста, чтобы не попал в чанки."""
    return re.sub(r"<!--\s*bm25:.*?-->", "", md_content, flags=re.IGNORECASE | re.DOTALL)

def normalize_score(score: float, ranked_by: str) -> float:
    """
    Нормализует score от Milvus или reranker в диапазон [0, 1].

    - milvus/cosine: расстояние (чем меньше, тем ближе) → 1 - dist
    - milvus/BM25: может быть > 1 → min(score / 10, 1.0)  (эмпирически)
    - reranker: уже в [0, 1], оставляем как есть
    """
    try:
        if ranked_by == "reranker":
            return max(0.0, min(1.0, score))  # гарантируем 0–1
        elif ranked_by == "milvus":
            # эвристика: если score <= 1 → cosine distance
            if 0 <= score <= 1:
                return 1.0 - score
            # иначе считаем BM25 (обычно десятки)
            return min(score / 10.0, 1.0)
    except Exception:
        return 0.0


def get_current_user(authorization: str = Header(...)):
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(401, "Invalid auth scheme")

        cfg = get_config()
        payload = jwt.decode(
            token,
            cfg.JWT_SECRET,                    # ← экземпляр
            algorithms=[cfg.JWT_ALGO],         # ← экземпляр
        )

        # Терпим разные имена claim'ов: auth-сервисы кладут по-разному
        sub = (
            payload.get("sub")
            or payload.get("user_id")
            or payload.get("uid")
            or payload.get("username")
        )
        if not sub or not isinstance(sub, str):
            raise HTTPException(
                401,
                f"JWT has no valid user identifier. Claims: {list(payload.keys())}",
            )
        return sub

    except HTTPException:
        raise                          # не оборачивать наши 401 в 500
    except (JWTError, ValueError) as e:
        raise HTTPException(401, f"Invalid token: {e}")


def remove_stopwords_russian(text: str) -> str:
    """Удаляет стоп-слова из текста для BM25-поиска."""
    if not text:
        return ""
    # Приводим к нижнему регистру и разбиваем на слова
    words = re.findall(r"\b\w+\b", text.lower())
    # Фильтруем
    filtered = [w for w in words if w not in config.rag_config.RUSSIAN_STOPWORDS]
    return " ".join(filtered)