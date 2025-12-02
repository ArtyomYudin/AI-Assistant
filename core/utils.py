import hashlib
import re
from typing import Optional
import tiktoken
from fastapi import HTTPException, Header
from jose import jwt, JWTError

import config.rag_config

_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer

def count_tokens(text: Optional[str], tokenizer=None) -> int:
    tokenizer = tokenizer or get_tokenizer()
    return len(tokenizer.encode(text or ""))

def truncate_text_by_tokens(text: str, max_tokens: int, tokenizer=None) -> str:
    tokenizer = tokenizer or get_tokenizer()
    tokens = tokenizer.encode(text or "")
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens])

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
            raise HTTPException(status_code=401, detail="Invalid auth scheme")
        payload = jwt.decode(token, config.rag_config.RAGConfig.JWT_SECRET, algorithms=[config.rag_config.RAGConfig.JWT_ALGO])
        return payload.get("sub")
    except (JWTError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid token")

def remove_stopwords_russian(text: str) -> str:
    """Удаляет стоп-слова из текста для BM25-поиска."""
    if not text:
        return ""
    # Приводим к нижнему регистру и разбиваем на слова
    words = re.findall(r"\b\w+\b", text.lower())
    # Фильтруем
    filtered = [w for w in words if w not in config.rag_config.RUSSIAN_STOPWORDS]
    return " ".join(filtered)