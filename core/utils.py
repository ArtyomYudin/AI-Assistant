import hashlib
import re
from typing import Optional
import tiktoken

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