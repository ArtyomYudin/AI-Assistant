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
