#!/usr/bin/env python3
"""
Генерирует тестовый JWT для локальной отладки RAG API.

Использование:
    python scripts/make_test_jwt.py                          # sub=tester, TTL 24h
    python scripts/make_test_jwt.py ivanov                   # sub=ivanov
    python scripts/make_test_jwt.py ivanov 3600              # TTL 1 час
"""
import sys
import time
from datetime import datetime, timezone

from jose import jwt

# Импорт конфига — тот же, что использует rag-api
from config.rag_config import RAGConfig

cfg = RAGConfig()


def make_jwt(sub: str = "tester", ttl_seconds: int = 24 * 3600) -> str:
    now = int(time.time())
    payload = {
        "sub": sub,                     # ← это и есть user_id в RAG
        "iat": now,
        "exp": now + ttl_seconds,
        "iss": "ai-assistant-test",
    }
    token = jwt.encode(payload, cfg.JWT_SECRET, algorithm=cfg.JWT_ALGO)
    return token


if __name__ == "__main__":
    sub = sys.argv[1] if len(sys.argv) > 1 else "tester"
    ttl = int(sys.argv[2]) if len(sys.argv) > 2 else 24 * 3600
    token = make_jwt(sub, ttl)
    print(token)