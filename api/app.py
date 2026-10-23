# api/app.py

import json
import os
import logging

from core.utils import get_current_user

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Query, Depends, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from config.rag_config import RAGConfig
from core.rag_core import RAGCore

logger = logging.getLogger(__name__)

cfg = RAGConfig()
core: RAGCore | None = None

ALLOWED_ORIGINS = [
    "http://127.0.0.1:4200",              # дев
    "https://itsupport.center-inform.ru",  # прод
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global core
    logger.info("🚀 Инициализация RAGCore...")
    try:
        core = RAGCore(cfg)                          # CollectionManager — внутри
        await core.collection_manager.build_router()
        core.create_retriever(k=cfg.K, fetch_k=cfg.FETCH_K)
        core.create_qa_generator()
        logger.info("✅ RAGCore успешно инициализирован")
    except Exception as e:
        logger.exception("❌ Ошибка инициализации RAGCore: %s", e)
        raise

    yield

    if core:
        await core.close()
        logger.info("🔌 RAGCore закрыт")


app = FastAPI(title="RAG API (Streaming)", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Публичный /health — для мониторинга ─────────────────────────
@app.get("/health")
async def health():
    checks = {}

    try:
        await core.milvus.client.list_collections()
        checks["milvus"] = "ok"
    except Exception as e:
        checks["milvus"] = f"error: {e}"

    try:
        core.collection_manager.redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    try:
        r = await core.embeddings.embed_query("ping")
        checks["embeddings"] = "ok" if r else "empty"
    except Exception as e:
        checks["embeddings"] = f"error: {e}"

    llm_stats = core.llm_limiter.stats()
    checks["llm"] = "saturated" if llm_stats["available_slots"] == 0 else "ok"

    status = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": status, "checks": checks, "llm_limiter": llm_stats}


# ─── RAG только для авторизованных ───────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Пользовательский вопрос",
    )


@app.get("/ask/sse")
async def ask_sse(
    request: Request,
    question: str = Query(..., min_length=1, max_length=2000),
    user_id: str = Depends(get_current_user),        # ← JWT-защита
):
    """
    SSE-стриминг ответа. Только для авторизованных.
    session_id = user_id из JWT — история привязана к пользователю.
    """
    async def event_publisher() -> AsyncGenerator[dict, None]:
        try:
            async for chunk in core.qa_chain_with_history(question, user_id=user_id):
                yield {
                    "event": "token",
                    "data": json.dumps(chunk, ensure_ascii=False),
                }
        except Exception as e:
            logger.exception("[%s] SSE stream error: %s", user_id, e)
            yield {
                "event": "token",
                "data": json.dumps(
                    {"type": "error", "text": "Ошибка при генерации ответа."},
                    ensure_ascii=False,
                ),
            }
        # done отдаём всегда — клиент знает, что поток завершён
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_publisher())


@app.post("/ask/stream")
async def ask_stream(
    req: AskRequest,
    user_id: str = Depends(get_current_user),        # ← JWT-защита
):
    """NDJSON-стриминг (для клиентов без поддержки SSE)."""
    async def generator():
        try:
            async for chunk in core.qa_chain_with_history(req.question, user_id=user_id):
                yield (json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8")
        except Exception as e:
            logger.exception("[%s] NDJSON stream error: %s", user_id, e)
            yield (json.dumps(
                {"type": "error", "text": "Ошибка при генерации ответа."},
                ensure_ascii=False,
            ) + "\n").encode("utf-8")

    return StreamingResponse(generator(), media_type="application/x-ndjson")


# ─── Индексация — только админ ───────────────────────────────────
@app.post("/index")
async def index_all(user_id: str = Depends(get_current_user)):
    """
    Переиндексация всех подпапок DATA_DIR.
    TODO: проверка user_id на админ-роль, если появится RBAC.
    """
    docs = core.load_documents(cfg.DATA_DIR)
    await core.setup_vectorstore(docs)
    return {"status": "indexed", "triggered_by": user_id}