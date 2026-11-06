# api/app.py

import json
import config.logging_config  # noqa: F401 — настраивает логи ДО core.*

import logging

from core.common.auth import get_current_user

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Query, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from config.rag_config import RAGConfig
from core.rag_core import RAGCore

logger = logging.getLogger(__name__)

cfg = RAGConfig()
core: RAGCore | None = None


# ─── Lifespan ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    global core
    logger.info("🚀 Инициализация RAGCore...")
    try:
        new_core = RAGCore(cfg)
        await new_core.collection_manager.build_router()
        new_core.create_retriever(k=cfg.K, fetch_k=cfg.FETCH_K)
        new_core.create_qa_generator()
        await new_core.warmup()
        core = new_core
        logger.info("✅ RAGCore успешно инициализирован")
    except Exception as e:
        logger.exception("❌ Ошибка инициализации RAGCore: %s", e)
        raise

    yield

    if core is not None:
        await core.close()
        logger.info("🔌 RAGCore закрыт")


app = FastAPI(title="RAG API (Streaming)", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,  # type: ignore[arg-type]
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Публичный /health ───────────────────────────────────────
@app.get("/health")
async def health():
    rag = core
    if rag is None:
        return {"status": "degraded", "checks": {"core": "not initialized"}}

    checks: dict[str, str] = {}

    try:
        await rag.milvus.client.list_collections()
        checks["milvus"] = "ok"
    except Exception as e:
        checks["milvus"] = f"error: {e}"

    try:
        await rag.collection_manager.redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    try:
        r = await rag.models.embeddings.embed_query("ping")
        checks["embeddings"] = "ok" if r else "empty"
    except Exception as e:
        checks["embeddings"] = f"error: {e}"

    llm_stats = rag.llm_limiter.stats()
    checks["llm"] = "saturated" if llm_stats["available_slots"] == 0 else "ok"

    status = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": status, "checks": checks, "llm_limiter": llm_stats}


# ─── RAG только для авторизованных ───────────────────────────
class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=2000,
        description="Пользовательский вопрос",
    )


async def _stream_sse(
    rag: RAGCore, question: str, user_id: str
) -> AsyncGenerator[dict, None]:
    """SSE-стример. Отдельная функция — чтобы Pylance видел non-None rag."""
    try:
        async for chunk in rag.qa_chain_with_history(question, user_id=user_id):
            yield {"event": "token", "data": json.dumps(chunk, ensure_ascii=False)}
    except Exception as e:
        logger.exception("[%s] SSE stream error: %s", user_id, e)
        yield {
            "event": "token",
            "data": json.dumps(
                {"type": "error", "text": "Ошибка при генерации ответа."},
                ensure_ascii=False,
            ),
        }
    yield {"event": "done", "data": ""}


@app.get("/ask/sse")
async def ask_sse(
    question: str = Query(..., min_length=1, max_length=2000),
    user_id: str = Depends(get_current_user),
):
    rag = core
    if rag is None:
        raise HTTPException(503, "RAGCore not initialized")
    return EventSourceResponse(_stream_sse(rag, question, user_id))


async def _stream_ndjson(
    rag: RAGCore, question: str, user_id: str
) -> AsyncGenerator[bytes, None]:
    try:
        async for chunk in rag.qa_chain_with_history(question, user_id=user_id):
            yield (json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8")
    except Exception as e:
        logger.exception("[%s] NDJSON stream error: %s", user_id, e)
        yield (json.dumps(
            {"type": "error", "text": "Ошибка при генерации ответа."},
            ensure_ascii=False,
        ) + "\n").encode("utf-8")


@app.post("/ask/stream")
async def ask_stream(
    req: AskRequest,
    user_id: str = Depends(get_current_user),
):
    rag = core
    if rag is None:
        raise HTTPException(503, "RAGCore not initialized")
    return StreamingResponse(
        _stream_ndjson(rag, req.question, user_id),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Индексация — только админ ───────────────────────────────
@app.post("/index")
async def index_all(user_id: str = Depends(get_current_user)):
    rag = core
    if rag is None:
        raise HTTPException(503, "RAGCore not initialized")

    try:
        await rag.collection_manager.build_all_collections()
        await rag.collection_manager.build_router()
    except Exception as e:
        logger.exception("[%s] /index failed: %s", user_id, e)
        raise HTTPException(500, f"Indexing failed: {e}")

    return {"status": "indexed", "triggered_by": user_id}