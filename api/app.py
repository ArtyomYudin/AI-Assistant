import json
import os
import logging

from core.collection_manager import CollectionManager
from core.utils import get_current_user

# Настраиваем логирование приложения
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

from contextlib import asynccontextmanager
from typing import List, Optional, AsyncGenerator

from fastapi import FastAPI, Body, Query, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from config.rag_config import RAGConfig
from core.rag_core import RAGCore
from core.chat_history import RedisChatHistory

logger = logging.getLogger(__name__)

cfg = RAGConfig()
core: RAGCore | None = None

# Список разрешённых фронтенд-адресов
ALLOWED_ORIGINS = [
    "http://127.0.0.1:4200",  # дев
    "https://itsupport.center-inform.ru",  # прод
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global core
    logger.info("🚀 Инициализация RAGCore...")
    try:
        core = RAGCore(cfg)
        core.collection_manager = CollectionManager(core)
        # Индексация и создание коллекций
        # await core.collection_manager.build_all_collections()
        await core.collection_manager.build_router()
        # Создаём ретривер с маршрутизацией
        core.create_retriever(k=cfg.K, fetch_k=cfg.FETCH_K)
        # QA генератор использует retriever с маршрутизацией
        core.create_qa_generator()
        logger.info("✅ RAGCore успешно инициализирован")
    except Exception as e:
        logger.exception("❌ Ошибка инициализации RAGCore: %s", e)
        raise

    yield

    # Закрытие при завершении
    if core:
        await core.close()
        logger.info("🔌 RAGCore закрыт")

app = FastAPI(title="RAG API (Streaming)", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., description="Пользовательский вопрос")
    session_id: Optional[str] = "api"

class TestRequest(BaseModel):
    question: str
    expected_keywords: List[str] = Field(default_factory=list)
    session_id: Optional[str] = "test"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # или ["http://localhost:4200"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/index")
async def index_all():
    docs = core.load_documents(cfg.DATA_DIR)
    await core.setup_vectorstore(docs)
    return {"status": "indexed"}

@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    async def generator():
        async for chunk in core.qa_chain_with_history(req.question, session_id=req.session_id or "api"):
            yield chunk
    # return StreamingResponse(generator(), media_type="text/plain; charset=utf-8")
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # важно для nginx
        "Content-Type": "text/event-stream; charset=utf-8",
    }
    return StreamingResponse(generator(), headers=headers)

@app.get("/ask/sse")
async def ask_sse(request: Request, question: str = Query(...), session_id: str = Query("api-sse")):
    async def event_publisher() -> AsyncGenerator[dict, None]:
        async for chunk in core.qa_chain_with_history(question, session_id=session_id):
            yield {"event": "token", "data": json.dumps(chunk)}
        yield {"event": "done", "data": ""}

    # origin = request.headers.get("origin")
    # if origin in ALLOWED_ORIGINS:
    #     allowed_origin = origin
    # else:
    #     allowed_origin = ""  # Можно вернуть пустой или заблокировать
    # headers = {
    #     "Access-Control-Allow-Origin": allowed_origin,
    #     "Cache-Control": "no-cache",
    #     "Connection": "keep-alive",
    #     "X-Accel-Buffering": "no",  # важно для nginx
    #     "Content-Type": "text/event-stream; charset=utf-8",
    # }
    return EventSourceResponse(event_publisher())

@app.post("/test/stream")
async def test_stream(req: TestRequest):
    from ..core.eval import stream_answer_and_evaluate
    async def generator():
        async for chunk in stream_answer_and_evaluate(core, req.question, req.expected_keywords, session_id=req.session_id or "test"):
            yield chunk
    return StreamingResponse(generator(), media_type="text/plain; charset=utf-8")

@app.post("/merge-session")
def merge_session(old_session_id: str, user_id: str = Depends(get_current_user)):
    moved = RedisChatHistory.merge(old_session_id, user_id)
    return {"status": "ok", "moved": moved}
