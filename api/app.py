from typing import List, Optional, AsyncGenerator
import json
import asyncio
import logging

from fastapi import FastAPI, Body, Query
from fastapi.responses import StreamingResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field

from config.rag_config import RAGConfig
from core.rag_core import RAGCore

logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API (Streaming)", version="1.0.0")

cfg = RAGConfig()
core = RAGCore(cfg)
core.create_retriever(k=cfg.K, fetch_k=cfg.FETCH_K)
core.create_qa_generator()

class AskRequest(BaseModel):
    question: str = Field(..., description="Пользовательский вопрос")
    session_id: Optional[str] = "api"

class TestRequest(BaseModel):
    question: str
    expected_keywords: List[str] = Field(default_factory=list)
    session_id: Optional[str] = "test"

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/index")
async def index_all():
    docs = core.load_documents(cfg.DATA_DIR)
    core.setup_vectorstore(docs)
    return {"status": "indexed"}

@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    async def generator():
        async for chunk in core.qa_chain_with_history(req.question, session_id=req.session_id or "api"):
            yield chunk
    return StreamingResponse(generator(), media_type="text/plain; charset=utf-8")

@app.get("/ask/sse")
async def ask_sse(question: str = Query(...), session_id: str = Query("api-sse")):
    async def event_publisher() -> AsyncGenerator[dict, None]:
        async for chunk in core.qa_chain_with_history(question, session_id=session_id):
            yield {"event": "token", "data": chunk}
        yield {"event": "done", "data": ""}
    return EventSourceResponse(event_publisher())

@app.post("/test/stream")
async def test_stream(req: TestRequest):
    from ..core.eval import stream_answer_and_evaluate
    async def generator():
        async for chunk in stream_answer_and_evaluate(core, req.question, req.expected_keywords, session_id=req.session_id or "test"):
            yield chunk
    return StreamingResponse(generator(), media_type="text/plain; charset=utf-8")
