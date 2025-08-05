# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import logging

from classes.rag_core import RAGCore

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic модели для API
class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"


class ClearSessionRequest(BaseModel):
    session_id: str


class DocumentResponse(BaseModel):
    content: str
    source: str
    score: Optional[float] = None


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[DocumentResponse]
    session_id: str
    error: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class ChatHistoryResponse(BaseModel):
    history: List[Dict[str, str]]


# Инициализация RAG
rag = RAGCore(
    llm_base_url="http://172.20.4.50:8001/v1",
    embedding_base_url="http://172.20.4.50:8000/v1",
)

# Загружаем документы и настраиваем систему при старте
def initialize_rag():
    logger.info("Загрузка документов...")
    docs = rag.load_documents_from_directory()
    if docs:
        logger.info("Настройка векторной БД...")
        rag.setup_vectorstore(docs)
        rag.create_retriever()
        rag.setup_qa_chain()
        logger.info("RAG система готова.")
    else:
        logger.warning("Документы не найдены. Убедитесь, что папка scraped_data содержит файлы.")


# FastAPI App
app = FastAPI(
    title="RAG API",
    description="API для RAG-системы с поддержкой чата, поиска и истории",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Для разработки. В продакшене укажите домены.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Автоматическая инициализация при старте
@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_rag()
    yield

app = FastAPI(lifespan=lifespan)


# === Маршруты API ===

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Задать вопрос и получить ответ с RAG.
    """
    try:
        result = rag.ask_question_rag(question=request.question, session_id=request.session_id)
        return result
    except Exception as e:
        logger.error(f"Ошибка в /ask: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=List[DocumentResponse])
async def search_documents(request: SearchRequest):
    """
    Поиск похожих документов (без генерации).
    """
    try:
        results = rag.search_similar_documents(query=request.query, k=request.k)
        return results
    except Exception as e:
        logger.error(f"Ошибка в /search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_history(session_id: str):
    """
    Получить историю чата по session_id.
    """
    try:
        history = rag.get_chat_history(session_id)
        return {"history": history}
    except Exception as e:
        logger.error(f"Ошибка в /history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
async def clear_session(request: ClearSessionRequest):
    """
    Очистить историю чата.
    """
    try:
        rag.clear_session(request.session_id)
        return {"status": "success", "message": f"Сессия {request.session_id} очищена"}
    except Exception as e:
        logger.error(f"Ошибка в /clear: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Проверка состояния API.
    """
    return {"status": "healthy", "model_loaded": True}