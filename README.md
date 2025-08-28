# RAG Project — Полный набор (Streaming + FastAPI + Gradio + Test Evaluator)

## 🚀 Возможности
- 📂 Индексация локальных PDF / MD / TXT (папка `scraped_data/`)  
- 🔍 Векторное хранилище **Milvus** (Hybrid: Dense + BM25 sparse)  
- 🧹 Дедупликация документов по SHA-256  
- ⚖️ Rerank (опционально через vLLM endpoint)  
- 💬 Вопрос–ответ с **потоковой генерацией (streaming)**  
- 🌐 API на **FastAPI** (StreamingResponse + SSE)  
- 🖥 UI на **Gradio** (чат + тестовый режим)  
- ✅ Модуль тест-оценки ответа (ключевые слова + метрики)  

---

## ⚡ Архитектура
```text
Данные (PDF/MD/TXT) → Чанкование → Embeddings → Milvus (dense + BM25)
                                                  │
        ┌─────────────────────────────────────────┘
        ▼
Запрос → Embedding → Milvus Hybrid Search → [опц. MMR] → [опц. Reranker] → LLM (vLLM)
        ▼
    Ответ (Streaming/SSE) + Логи
```

---

## 🏁 Быстрый старт

### 1. Установите зависимости
```bash
pip install -r requirements.txt
```

### 2. Подготовьте данные
Положите файлы в `scraped_data/`.

### 3. Индексация + демо (CLI)
```bash
python -m main
```

### 4. Запуск FastAPI
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:
- `POST /index` — индексация (читает `scraped_data/`)  
- `POST /ask/stream` — стриминг текста  
- `GET /ask/sse?question=...` — SSE (text/event-stream)  
- `POST /test/stream` — ответ + метрики в JSON-маркере  

### 5. Gradio UI
```bash
python -m ui.app
```

---

## Переменные окружения
(см. `config/rag_config.py`)

| Переменная | Описание | Значение по умолчанию |
|------------|----------|------------------------|
| `RAG_LLM_NAME` | LLM модель | Qwen3-8B-AWQ |
| `RAG_LLM_BASE_URL` | OpenAI-compatible endpoint | http://localhost:8000/v1 |
| `RAG_EMBEDDING_NAME` | embedding модель | multilingual-e5-large |
| `RAG_EMBEDDING_BASE_URL` | endpoint для embedding | http://localhost:8000/v1 |
| `RAG_MILVUS_URI` | Milvus URI | http://localhost:19530 |
| `RAG_COLLECTION` | имя коллекции | demo_ci_rag |
| `RAG_RECREATE_COLLECTION` | пересоздавать коллекцию | false |
| `RAG_CHECK_DUPLICATES` | проверка дубликатов | true |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | параметры чанкования | 512 / 128 |
| `RAG_K` / `RAG_FETCH_K` | top_k / fetch_k для поиска | 7 / 15 |
| `RAG_RERANKER_BASE_URL` | endpoint для reranker | (опц.) |
| `RAG_MAX_CONTEXT_TOKENS` | максимум токенов в контексте | 8192 |
| `RAG_RESERVED_FOR_COMPLETION` | запас для генерации | 2048 |
| `RAG_RESERVED_FOR_OVERHEAD` | запас под служебные токены | 512 |

---

## 🖥️ Тестовая инфраструктура

**Хост:**  
```
IBM x3650 M5, 2× E5-2695 v3, 96GB RAM  
2× AMD Instinct MI50 (16GB)  
Ubuntu 24.04 + ROCm 6.3  
vLLM 0.92
```

---

## 🧠 Используемые модели

**LLM:**
- Qwen3-8B-AWQ

**Embeddings:**
- multilingual-e5-large ✅ (лучше работает с русским)  
- Qwen3-Embedding-4B ❌ плохо с русским  
- BGE-3m ❌ плохо с русским  
- e5-mistral-7b-instruct ❌ ошибка `Token id 98285 is out of vocabulary`  
- Giga-Embeddings-instruct ❌ не запускается на vLLM

---

## 📌 TODO
- [ ] Добавить историю чата (Redis)
- [ ] Добавить кэширование вопросов (Redis)
- [ ] Добавить Maximal Marginal Relevance (MMR) для диверсификации кандидатов  
- [ ] Автообновление индекса при изменении данных  
- [ ] Вынести конфигурацию в `.env`  
- [ ] Оптимизировать запуск LLM моделей  

---

## 🔎 Пример запроса к embedding API
```bash
curl http://localhost:8000/v1/embeddings   -H "Content-Type: application/json"   -d '{
        "model": "ai-forever/FRIDA",
        "input": ["Привет мир", "Как дела?"]
      }'
```