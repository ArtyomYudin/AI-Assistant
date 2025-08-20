# RAG Project — Полный набор (Streaming + FastAPI + Gradio + Test Evaluator)
Включает:
- Индексация локальных PDF/MD/TXT (папка `scraped_data/` на один уровень выше пакета `project/`)
- Milvus (Hybrid: Dense + BM25 sparse) + дедупликация по SHA-256
- Реранк (опционально через vLLM endpoint)
- QA со стримингом ответа
- FastAPI (StreamingResponse + SSE)
- Gradio UI (чат + тестовый вопрос) со стримингом
- Модуль тест-оценки ответа (покрытие по ключевым словам + метрики)

# Быстрый старт
1) Установите зависимости:
```bash
pip install -r requirements.txt
```

2) Подготовьте данные: положите файлы в `scraped_data/`.

3) Индексация + демо (CLI):
```bash
python -m main
```

4) FastAPI (Swagger: /docs):
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
- `POST /index` — индексация (читает `scraped_data/`)
- `POST /ask/stream` — стрим текста
- `GET /ask/sse?question=...` — SSE (text/event-stream)
- `POST /test/stream` — стрим ответа + финальные метрики в JSON-маркере

5) Gradio UI:
```bash
python -m ui.app
```

## Переменные окружения (см. проект/config/rag_config.py)
- `RAG_LLM_NAME` (default: Qwen3-8B-AWQ)
- `RAG_LLM_BASE_URL` (OpenAI-compatible, default: http://localhost:8000/v1)
- `RAG_EMBEDDING_NAME` (default: multilingual-e5-large)
- `RAG_EMBEDDING_BASE_URL` (default: http://localhost:8000/v1)
- `RAG_MILVUS_URI` (default: http://localhost:19530)
- `RAG_COLLECTION` (default: demo_ci_rag)
- `RAG_RECREATE_COLLECTION` (true/false)
- `RAG_CHECK_DUPLICATES` (true/false)
- `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP`
- `RAG_K` / `RAG_FETCH_K`
- `RAG_RERANKER_BASE_URL` (опционально)
- `RAG_MAX_CONTEXT_TOKENS` (по умолчанию 8192)
- `RAG_RESERVED_FOR_COMPLETION` (2048)
- `RAG_RESERVED_FOR_OVERHEAD` (512)

## Заметки
- Размерность dense-вектора определяется автоматически по первому эмбеддингу — коллекция Milvus будет создана с подходящей размерностью.
- Для BM25 добавлена функция на поле `text`, генерирующая `sparse_vector`. Вставлять `sparse_vector` вручную не нужно.
- Дедупликация по `hash` реализована через `query` (точное совпадение), без некорректного ANN-поиска по строковому полю.


## Host:
    server IBM x3650 M5, 2х E5-2695 v3, 96GB Ram
    2x AMD Instinct mi50 16Gb
    Ubuntu 24.04
    ROCm 6.3
    vLLM 0.92

## Docker container GPU1 LLM:
    docker run -d --rm --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 8G \
	    --security-opt seccomp=unconfined \
    	--security-opt apparmor=unconfined \
    	--cap-add=SYS_PTRACE \
    	-v /storage/models:/models \
	    -p 8001:8001 \
	    --env CUDA_VISIBLE_DEVICES=0 \
	    nalanzeyu/vllm-gfx906  vllm serve /models/Qwen/Qwen3-8B-AWQ \
	    --swap-space 8 \
    	--disable-log-requests \
    	--dtype float16 \
        --quantization awq \
        --gpu-memory-utilization=0.90\
        --max-model-len 10240 \
        --max-num-batched-tokens 10240 \
        --max-seq-len-to-capture 32768 \
        --max-num-seqs 64 \
        --port 8001 \
        --served-model-name Qwen3-8B-AWQ

## Docker container GPU2 Embedding:
    docker run -d --rm --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 8G \
	    --security-opt seccomp=unconfined \
    	--security-opt apparmor=unconfined \
    	--cap-add=SYS_PTRACE \
    	-v /storage/models:/models \
	    -p 8000:8000 \
        --env CUDA_VISIBLE_DEVICES=1 \
	    nalanzeyu/vllm-gfx906  vllm serve /models/intfloat/multilingual-e5-large \
	    --swap-space 8 \
    	--disable-log-requests \
    	--dtype float16 \
        --gpu-memory-utilization=0.45\
		--task embed \
        --port 8000 \
        --served-model-name multilingual-e5-large

## Docker container GPU2 Reranker:
    docker run -d --rm --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 8G \
	    --security-opt seccomp=unconfined \
    	--security-opt apparmor=unconfined \
    	--cap-add=SYS_PTRACE \
    	-v /storage/models:/models \
	    -p 8002:8002 \
        --env CUDA_VISIBLE_DEVICES=1 \
	    nalanzeyu/vllm-gfx906  vllm serve /models/BAAI/bge-reranker-v2-m3 \
	    --swap-space 8 \
    	--disable-log-requests \
    	--dtype float16 \
        --gpu-memory-utilization=0.45\
        --port 8002 \
        --served-model-name bge-reranker-v2-m3

## LLM
    LLM model:
        Qwen3-8B-AWQ
    Embedding model:
        Qwen3-Embedding-4B - works poorly with Russian.
        BGE-3m - works poorly with Russian as well.
        e5-mistral-7b-instruct - gives an error 'Token id 98285 is out of vocabulary', could not be fixed.
        Giga-Embeddings-instruct - could not be launched on VLLM.
        multilingual-e5-large - copes better with Russian than the listed ones. Stops at it.



In the future, the launch of LLM models will be optimized to improve performance.
