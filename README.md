# AI-Assistant · RAG для ИТ-поддержки

Корпоративная RAG-система для поиска по базе знаний отдела ИТО: камеры видеонаблюдения, серверы, VLAN, СКУД, телефония, регламенты и инструкции. Работает на AMD MI50 (gfx906/ROCm), поддерживает гибридный поиск (dense + BM25), кросс-энкодерный реранкер и стриминг ответов через SSE.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Milvus](https://img.shields.io/badge/Milvus-2.6.24-orange.svg)](https://milvus.io/)
[![ROCm](https://img.shields.io/badge/ROCm-6.3-red.svg)](https://rocm.docs.amd.com/)

---

## ✨ Особенности

- **Multi-collection routing** — вопрос автоматически маршрутизируется в одну или несколько коллекций Milvus (`cameras`, `acti_nvr`, `it_runbooks`, …) по косинусной близости центроидов
- **Per-collection threshold** — относительный порог (`max_score × 0.3`) в каждой коллекции, чтобы вторичные источники не проигрывали лидеру на составных вопросах
- **Fallback на все коллекции** — если в выбранных нашлось меньше `N` документов, автоматически доискивает в остальных
- **Гибридный поиск** — HNSW (dense, COSINE) + `SPARSE_INVERTED_INDEX` (BM25), нативный Milvus `FunctionType.BM25`
- **Cross-encoder reranker** — `BAAI/bge-reranker-v2-m3` через `sentence-transformers`, ~140 мс на 30 пар
- **Token-aware контекст** — обрезка истории диалога + резюме старых сообщений + детерминированный порядок документов по `score`
- **Streaming через SSE** — `/ask/sse` отдаёт токены по мере генерации
- **Redis-кэш** — история диалогов (TTL 7 дней) + эмбеддинги запросов (TTL 24 часа)
- **AMD ROCm / gfx906** — протестировано на MI50/MI60 (16–32 ГБ)

---

## 🏗️ Архитектура

```
┌──────────┐   ┌────────────────────┐   ┌──────────────┐
│ Frontend │──▶│ FastAPI /ask/sse   │──▶│   RAGCore    │
└──────────┘   └────────────────────┘   └──────────────┘
                                                │
                    ┌───────────────────────────┼────────────────────────────┐
                    ▼                           ▼                            ▼
           ┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
           │CollectionManager │        │  MilvusManager   │        │ LocalEmbeddings  │
           │  (routing +      │        │  (hybrid_search  │        │  (FRIDA via      │
           │   fallback)      │        │   + reranker)    │        │   HTTP :8000)    │
           └──────────────────┘        └──────────────────┘        └──────────────────┘
                    │                           │
                    ▼                           ▼
           ┌──────────────────┐        ┌──────────────────┐
           │      Redis       │        │  Milvus 2.6.24   │
           │  (router state,  │        │  (HNSW + BM25,   │
           │   history,       │        │   6 collections) │
           │   emb cache)     │        └──────────────────┘
           └──────────────────┘                │
                                               ▼
                                     ┌──────────────────┐
                                     │ bge-reranker-v2- │
                                     │ m3 @ :8002       │
                                     │ (sentence-trans- │
                                     │  formers, GPU1)  │
                                     └──────────────────┘

                                     ┌──────────────────┐
                                     │ vLLM T-lite-     │
                                     │ it-2.1-AWQ       │
                                     │ @ :8001          │
                                     └──────────────────┘
```

### Поток обработки запроса

1. **Polite check** — «привет», «спасибо» и т.п. отсекаются без LLM
2. **Параллельно**: `retriever` + загрузка истории диалога из Redis
3. **Router**: эмбеддинг запроса → косинус к центроидам коллекций → `pick_collections(top_k=2, min_margin=0.1)`
4. **Parallel hybrid_search** в выбранных коллекциях (dense + BM25 + reranker)
5. **Per-collection threshold**: `max(local_max × 0.3, 0.03)` в каждой
6. **Merge + dedup по hash + per-source cap=2**
7. **Fallback**: если `docs < max(2, N_collections)` — доискивает в остальных
8. **Build context**: обрезка истории по `MAX_HISTORY_TOKENS`, сортировка документов по `score`, обрезка контекста по `MAX_CONTEXT_TOKENS`
9. **LLM streaming** через `ChatOpenAI.astream(messages)`
10. **Save history** в Redis

---

## 🧰 Технологический стек

| Компонент | Технология | Роль |
|---|---|---|
| **LLM** | [T-lite-it-2.1-AWQ-4bit](https://huggingface.co/t-tech/T-lite-it-2.1) via vLLM | Генерация ответов |
| **Embeddings** | [FRIDA](https://huggingface.co/ai-forever/FRIDA) (`T5EncoderModel`) | Векторизация документов и запросов |
| **Reranker** | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) via sentence-transformers | Cross-encoder re-ranking |
| **Vector DB** | Milvus 2.6.24 | HNSW (dense) + BM25 (sparse) |
| **Cache / History** | Redis 7 | Router state, chat history, embedding cache |
| **API** | FastAPI + `sse-starlette` | `/ask/sse`, `/health`, `/index` |
| **UI** | Gradio | Чат-интерфейс |
| **PDF parsing** | [Docling](https://github.com/DS4SD/docling) | Извлечение структуры + таблиц |
| **GPU** | AMD Instinct MI50 (gfx906) | 2× карты: LLM на GPU0, embed/rerank на GPU1 |
| **Оркестрация** | Docker Compose | 8 контейнеров |
| **CI/CD** | GitHub Actions → Harbor | Автосборка образов по тегам |

---

## 🚀 Быстрый старт

### Требования

- Docker 24+ и Docker Compose v2
- AMD GPU с ROCm 6.3+ (для GPU-ускорения LLM/embeddings/reranker)
- ≥ 64 ГБ RAM, ≥ 60 ГБ диска под модели (LLM 30 ГБ + FRIDA + reranker + образы)
- Milvus-зависимости: `etcd`, `minio`

### 1. Клонирование

```bash
git clone https://github.com/<your-org>/ai-assistant.git
cd ai-assistant
```

### 2. Скачивание моделей

```bash
mkdir -p /storage/models
# LLM (≈ 4.5 ГБ в AWQ-4bit)
huggingface-cli download t-tech/T-lite-it-2.1-AWQ-4bits --local-dir /storage/models/T-lite-it-2.1-AWQ-4bits
# Embeddings
huggingface-cli download ai-forever/FRIDA --local-dir /storage/models/FRIDA
# Reranker
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir /storage/models/bge-reranker-v2-m3
```

### 3. Конфигурация `.env`

```env
# Redis
REDIS_HOST=172.20.4.50
REDIS_PORT=6379

# Milvus
MILVUS_URI=http://172.20.4.50:19530

# LLM
LLM_NAME=T-lite-it-2.1-AWQ
LLM_MODEL_PATH=/models/T-lite-it-2.1-AWQ-4bits
LLM_BASE_URL=http://172.20.4.50:8001/v1

# Embeddings
EMBEDDING_NAME=ai-forever/FRIDA
EMBEDDING_BASE_URL=http://172.20.4.50:8000/v1

# Reranker
RERANKER_BASE_URL=http://172.20.4.50:8002

# JWT (для /merge-session)
JWT_SECRET=change_me
JWT_ALGO=HS256

# Режим: rag | hybrid | llm_only
MODE=hybrid
```

### 4. Запуск

```bash
docker compose up -d
docker compose logs -f rag-api
```

Сервисы после запуска:

| Сервис | URL |
|---|---|
| RAG API | http://localhost:8080/docs |
| Gradio UI | http://localhost:7860 |
| Embeddings (FRIDA) | http://localhost:8000 |
| Reranker | http://localhost:8002 |
| LLM (vLLM) | http://localhost:8001 |
| Milvus | http://localhost:19530 |

### 5. Индексация документов

```bash
# Индексация всех подпапок DATA_DIR как отдельных коллекций
curl -X POST http://localhost:8080/index
```

Документы кладутся в `scraped_data/<collection_name>/`:

```
scraped_data/
├── cameras/
│   └── cameras.md
├── acti_nvr/
│   └── nvr-servers.md
└── it_runbooks_and_reglaments/
    ├── vlan-registry.md
    └── policy.md
```

### 6. Проверка

```bash
curl -N "http://localhost:8080/ask/sse?question=какие%20камеры%20на%202%20этаже"
```

---

## 📡 API

### `GET /health`

```json
{"status": "ok"}
```

### `POST /index`

Индексирует все подпапки `DATA_DIR` как отдельные коллекции.

```json
{"status": "indexed"}
```

### `GET /ask/sse`

SSE-стриминг ответа. Параметры: `question`, `session_id` (по умолчанию `api-sse`).

```bash
curl -N "http://localhost:8080/ask/sse?question=какой%20сервер%20видеонаблюдения&session_id=test"
```

События:

```
event: token
data: {"type": "status", "text": "📚 Думаю над вашим вопросом...\n"}

event: token
data: {"type": "response", "text": "Сервер ACTi NVR 3.0, IP 172.30.4.135."}

event: done
data: 
```

Типы `type`: `status` | `response` | `info` | `error`.

### `POST /merge-session`

Переносит историю чата из анонимной сессии в пользовательскую после логина.

```bash
curl -X POST "http://localhost:8080/merge-session?old_session_id=anon_abc" \
  -H "Authorization: Bearer <jwt>"
```

---

## ⚙️ Ключевые параметры конфигурации

Все параметры в `config/rag_config.py`, переопределяются через `.env`.

| Параметр | По умолчанию | Описание |
|---|---|---|
| `ROUTER_MIN_MARGIN` | `0.1` | Минимальная разница между top-1 и top-2 коллекцией для выбора второй. Меньше — больше коллекций на запрос |
| `COLLECTION_THRESHOLD_RATIO` | `0.3` | Относительный порог внутри коллекции: `max_score × ratio` |
| `COLLECTION_MIN_ABSOLUTE` | `0.03` | Абсолютный пол, чтобы отсеять «шумные» коллекции |
| `PER_SOURCE_CAP` | `2` | Не более N чанков с одного файла в итоговом контексте |
| `K` | `7` | Сколько документов после реранкера уходит в LLM |
| `FETCH_K` | `20` | Сколько кандидатов берёт Milvus до reranker |
| `CHUNK_SIZE` | `500` | Размер чанка (символы) |
| `CHUNK_OVERLAP` | `100` | Перекрытие чанков |
| `MAX_CONTEXT_TOKENS` | `12800` | Бюджет на контекст LLM |
| `MAX_HISTORY_TOKENS` | `1024` | Бюджет на историю диалога |
| `HYBRID_SEARCH_SCORE_THRESHOLD` | `0.3` | Порог для одиночных коллекций (не в multi-collection пути) |

### Тюнинг

- **Больше контекста, меньше шума**: `PER_SOURCE_CAP=1`, `COLLECTION_THRESHOLD_RATIO=0.4`
- **Составные вопросы** (камеры + сервер): `ROUTER_MIN_MARGIN=0.15`, `PER_SOURCE_CAP=3`
- **Скорость**: `FETCH_K=10`, `K=5`

---

## 📝 Формат документов

Каждый `.md`-документ должен начинаться с:

```markdown
# Заголовок документа
<!-- bm25: ключевые слова синонимы объект подсеть vlan камеры сервер -->

Краткая сводка в 2–3 строки: ключевые факты, IP, ссылки на смежные документы.

## Секция 1
...тело до 500 символов...

## Секция 2
...
```

### Правила

| Элемент | Как оформлять | Зачем |
|---|---|---|
| `# H1` | Один на документ | Идёт в `title` всех чанков |
| `<!-- bm25: ... -->` | 15–25 синонимов, ключевых фраз, IP | Обогащает BM25 во всех чанках; удаляется из тела |
| Вводный абзац | 3–5 ключевых фактов (не «этот документ описывает») | Отвечает на обзорные вопросы |
| `## H2` | Одна логическая единица (этаж, сервис, VLAN) | Отдельный чанк |
| `### H3` | Только если тело H2 > 500 символов | Иначе слить в H2 |
| Тело | До 500 символов, `- **ключ** — значение` | Избегает рекурсивного дробления |

**Плохо** (мета-секции, дублирование):

```markdown
## Ответы на типовые вопросы
### «Где сервер?»
На сервере 172.30.4.135.
```

**Хорошо** (сводка в H1-intro):

```markdown
# Серверы видеонаблюдения
<!-- bm25: сервер nvr acti naomi 172.30.4.135 172.30.4.136 архив -->

- **Главный сервер NVR:** 172.30.4.135 (ACTi NVR 3.0, RAID)
- **Сервер архива:** 172.30.4.136
- **Панель управления:** http://naomi.center-inform.ru
```

---

## 📂 Структура проекта

```
.
├── api/
│   └── app.py                    # FastAPI: /ask/sse, /index, /health
├── config/
│   └── rag_config.py             # Все параметры через env
├── core/
│   ├── rag_core.py               # RAGCore — оркестратор
│   ├── collection_manager.py     # Multi-collection routing + fallback
│   ├── milvus_manager.py         # Hybrid search + reranker
│   ├── splitters.py              # Markdown splitter с bm25-обогащением
│   ├── chat_history.py           # RedisChatHistory
│   ├── embedding_cache.py        # RedisEmbeddingCache
│   ├── local_embeddings.py       # FRIDA через httpx
│   ├── polite_detector.py        # Regex-детектор «спасибо/привет»
│   ├── document_loader.py        # Загрузка MD/TXT
│   ├── document_converter.py     # PDF → Markdown (Docling)
│   └── utils.py                  # count_tokens, hash_text, JWT
├── gradio/
│   └── app.py                    # UI
├── docker/
│   ├── Dockerfile                # RAG API
│   ├── Dockerfile.rag-gradio     # UI
│   ├── Dockerfile.frida-embed    # FRIDA сервер
│   └── Dockerfile.reranker       # bge-reranker сервер
├── scraped_data/                 # Документы по коллекциям
│   ├── cameras/
│   ├── acti_nvr/
│   └── it_runbooks_and_reglaments/
├── scripts/
│   └── reindex.py                # Полная переиндексация
├── docker-compose.yml
└── .env.example
```

---

## 🔧 Развёртывание

### CI/CD

GitHub Actions собирает и пушит образы в Harbor по тегам `v*`:

- `ai-assistant/rag` — основное API
- `ai-assistant/reranker` — reranker-сервис

```bash
git tag v1.2.0 && git push origin v1.2.0
```

### Обновление на сервере

```bash
docker compose pull
docker compose up -d --force-recreate
```

### Переиндексация с нуля

```bash
# Через env-флаг (удаляет все коллекции и Redis-состояние роутера)
DROP_ALL_COLLECTIONS=true docker compose up -d --force-recreate rag-api
# После завершения — вернуть флаг в false
```

---

## ⚠️ Известные ограничения

1. **vLLM на gfx906 не парсит tool-calling** — модель генерирует `<tool_call>{...}}}` с лишней `}`, парсеры `hermes`/`qwen3_xml` не справляются. Решение: fallback-парсер в приложении или `tool_choice="required"` (в текущей версии tool-calling не используется, RAG — основное)
2. **Токенизатор модели** — путь `/models/T-lite-it-2.1-AWQ-4bits` должен содержать HF-токенизатор, иначе используется `tiktoken` с калибровкой ×0.75 под русский
3. **Reranker через sentence-transformers, не через vLLM** — vLLM на ROCm возвращает одинаковые скоры (`0.00065`) для всех пар из-за бага в pooling-слое. Обход: отдельный сервис на `sentence-transformers` + GPU1
4. **BM25-запрос передаётся без `remove_stopwords_russian`** — индексация идёт со стоп-словами, запрос без — recall деградирует. Рекомендуется передавать `query_text` as-is
5. **PDF-документы** — `PyPDF2` даёт плоский текст без таблиц. Для качественной индексации используйте Docling (`core/document_converter.py`) и кэшируйте `.md` рядом с `.pdf`

---

## 🧪 Тестирование

```bash
# E2E проверка через SSE
curl -N "http://localhost:8080/ask/sse?question=покажи+все+камеры+первого+этажа"

# Проверка health
curl http://localhost:8080/health

# Проверка reranker напрямую
curl -X POST http://localhost:8002/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"query":"камеры 2 этаж","documents":["Камеры второго этажа: IP 172.20.58.11","Камеры цокольного этажа: IP 172.20.58.14"]}'
```

Ожидаемый ответ reranker: `0.98` для релевантного, `0.13` для нерелевантного.

---

## 📊 Метрики производительности

Замеры на MI50 (32 ГБ):

| Стадия | Время |
|---|---|
| Router | ~5–10 мс |
| Hybrid search (1 коллекция) | ~100–200 мс |
| Reranker (30 пар) | ~140 мс (fp16) |
| Reranker (2 пары) | ~70 мс |
| Context build | ~5 мс |
| Первый токен LLM | ~500–700 мс |
| Полный ответ (RAG, ~150 токенов) | ~2.5 с |
| Полный ответ (составной, 2 коллекции) | ~3.5 с |

---

## 🤝 Вклад


## 📄 Лицензия


## 🙏 Благодарности

- [vLLM](https://github.com/vllm-project/vllm) — inference LLM
- [Milvus](https://milvus.io/) — vector database
- [sentence-transformers](https://www.sbert.net/) — reranker
- [Docling](https://github.com/DS4SD/docling) — PDF parsing
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) — модель `bge-reranker-v2-m3`
- [t-tech](https://huggingface.co/t-tech) — LLM T-lite-it-2.1
- [SberDevices](https://huggingface.co/ai-forever) — embeddings FRIDA