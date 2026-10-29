# AI-Assistant · RAG for IT Support

**English** · [Русский](README.ru.md)

Enterprise-grade RAG system for the IT support knowledge base: CCTV cameras, servers, VLANs, access control, telephony, regulations, and runbooks. Runs on AMD Instinct MI50 (gfx906 / ROCm) and supports hybrid retrieval (dense + BM25), cross-encoder re-ranking, tool-calling, and SSE streaming.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Milvus](https://img.shields.io/badge/Milvus-2.6.24-orange.svg)](https://milvus.io/)
[![ROCm](https://img.shields.io/badge/ROCm-6.3-red.svg)](https://rocm.docs.amd.com/)
[![vLLM](https://img.shields.io/badge/vLLM-0.32.0-blueviolet.svg)](https://github.com/vllm-project/vllm)

---

## ✨ Features

- **Tool-calling works on gfx906** — after switching to `unverbraucht/vllm-gfx906:0.32.0-rocm-7.14` with the `hermes` tool parser, the tool flow works end-to-end. No fallback JSON parser needed.
- **Multi-collection routing** — a question is automatically routed to one or more Milvus collections (`cameras`, `acti_nvr`, `it_runbooks`, …) by cosine similarity of collection centroids.
- **Per-collection threshold** — a relative threshold (`max_score × 0.3`) is applied inside each collection so secondary sources don't lose to the leader on compound questions.
- **Fallback to all collections** — if fewer than `N` documents were found in the selected collections, the retriever automatically scans the rest.
- **Hybrid search** — HNSW (dense, COSINE) + `SPARSE_INVERTED_INDEX` (BM25) via native Milvus `FunctionType.BM25`.
- **Small-to-big expansion** — for broad-intent queries ("all cameras", "full list"), the whole source document is pulled into the context using the `chunk_index` ordering. Smart mode detects broad intent and lowers the threshold.
- **Follow-up detection** — short pronoun-type queries ("which ones?", "more", "show me") stay on the previously used collection.
- **Cross-encoder re-ranking** — `BAAI/bge-reranker-v2-m3` via `sentence-transformers`, ~140 ms per 30 pairs on MI50.
- **Token-aware context** — history truncation by tokens, summary of skipped messages, deterministic document ordering by score, `source`-aware fallback.
- **Streaming via SSE** — `/ask/sse` yields tokens as they are generated.
- **Redis cache** — chat history (TTL 7 days) + query embeddings (TTL 24 hours) + router state.
- **Two-tier LLM rate limiter** — concurrent stream limit (`LLM_MAX_CONCURRENT`) with queue, plus a sliding-window per-user limit (`LLM_MAX_PER_USER_PER_MIN`).
- **AMD ROCm / gfx906** — tested on MI50 / MI60 (16–32 GB).

---

## 🏗️ Architecture

```
┌──────────┐   ┌──────────────────────┐   ┌──────────────┐
│ Frontend │──▶│ FastAPI /ask/sse     │──▶│   RAGCore    │
└──────────┘   └──────────────────────┘   └──────────────┘
                                                │
                    ┌───────────────────────────┼────────────────────────────┐
                    ▼                           ▼                            ▼
           ┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
           │CollectionManager │        │  MilvusManager   │        │ LocalEmbeddings  │
           │  (routing +      │        │  (hybrid_search  │        │  (FRIDA via      │
           │   expansion +    │        │   + reranker)    │        │   HTTP :8000)    │
           │   fallback)      │        └──────────────────┘        └──────────────────┘
           └──────────────────┘                │
                    │                          │
                    ▼                          ▼
           ┌──────────────────┐        ┌──────────────────┐
           │      Redis       │        │  Milvus 2.6.24   │
           │  (router state,  │        │  (HNSW + BM25,   │
           │   history,       │        │   N collections) │
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
                                     │ + tool-calling   │
                                     └──────────────────┘
```

### Request pipeline

1. **Polite check** — "hi", "thanks", etc. are answered without the LLM.
2. **Parallel**: `retriever` + history from Redis + tool-check routing.
3. **Tool-check** — the model decides whether a tool is needed (`get_weather`, …). If yes, tools run in parallel, and the final answer is generated with the tool result injected as `ToolMessage`. If no, the branch below is used.
4. **Router**: query embedding → cosine to collection centroids → `pick_collections(top_k=2, min_margin=0.1)`.
5. **Parallel hybrid search** in the selected collections (dense + BM25 + reranker).
6. **Per-collection threshold**: `max(local_max × 0.3, 0.03)` inside each collection.
7. **Merge + dedup by hash + per-source cap = 2**.
8. **Small-to-big expansion** — for broad intent or high-scoring top chunk, pull all chunks of that source from the collection.
9. **Fallback**: if `docs < max(2, N_collections)` — scan the remaining collections.
10. **Build context**: trim history by `MAX_HISTORY_TOKENS`, group by `source`, order by score, truncate by `MAX_CONTEXT_TOKENS`.
11. **LLM streaming** through `ChatOpenAI.astream(messages)`.
12. **Save history** to Redis.

---

## 🧰 Tech stack

| Component | Technology | Role |
|---|---|---|
| **LLM** | [T-lite-it-2.1-AWQ-4bit](https://huggingface.co/t-tech/T-lite-it-2.1) via vLLM `unverbraucht/vllm-gfx906:0.32.0-rocm-7.14` | Answer generation + tool-calling (`hermes` parser) |
| **Embeddings** | [FRIDA](https://huggingface.co/ai-forever/FRIDA) (`T5EncoderModel`) | Document & query vectorization |
| **Reranker** | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) via `sentence-transformers` | Cross-encoder re-ranking |
| **Vector DB** | Milvus 2.6.24 | HNSW (dense) + BM25 (sparse) |
| **Cache / History** | Redis 7 | Router state, chat history, embedding cache |
| **API** | FastAPI + `sse-starlette` | `/ask/sse`, `/ask/stream`, `/health`, `/index` |
| **UI** | Gradio | Chat interface |
| **PDF parsing** | [Docling](https://github.com/DS4SD/docling) | Structured extraction + tables |
| **GPU** | AMD Instinct MI50 (gfx906) | 2× cards: LLM on GPU0, embed/rerank on GPU1 |
| **Orchestration** | Docker Compose | 8 containers |
| **CI/CD** | GitHub Actions → Harbor | Tag-based image builds |

---

## 🚀 Quick start

### Requirements

- Docker 24+ and Docker Compose v2
- AMD GPU with ROCm 6.3+ (for GPU-accelerated LLM / embeddings / reranker)
- ≥ 64 GB RAM, ≥ 60 GB disk for models (LLM ≈ 4.5 GB AWQ + FRIDA + reranker + images)
- Milvus dependencies: `etcd`, `minio`

### 1. Clone

```bash
git clone https://github.com/<your-org>/ai-assistant.git
cd ai-assistant
```

### 2. Download models

```bash
mkdir -p /storage/models
# LLM (~4.5 GB in AWQ-4bit)
huggingface-cli download t-tech/T-lite-it-2.1-AWQ-4bits \
  --local-dir /storage/models/pimenovdv/T-lite-it-2.1-AWQ-4bits
# Embeddings
huggingface-cli download ai-forever/FRIDA \
  --local-dir /storage/models/FRIDA
# Reranker
huggingface-cli download BAAI/bge-reranker-v2-m3 \
  --local-dir /storage/models/bge-reranker-v2-m3
```

### 3. Configure `.env`

```env
# Redis
REDIS_HOST=YOUR_HOST_IP
REDIS_PORT=6379

# Milvus
MILVUS_URI=http://YOUR_HOST_IP:19530

# LLM
LLM_NAME=T-lite-it-2.1-AWQ
LLM_MODEL_PATH=/models/pimenovdv/T-lite-it-2.1-AWQ-4bits
LLM_BASE_URL=http://YOUR_HOST_IP:8001/v1

# Embeddings
EMBEDDING_NAME=ai-forever/FRIDA
EMBEDDING_BASE_URL=http://YOUR_HOST_IP:8000/v1

# Reranker
RERANKER_BASE_URL=http://YOUR_HOST_IP:8002

# JWT (for /merge-session, /index)
JWT_SECRET=change_me
JWT_ALGO=HS256

# Mode: rag | hybrid | llm_only
MODE=hybrid
```

> The `*_BASE_URL` values currently live in `config/rag_config.py` and are **not** read from env. Set your host IP there, or override the fields in a subclass. This will be moved to env in the next iteration.

### 4. Launch

```bash
docker compose -f docker-compose-frida.yaml up -d
docker compose -f docker-compose-frida.yaml logs -f rag-api
```

Services after startup:

| Service | URL |
|---|---|
| RAG API | http://localhost:8008/docs |
| Gradio UI | http://localhost:7860 |
| Embeddings (FRIDA) | http://localhost:8000 |
| Reranker | http://localhost:8002 |
| LLM (vLLM) | http://localhost:8001 |
| Milvus | http://localhost:19530 |

### 5. Index documents

```bash
# Index every subfolder of DATA_DIR as a separate collection
curl -X POST http://localhost:8008/index \
  -H "Authorization: Bearer <jwt>"
```

Documents live in `scraped_data/<collection_name>/`:

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

### 6. Smoke test

```bash
curl -N "http://localhost:8008/ask/sse?question=which+cameras+are+on+the+2nd+floor" \
  -H "Authorization: Bearer <jwt>"
```

---

## 📡 API

### `GET /health`

```json
{
  "status": "ok",
  "checks": {
    "milvus": "ok",
    "redis": "ok",
    "embeddings": "ok",
    "llm": "ok"
  },
  "llm_limiter": {
    "max_concurrent": 2,
    "active": 0,
    "available_slots": 2,
    "max_queue_wait_s": 30,
    "max_per_user_per_min": 10
  }
}
```

### `POST /index`

Indexes every subfolder of `DATA_DIR` as a separate Milvus collection.

```json
{"status": "indexed", "triggered_by": "user-sub"}
```

### `GET /ask/sse`

SSE streaming. Requires JWT. `session_id = user_id` (history is bound to the user).

```bash
curl -N "http://localhost:8008/ask/sse?question=which+camera+server" \
  -H "Authorization: Bearer <jwt>"
```

Event stream:

```
event: token
data: {"type": "status", "text": "🔍 Looking up information...\n"}

event: token
data: {"type": "response", "text": "ACTi NVR 3.0 server, IP 172.30.4.135."}

event: done
data: 
```

`type` values: `status` | `response` | `info` | `error`.

### `POST /ask/stream`

NDJSON streaming for clients without SSE. Same JWT requirement.

### `POST /merge-session`

Moves chat history from an anonymous session into an authenticated one after login.

```bash
curl -X POST "http://localhost:8008/merge-session?old_session_id=anon_abc" \
  -H "Authorization: Bearer <jwt>"
```

---

## ⚙️ Key configuration

All parameters live in `config/rag_config.py` and are overridable via `.env` (where marked).

| Parameter | Default | Description |
|---|---|---|
| `ROUTER_MIN_MARGIN` | `0.1` | Min gap between top-1 and top-2 collection to include the second |
| `COLLECTION_THRESHOLD_RATIO` | `0.3` | Relative threshold inside a collection: `max_score × ratio` |
| `COLLECTION_MIN_ABSOLUTE` | `0.03` | Absolute floor to drop noisy collections |
| `PER_SOURCE_CAP` | `2` | Max chunks per file in the final context |
| `K` | `7` | Documents to keep after reranker |
| `FETCH_K` | `20` | Candidates Milvus returns before reranker |
| `CHUNK_SIZE` | `500` | Chunk size (chars) |
| `CHUNK_OVERLAP` | `100` | Chunk overlap |
| `MAX_CONTEXT_TOKENS` | `5120` | LLM context budget |
| `MAX_HISTORY_TOKENS` | `1024` | History budget for the RAG branch |
| `MAX_TOOL_HISTORY_TOKENS` | `2048` | History budget for tool-check |
| `LLM_MAX_CONCURRENT` | `2` | Max concurrent LLM streams |
| `LLM_MAX_QUEUE_WAIT` | `30` | Max seconds a request waits for a free slot |
| `LLM_MAX_PER_USER_PER_MIN` | `10` | Per-user sliding window (0 disables) |
| `EXPANSION_MODE` | `smart` | `off` / `always` / `smart` (broad-intent aware) |
| `EXPANSION_THRESHOLD` | `0.6` | Top-score threshold to expand a source |
| `MAX_EXPANDED_SOURCES` | `2` | Cap on sources to expand |
| `MAX_CHUNKS_PER_SOURCE` | `20` | Safety cap when pulling a source |
| `EXPANSION_SKIP_MAX_MATCHED` | `1` | Skip expansion if matched chunks ≤ N |
| `EXPANSION_SKIP_SCORE` | `0.9` | and top-score ≥ this (narrow, confident query) |
| `HYBRID_SEARCH_SCORE_THRESHOLD` | `0.3` | Threshold for single-collection path |

### Tuning recipes

- **More context, less noise**: `PER_SOURCE_CAP=1`, `COLLECTION_THRESHOLD_RATIO=0.4`
- **Compound questions** (cameras + servers): `ROUTER_MIN_MARGIN=0.15`, `PER_SOURCE_CAP=3`
- **Speed**: `FETCH_K=10`, `K=5`, `EXPANSION_MODE=off`
- **Broad "list everything" queries**: `EXPANSION_MODE=always`

---

## 📝 Document format

Every `.md` document should start with:

```markdown
# Document title
<!-- bm25: keywords synonyms object subnet vlan cameras server -->

Short 2–3 line summary: key facts, IPs, links to related docs.

## Section 1
...body up to 500 chars...

## Section 2
...
```

### Rules

| Element | How to write | Why |
|---|---|---|
| `# H1` | One per document | Becomes `title` for every chunk |
| `<!-- bm25: ... -->` | 15–25 synonyms, key phrases, IPs | Enriches BM25 in every chunk; stripped from body |
| Intro paragraph | 3–5 key facts (not "this doc describes…") | Answers overview questions |
| `## H2` | One logical unit (floor, service, VLAN) | Becomes one chunk |
| `### H3` | Only if the H2 body exceeds 500 chars | Otherwise keep it in H2 |
| Body | Up to 500 chars, `- **key** — value` | Avoids recursive splitting |

**Bad** (meta sections, duplications):

```markdown
## FAQ
### "Where is the server?"
The server is at 172.30.4.135.
```

**Good** (summary in H1 intro):

```markdown
# CCTV servers
<!-- bm25: server nvr acti naomi 172.30.4.135 172.30.4.136 archive -->

- **Main NVR server:** 172.30.4.135 (ACTi NVR 3.0, RAID)
- **Archive server:** 172.30.4.136
- **Control panel:** http://naomi.example.com
```

---

## 📂 Project structure

```
.
├── api/
│   └── app.py                    # FastAPI: /ask/sse, /index, /health
├── config/
│   └── rag_config.py             # All parameters via env
├── core/
│   ├── rag_core.py               # RAGCore — orchestrator
│   ├── collection_manager.py     # Multi-collection routing + expansion + fallback
│   ├── milvus_manager.py         # Hybrid search + reranker
│   ├── splitters.py              # Markdown splitter with bm25 enrichment
│   ├── chat_history.py           # RedisChatHistory
│   ├── embedding_cache.py        # RedisEmbeddingCache
│   ├── local_embeddings.py       # FRIDA via httpx
│   ├── polite_detector.py        # Regex "thanks / hi" detector
│   ├── rate_limiter.py           # Two-tier LLM limiter
│   ├── eval.py                   # Streaming keyword coverage eval
│   ├── context.py                # ContextVar for session_id
│   ├── document_loader.py        # MD/TXT loader
│   ├── document_converter.py     # PDF → Markdown (Docling)
│   ├── tools/
│   │   ├── registry.py           # ALL_TOOLS / TOOL_REGISTRY
│   │   ├── runner.py             # Streaming tool-loop
│   │   ├── parser.py             # Robust JSON tool-call parser
│   │   ├── weather_tool.py       # get_weather (wttr.in)
│   │   └── kb_tool.py            # search_knowledge_base
│   └── utils.py                  # count_tokens, hash_text, JWT
├── gradio/
│   └── app.py                    # Gradio UI
├── docker/
│   ├── Dockerfile                # RAG API
│   ├── Dockerfile.rag-gradio     # UI
│   ├── Dockerfile.frida-embed    # FRIDA server
│   └── Dockerfile.reranker       # bge-reranker server
├── scraped_data/                 # Documents per collection
├── docker-compose-frida.yaml
└── .env.example
```

---

## 🔧 Deployment

### CI/CD

GitHub Actions builds and pushes images to Harbor on `v*` tags:

- `ai-assistant/rag` — main API + UI
- `ai-assistant/reranker` — reranker service

```bash
git tag v1.2.0 && git push origin v1.2.0
```

### Update on the server

```bash
docker compose -f docker-compose-frida.yaml pull
docker compose -f docker-compose-frida.yaml up -d --force-recreate
```

### Full re-index

```bash
# Wipes all collections and Redis router state
DROP_ALL_COLLECTIONS=true docker compose -f docker-compose-frida.yaml up -d --force-recreate rag-api
# After it's done, set the flag back to false
```

---

## ⚠️ Known limitations

1. **Reranker runs via `sentence-transformers`, not vLLM** — vLLM on ROCm returns identical scores (`0.00065`) for every pair due to a bug in the pooling layer. Workaround: separate service on `sentence-transformers` + GPU1.
2. **Tokenizer path** — `/models/pimenovdv/T-lite-it-2.1-AWQ-4bits` must contain a HF tokenizer; otherwise `tiktoken` with ×0.75 calibration for Russian is used. Set `LLM_MODEL_PATH` in docker-compose for accurate counting.
3. **BM25 query is passed without stopword removal** — indexing is done with stopwords, query without → recall degrades. Recommend passing `query_text` as-is (current behaviour).
4. **PDF documents** — `PyPDF2` produces flat text without tables. For quality indexing use Docling (`core/document_converter.py`) and cache `.md` files next to `.pdf`.
5. **`*_BASE_URL` values are not read from env** — they are hardcoded in `config/rag_config.py`. Set your host IP there, or override in a subclass.
6. **Tool registry is currently minimal** — only `get_weather` is active. `search_knowledge_base` is commented out in `core/tools/registry.py` while KB tools are consolidated.

---

## 🧪 Testing

```bash
# E2E via SSE
curl -N "http://localhost:8008/ask/sse?question=show+all+cameras+on+the+first+floor" \
  -H "Authorization: Bearer <jwt>"

# Health
curl http://localhost:8008/health

# Reranker directly
curl -X POST http://localhost:8002/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "2nd floor cameras",
    "documents": [
      "Second-floor cameras: IP 172.20.58.11",
      "Basement cameras: IP 172.20.58.14"
    ]
  }'
```

Expected reranker output: `0.98` for the relevant document, `0.13` for the irrelevant one.

---

## 📊 Performance

Measured on MI50 (32 GB):

| Stage | Time |
|---|---|
| Router | ~5–10 ms |
| Hybrid search (1 collection) | ~100–200 ms |
| Reranker (30 pairs) | ~140 ms (fp16) |
| Reranker (2 pairs) | ~70 ms |
| Context build | ~5 ms |
| First LLM token | ~500–700 ms |
| Full RAG answer (~150 tokens) | ~2.5 s |
| Full compound answer (2 collections) | ~3.5 s |

---

## 🤝 Contributing

Contributions are welcome. Please open an issue or a pull request on GitHub. For significant changes, discuss the design first.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -am 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📄 License

*This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.*

---

## 🙏 Acknowledgements

- [vLLM](https://github.com/vllm-project/vllm) — LLM inference
- [Milvus](https://milvus.io/) — vector database
- [sentence-transformers](https://www.sbert.net/) — reranker
- [Docling](https://github.com/DS4SD/docling) — PDF parsing
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) — `bge-reranker-v2-m3`
- [t-tech](https://huggingface.co/t-tech) — LLM T-lite-it-2.1
- [SberDevices](https://huggingface.co/ai-forever) — FRIDA embeddings