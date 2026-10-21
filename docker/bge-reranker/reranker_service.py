from contextlib import asynccontextmanager

import torch
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

# Диагностика: должно быть "Visible GPUs: 1, current: 0"
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.version.hip: {torch.version.hip}")
if torch.cuda.is_available():
    print(f"Visible GPUs: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU visible to PyTorch. Check devices and HIP_VISIBLE_DEVICES.")


MODEL_PATH = "/models/BAAI/bge-reranker-v2-m3"

model = CrossEncoder(
    MODEL_PATH,
    device="cuda",       # после изоляции HIP_VISIBLE_DEVICES=1 это GPU1
    max_length=8192,
    # model_kwargs={"torch_dtype": torch.float16},
)
# Вместо model_kwargs={"torch_dtype": "float16"}:
model.model.half()

class RerankRequest(BaseModel):
    query: str
    documents: List[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Код при старте (Startup) ---
    print("Reranker warming up...")
    warmup_pairs = [
        ["есть ли камеры на втором этаже?", "Камеры второго этажа: левое крыло, правое крыло"],
    ]
    with torch.inference_mode():
        model.predict(warmup_pairs, batch_size=1)
        model.predict(warmup_pairs, batch_size=1)  # Второй прогон для горячего кэша
    print("Reranker warmed up and ready.")

    yield  # <-- Здесь приложение начинает принимать запросы

    # --- Код при остановке (Shutdown) ---
    # Здесь можно освободить ресурсы, если нужно
    print("Reranker shutting down.")

# Передаём lifespan в FastAPI вместо @app.on_event
app = FastAPI(title="Reranker API", lifespan=lifespan)

@app.post("/v1/rerank")
async def rerank(req: RerankRequest):
    pairs = [[req.query, doc] for doc in req.documents]
    scores = model.predict(pairs, batch_size=64)
    results = [
        {"index": i, "relevance_score": float(score)}
        for i, score in enumerate(scores)
    ]
    return {"results": sorted(results, key=lambda x: x["relevance_score"], reverse=True)}