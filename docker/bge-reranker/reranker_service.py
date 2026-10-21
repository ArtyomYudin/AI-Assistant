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

app = FastAPI()

MODEL_PATH = "/models/BAAI/bge-reranker-v2-m3"

model = CrossEncoder(
    MODEL_PATH,
    device="cuda",       # после изоляции HIP_VISIBLE_DEVICES=1 это GPU1
    max_length=8192,
)

class RerankRequest(BaseModel):
    query: str
    documents: List[str]

@app.post("/v1/rerank")
async def rerank(req: RerankRequest):
    pairs = [[req.query, doc] for doc in req.documents]
    scores = model.predict(pairs, batch_size=8)
    results = [
        {"index": i, "relevance_score": float(score)}
        for i, score in enumerate(scores)
    ]
    return {"results": sorted(results, key=lambda x: x["relevance_score"], reverse=True)}