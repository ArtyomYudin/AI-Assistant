import os
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, T5EncoderModel
import torch
import logging

logger = logging.getLogger(__name__)

# --- Конфигурация ---
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "ai-forever/FRIDA")
MODEL_LOCAL_PATH = f"/models/{MODEL_NAME}"
BATCH_SIZE = 16   # максимальный размер батча
BATCH_TIMEOUT = 0.05  # время ожидания перед запуском (сек)

# --- Загрузка модели ---
if os.path.exists(MODEL_LOCAL_PATH):
    logger.info(f"Загружаем модель локально из {MODEL_LOCAL_PATH}")
    model_path = MODEL_LOCAL_PATH
else:
    logger.info(f"Локальная модель не найдена, загружаем из HuggingFace: {MODEL_NAME}")
    model_path = MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = T5EncoderModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # --- Определяем оптимальный dtype ---
# if device.type == "cuda":
#     if torch.cuda.is_bf16_supported():
#         dtype = torch.bfloat16
#         logger.info("Загружаем модель в BF16")
#         model = model.to(dtype).to(device)
#     else:
#         dtype = torch.float32
#         logger.info("BF16 не поддерживается, работаем в FP32")
#         model = model.to(dtype).to(device)
# else:
#     dtype = torch.float32
#     logger.info("GPU нет, работаем в FP32")
#     model = model.to(dtype).to(device)

model = model.to(device)
model.eval()

# --- FastAPI ---
app = FastAPI(title="FRIDA Embeddings API (CLS + Batching + Queue)")

class EmbeddingRequest(BaseModel):
    model: str
    input: List[str]

class EmbeddingResponse(BaseModel):
    data: List[List[float]]

# --- Очередь для батчинга ---
queue: asyncio.Queue = asyncio.Queue()

class BatchItem:
    def __init__(self, text: str):
        self.text = text
        self.future: asyncio.Future = asyncio.get_event_loop().create_future()

def cls_pooling(model_output):
    """CLS токен"""
    return model_output.last_hidden_state[:, 0]

async def batch_worker():
    """Фоновый воркер для сбора батчей"""
    while True:
        batch: List[BatchItem] = []

        # ждём первый элемент
        item = await queue.get()
        batch.append(item)

        # собираем остальных с таймаутом
        try:
            while len(batch) < BATCH_SIZE:
                item = await asyncio.wait_for(queue.get(), timeout=BATCH_TIMEOUT)
                batch.append(item)
        except asyncio.TimeoutError:
            pass

        # --- инференс ---
        texts = [it.text for it in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)

        embeddings = cls_pooling(outputs)  # CLS pooling
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        embeddings = embeddings.cpu().tolist()

        # --- возвращаем каждому ---
        for it, emb in zip(batch, embeddings):
            if not it.future.done():
                it.future.set_result(emb)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_worker())
    # прогрев (без autocast, всё уже в BF16/FP32)
    dummy = tokenizer("warmup", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**dummy)
    print("FRIDA Embeddings API запущен и прогрет")

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    tasks = []
    for text in request.input:
        logger.warning(f"Получены токены вместо текста: {text[:10]}... (len={len(text)})")
        # --- Предохранитель ---
        if isinstance(text, list):
            logger.warning(f"Получены токены вместо текста: {text[:10]}... (len={len(text)})")
            text = " ".join(map(str, text))
        elif not isinstance(text, str):
            logger.warning(f"Получен нестроковый input: {type(text)} -> {text}")
            text = str(text)
        item = BatchItem(text)
        await queue.put(item)
        tasks.append(item.future)

    results = await asyncio.gather(*tasks)
    return EmbeddingResponse(data=results)
