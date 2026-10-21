FROM rocm/dev-ubuntu-22.04:6.3-complete

# --- базовые зависимости ---
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git wget curl python3 python3-pip python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- ставим PyTorch ROCm 6.3 ---
RUN pip install --upgrade pip
RUN pip install torch==2.7.1+rocm6.3 torchvision==0.22.1+rocm6.3 torchaudio==2.7.1+rocm6.3 \
    --extra-index-url https://download.pytorch.org/whl/rocm6.3

# --- зависимости для inference ---
# Ключевое отличие: добавляем sentence-transformers для CrossEncoder
RUN pip install sentence-transformers==2.7.0 \
    transformers==4.45.2 accelerate fastapi uvicorn pydantic

# --- сервер реранкера ---
WORKDIR /app
COPY reranker_service.py /app/reranker_service.py

EXPOSE 8002
CMD ["uvicorn", "reranker_service:app", "--host", "0.0.0.0", "--port", "8002"]