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
RUN pip install transformers==4.45.2 accelerate fastapi uvicorn pydantic

# --- твой сервер ---
WORKDIR /app
COPY server.py /app/server.py

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]