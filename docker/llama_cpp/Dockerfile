FROM rocm/dev-ubuntu-22.04:6.3-complete
# Установка зависимостей
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    wget \
    libuv1-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Клонируем llama.cpp с поддержкой ROCm
WORKDIR /app
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /app/llama.cpp

# Собираем с поддержкой ROCm и сервером
RUN make clean && make -j$(nproc) LLAMA_HIPBLAS=1 LLAMA_CUDA=0

# Устанавливаем Python-биндинги (опционально)
RUN pip install -r requirements.txt

# Копируем модель (будет монтироваться извне, но можно и так)
# COPY ./qwen3-8b-q4_k_m.gguf /models/model.gguf

# Экспонируем порт
EXPOSE 8001

# Запуск сервера
CMD ["./server", "-m", "/models/model.gguf", "--port", "8001", "--ctx-size", "4096", "--n-gpu-layers", "40"]