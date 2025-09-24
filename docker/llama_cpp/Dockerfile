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

# Клонируем llama.cpp
WORKDIR /app
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /app/llama.cpp

# Создаём build-директорию
RUN mkdir build && cd build && \
    cmake .. \
      -DLLAMA_HIPBLAS=ON \
      -DLLAMA_CUDA=OFF \
      -DLLAMA_ACCELERATE=OFF \
      -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_SERVER=ON \
      -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# Устанавливаем Python-биндинги (опционально, для будущего использования)
# RUN pip install -r requirements.txt

# Экспонируем порт
EXPOSE 8001

# Запуск сервера
CMD ["./build/bin/server", "-m", "/models/model.gguf", "--port", "8001", "--ctx-size", "4096", "--n-gpu-layers", "40"]