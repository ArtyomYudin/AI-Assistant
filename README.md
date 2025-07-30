# AI-Assistant

## Хост:
    сервер IBM x3650 M5, 2х E5-2695 v3, 96GB Ram
    2x AMD Instinct mi50 16Gb
    Ubuntu 24.04
    ROCm 6.3
    vLLM 0.92

## Docker container GPU1:
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
        --max-model-len 20480 \
        --max-num-batched-tokens 20480 \
        --max-seq-len-to-capture 32768 \
        --max-num-seqs 64 \
        --port 8001 \
        --served-model-name Qwen3-8B-AWQ

## Docker container GPU2:
    docker run -d --rm --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 8G \
	    --security-opt seccomp=unconfined \
    	--security-opt apparmor=unconfined \
    	--cap-add=SYS_PTRACE \
    	-v /storage/models:/models \
	    -p 8000:8000 \
        --env CUDA_VISIBLE_DEVICES=1 \
	    nalanzeyu/vllm-gfx906  vllm serve /models/Qwen/Qwen3-Embedding-4B \
	    --swap-space 8 \
    	--disable-log-requests \
    	--dtype float16 \
        --gpu-memory-utilization=0.90\
        --max-model-len 20480 \
        --max-num-batched-tokens 20480 \
        --max-seq-len-to-capture 32768 \
        --max-num-seqs 64 \
        --port 8000 \
        --served-model-name Qwen3-Embedding-4B

## LLM
    LLM model - Qwen3-8B-AWQ
    Embedding model - Qwen3-Embedding-4B


В дальнейшем будет произведена оптимизация для повышения производительности


