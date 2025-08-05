# AI-Assistant

## Dependencies
pip install --upgrade --quiet  langchain langchain-core langchain-community langchain-milvus langchain-openai bs4 fastapi \
    uvicorn gradio

## Host:
    server IBM x3650 M5, 2х E5-2695 v3, 96GB Ram
    2x AMD Instinct mi50 16Gb
    Ubuntu 24.04
    ROCm 6.3
    vLLM 0.92

## Docker container GPU1 LLM:
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

## Docker container GPU2 Embedding:
    docker run -d --rm --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 8G \
	    --security-opt seccomp=unconfined \
    	--security-opt apparmor=unconfined \
    	--cap-add=SYS_PTRACE \
    	-v /storage/models:/models \
	    -p 8000:8000 \
        --env CUDA_VISIBLE_DEVICES=1 \
	    nalanzeyu/vllm-gfx906  vllm serve /models/intfloat/multilingual-e5-large \
	    --swap-space 8 \
    	--disable-log-requests \
    	--dtype float16 \
        --gpu-memory-utilization=0.90\
		--task embed \
        --port 8000 \
        --served-model-name multilingual-e5-large

## LLM
    LLM model:
        Qwen3-8B-AWQ
    Embedding model:
        Qwen3-Embedding-4B - works poorly with Russian.
        BGE-3m - works poorly with Russian as well.
        e5-mistral-7b-instruct - gives an error 'Token id 98285 is out of vocabulary', could not be fixed.
        Giga-Embeddings-instruct - could not be launched on VLLM.
        multilingual-e5-large - copes better with Russian than the listed ones. Stops at it.



In the future, the launch of LLM models will be optimized to improve performance.


# Start
uvicorn main:app --host 0.0.0.0 --port 8000 --reload




