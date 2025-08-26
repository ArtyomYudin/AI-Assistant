import gradio as gr
import asyncio, json
from typing import List

from config.rag_config import RAGConfig
from core.rag_core import RAGCore
from core.eval import stream_answer_and_evaluate

# --- Глобальный core (инициализация внутри async) ---
cfg = RAGConfig()
core: RAGCore | None = None


# --- Асинхронная инициализация RAG ---
async def init_rag():
    global core
    if core is None:
        core = RAGCore(cfg)
        core.create_retriever(k=cfg.K, fetch_k=cfg.FETCH_K)
        core.create_qa_generator()
    return core


# --- Стриминговый диалог ---
async def chat_answer_stream(message, history, session_id="gradio"):
    rag = await init_rag()

    full_response = ""
    buffer = ""

    async for chunk in rag.qa_chain_with_history(message, session_id=session_id):
        buffer += chunk
        # отдаём партиями, например каждые 10 символов
        if len(buffer) >= 10:
            full_response += buffer
            buffer = ""
            yield full_response

    if buffer:
        full_response += buffer
        yield full_response


# --- Стриминговый тест с метриками ---
async def test_answer_stream(message, keywords_csv, history, session_id="gradio-test"):
    rag = await init_rag()

    expected = [k.strip() for k in (keywords_csv or "").split(",") if k.strip()]
    full_response = ""
    buffer = ""

    async for chunk in stream_answer_and_evaluate(rag, message, expected, session_id=session_id):
        if "[METRICS]" in chunk:
            parts = chunk.split("[METRICS]", 1)
            text_part = parts[0]
            metrics_part = parts[1] if len(parts) > 1 else ""

            if text_part:
                buffer += text_part
                full_response += buffer
                buffer = ""

            try:
                metrics = json.loads(metrics_part.strip())
                pretty = (f"\n\n⏱ {metrics['duration_sec']}s  |  "
                          f"Ключевые слова: {metrics['keywords_found']}/{metrics['keywords_total']}  |  "
                          f"Покрытие: {metrics['coverage_percent']}%\n"
                          f"Детали: {metrics['found_map']}")
                yield full_response + pretty
            except Exception:
                yield full_response + "\n\n[Не удалось распарсить метрики]"
        else:
            buffer += chunk
            if len(buffer) >= 10:
                full_response += buffer
                buffer = ""
                yield full_response

    if buffer:
        full_response += buffer
        yield full_response


# --- Интерфейс Gradio ---
with gr.Blocks(title="RAG — Streaming UI") as demo:
    gr.Markdown("## 🔎 RAG QA (Streaming)")
    with gr.Tab("Диалог"):
        gr.ChatInterface(fn=chat_answer_stream, type="messages", title="QA")
    with gr.Tab("Тестовый вопрос"):
        prompt = gr.Textbox(label="Вопрос")
        keywords = gr.Textbox(label="Ожидаемые ключевые слова (через запятую)")
        out = gr.Textbox(label="Стрим ответа + метрики", lines=12)
        btn = gr.Button("Проверить и стримить ответ")
        btn.click(fn=test_answer_stream, inputs=[prompt, keywords, out], outputs=out)


if __name__ == "__main__":
    demo.queue().launch()
