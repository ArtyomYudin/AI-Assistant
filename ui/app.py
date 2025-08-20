import gradio as gr
import asyncio, json
from typing import List

from config.rag_config import RAGConfig
from core.rag_core import RAGCore
from core.eval import stream_answer_and_evaluate

cfg = RAGConfig()
core = RAGCore(cfg)
core.create_retriever(k=cfg.K, fetch_k=cfg.FETCH_K)
core.create_qa_generator()

async def chat_answer_stream(message, history, session_id="gradio"):
    full_response = ""
    async for chunk in core.qa_chain_with_history(message, session_id=session_id):
        full_response += chunk
        yield full_response
        # yield chunk

async def test_answer_stream(message, keywords_csv, history, session_id="gradio-test"):
    expected = [k.strip() for k in (keywords_csv or "").split(",") if k.strip()]
    full_response = ""
    async for chunk in stream_answer_and_evaluate(core, message, expected, session_id=session_id):
        if chunk.startswith("\n\n[METRICS]"):
            try:
                metrics = json.loads(chunk.replace("\n\n[METRICS]",""))
                pretty = (f"\n\n⏱ {metrics['duration_sec']}s  |  "
                          f"Ключевые слова: {metrics['keywords_found']}/{metrics['keywords_total']}  |  "
                          f"Покрытие: {metrics['coverage_percent']}%\n"
                          f"Детали: {metrics['found_map']}")
                yield pretty
            except Exception:
                yield "\n\n[Не удалось распарсить метрики]"
        else:
            full_response += chunk
            yield full_response
            # yield chunk

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
