import gradio as gr
import json
from typing import List

from langchain_core.messages import HumanMessage, AIMessage

from config.rag_config import RAGConfig
from core.rag_core import RAGCore
from core.eval import stream_answer_and_evaluate
from core.chat_history import RedisChatHistory

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

def load_history(session_id="gradio"):
    store = RedisChatHistory(host="172.20.4.50", session_id=session_id)
    msgs = store.get_messages()

    history = []
    for m in msgs:
        if m.type == "human":
            history.append({"role": "user", "content": m.content})
        elif m.type == "ai":
            history.append({"role": "assistant", "content": m.content})
    return history

# --- Стриминговый диалог ---
async def chat_answer_stream(message, history, session_id="gradio"):
    rag = await init_rag()

    full_response = ""
    buffer = ""
    async for chunk in rag.qa_chain_with_history(message, session_id=session_id):
        # --- FIX: chunk может быть dict ---
        text = chunk.get("text") if isinstance(chunk, dict) else str(chunk)

        buffer += text
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
            # buffer += chunk
            text = chunk.get("text") if isinstance(chunk, dict) else str(chunk)
            buffer += text
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
        chat = gr.ChatInterface(fn=chat_answer_stream, type="messages", title="QA")
        # # при загрузке UI — подтягиваем Redis-историю
        # demo.load(lambda: load_history("gradio"), None, chat.chatbot)
    with gr.Tab("Тестовый вопрос"):
        prompt = gr.Textbox(label="Вопрос")
        keywords = gr.Textbox(label="Ожидаемые ключевые слова (через запятую)")
        out = gr.Textbox(label="Стрим ответа + метрики", lines=12)
        btn = gr.Button("Проверить и стримить ответ")
        btn.click(fn=test_answer_stream, inputs=[prompt, keywords, out], outputs=out)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
