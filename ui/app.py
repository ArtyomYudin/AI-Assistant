# gradio/app.py
import asyncio
import config.logging_config # noqa: F401 — настраивает логи ДО core.*
import json
import uuid
import logging

import gradio as gr

from config.rag_config import RAGConfig
from core.rag_core import RAGCore

logger = logging.getLogger(__name__)

cfg = RAGConfig()
core: RAGCore | None = None
_warmed_up = False

DEBUG = cfg.GRADIO_DEBUG
GRADIO_USER = cfg.GRADIO_USER
GRADIO_PASSWORD = cfg.GRADIO_PASSWORD


# ─── Инициализация ─────────────────────────────────────────────
async def init_rag():
    global core
    if core is None:
        core = RAGCore(cfg)                        # CollectionManager внутри
        await core.collection_manager.build_router()
        core.create_retriever(k=cfg.K, fetch_k=cfg.FETCH_K)
        core.create_qa_generator()
        logger.info("RAGCore создан (loop=%s)", id(asyncio.get_running_loop()))
    return core


async def init_rag_and_warmup():
    """Прогрев в том же event loop, что использует Gradio."""
    global _warmed_up
    rag = await init_rag()
    if not _warmed_up:
        await rag.warmup()
        _warmed_up = True
    # return rag


def new_session_id() -> str:
    return f"gradio_{uuid.uuid4().hex[:12]}"


# ─── Диалог ───────────────────────────────────────────────────
async def chat_answer_stream(message: str, history: list, session_id: str):
    if not session_id:
        session_id = new_session_id()
    rag = await init_rag()

    full = ""
    buffer = ""
    async for chunk in rag.qa_chain_with_history(message, user_id=session_id):
        text = chunk.get("text") if isinstance(chunk, dict) else str(chunk)
        buffer += text
        if len(buffer) >= 10:
            full += buffer
            buffer = ""
            yield full
    if buffer:
        full += buffer
        yield full


# ─── Тестовый вопрос с метриками (только при DEBUG=true) ──────
async def test_answer_stream(message: str, keywords_csv: str, session_id: str):
    from core.eval import stream_answer_and_evaluate   # ленивый импорт

    if not session_id:
        session_id = new_session_id()
    rag = await init_rag()
    expected = [k.strip() for k in (keywords_csv or "").split(",") if k.strip()]

    full = ""
    async for part in stream_answer_and_evaluate(
        rag, message, expected, session_id=session_id
    ):
        if "[METRICS]" in part:
            text_part, metrics_part = part.split("[METRICS]", 1)
            full += text_part
            try:
                m = json.loads(metrics_part.strip())
                pretty = (
                    f"\n\n⏱ {m['duration_sec']}s | "
                    f"Ключевые: {m['keywords_found']}/{m['keywords_total']} | "
                    f"Покрытие: {m['coverage_percent']}%\n"
                    f"Детали: {m['found_map']}"
                )
                yield full + pretty
            except Exception:
                yield full + "\n\n[Не удалось распарсить метрики]"
        else:
            full += part
            yield full


# ─── UI ───────────────────────────────────────────────────────
with gr.Blocks(title="RAG — Internal UI") as demo:
    gr.Markdown("## 🔎 RAG QA — внутренний инструмент")
    gr.Markdown(
        "⚠️ Только для разработчиков и тестирования. "
        "Продовый RAG — через `/ask/stream` с JWT."
    )

    sid_state = gr.State()
    # Инициализируем уникальный session_id на каждую сессию браузера
    demo.load(fn=new_session_id, inputs=None, outputs=sid_state)

    # Прогрев в Gradio-loop (а не в отдельном через new_event_loop)
    demo.load(fn=init_rag_and_warmup, inputs=None, outputs=None)

    with gr.Tab("Диалог"):
        gr.ChatInterface(
            fn=chat_answer_stream,
            additional_inputs=[sid_state],
            title="QA",
        )

    if DEBUG:
        with gr.Tab("Тестовый вопрос (DEBUG)"):
            prompt = gr.Textbox(label="Вопрос")
            keywords = gr.Textbox(
                label="Ожидаемые ключевые слова (через запятую)",
                placeholder="172.20.58.11, E12A, левое крыло",
            )
            out = gr.Textbox(label="Стрим ответа + метрики", lines=12)
            btn = gr.Button("Проверить и стримить")
            btn.click(
                fn=test_answer_stream,
                inputs=[prompt, keywords, sid_state],
                outputs=out,
            )


if __name__ == "__main__":
    launch_kwargs = {
        "server_name": "127.0.0.1",
        "server_port": 7860,
    }
    if GRADIO_USER and GRADIO_PASSWORD:
        launch_kwargs["auth"] = [(GRADIO_USER, GRADIO_PASSWORD)]
        logger.info("Basic-auth включён для %s", GRADIO_USER)
    else:
        logger.warning("GRADIO_USER/GRADIO_PASSWORD не заданы — запуск без auth")

    demo.queue().launch(**launch_kwargs)