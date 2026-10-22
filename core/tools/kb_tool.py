# core/tools/kb_tool.py
import logging

from core.context import current_session_id

logger = logging.getLogger(__name__)


# Глобальная ссылка на RAGCore, чтобы не тянуть его через аргументы
_rag_core_ref = None


def set_rag_core(core) -> None:
    """Устанавливает глобальную ссылку на RAGCore при старте."""
    global _rag_core_ref
    _rag_core_ref = core


def _get_rag_core():
    if _rag_core_ref is None:
        raise RuntimeError("RAGCore не инициализирован (set_rag_core не вызван)")
    return _rag_core_ref


KB_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Поиск по базе знаний ИТ-поддержки: камеры видеонаблюдения, "
                "серверы, VLAN, сетевое оборудование, СКУД, телефония, "
                "виртуализация, регламенты, инструкции. "
                "Используй для ВСЕХ вопросов, кроме погоды и точных lookup'ов "
                "по VLAN по номеру/порту/имени."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Поисковый запрос на русском, сформулированный из "
                            "вопроса пользователя. Сохраняй ключевые сущности "
                            "(номера этажей, имена систем, IP, названия объектов)."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
]


async def search_knowledge_base(query: str) -> dict:
    """
    Обёртка над текущим RAG-пайплайном.
    Возвращает готовый контекст для LLM.
    """
    core = _get_rag_core()
    sid = current_session_id.get()

    logger.info("[%s] 🔎 search_knowledge_base: query=%r", sid, query)

    try:
        docs = await core.retriever(query, sid)
    except Exception as e:
        logger.exception("search_knowledge_base failed: %s", e)
        return {"found": False, "error": str(e), "context": ""}

    # === ЛОГИРОВАНИЕ RETRIEVAL (как в старой версии) ===
    logger.info(
        "[%s] retrieval: docs=%d, collections=%s, scores=%s, sources=%s",
        sid,
        len(docs),
        [d.metadata.get("collection") for d in docs],
        [round(d.metadata.get("score", 0.0), 3) for d in docs],
        [d.metadata.get("source") for d in docs],
    )
    for d in docs:
        logger.info(
            "  chunk: %s | %r",
            d.metadata.get("title"),
            d.page_content[:80],
        )
    # ================================================

    if not docs:
        return {"found": False, "context": ""}

    parts = []
    for d in docs:
        src = d.metadata.get("source", "N/A")
        parts.append(f"[Источник: {src}] {d.page_content.strip()}")

    context = "\n\n".join(parts)

    # === ПОЛНЫЙ КОНТЕКСТ, КОТОРЫЙ УЙДЁТ В LLM ===
    logger.info("[%s] КОНТЕКСТ ДЛЯ LLM:\n%s", sid, context)
    # ============================================

    return {
        "found": True,
        "context": context,
        "sources": [d.metadata.get("source") for d in docs],
        "scores": [round(d.metadata.get("score", 0.0), 3) for d in docs],
    }