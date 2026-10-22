# core/tools/runner.py
import asyncio
import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.tools.parser import parse_tool_calls_fallback
from core.tools.registry import TOOL_REGISTRY

logger = logging.getLogger(__name__)


def _extract_last_question(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


# Сколько символов набрать, чтобы понять, что это текст, а не tool_call
_ABORT_THRESHOLD = 20


async def run_tool_loop_stream(llm, messages, tools, max_iters=5):
    """
    Гибридный agentic loop с параллельным retrieval:
      1. Запускаем retrieval в фоне (не блокирует GPU)
      2. Стримим LLM: смотрим первые ~20 символов
         • если это <tool_call> — ждём полный JSON, выполняем tool
         • если это текст — обрываем стрим (не тратим 3 с на выброс)
      3. Если был tool_call — стримим ответ после результата tool
         Если был текст — стримим ответ с уже готовым контекстом
    """
    question = _extract_last_question(messages)

    # 1. Retrieval параллельно с проверкой LLM (I/O-bound, не бьётся с GPU)
    retrieval_task = None
    kb_func = TOOL_REGISTRY.get("search_knowledge_base")
    if kb_func and question:
        retrieval_task = asyncio.create_task(kb_func(query=question))
        logger.debug("Parallel retrieval started for %r", question[:80])

    # 2. Стрим LLM с ранним abort
    llm_with_tools = llm.bind_tools(tools)
    buffer = ""
    is_tool_call = False

    try:
        async for chunk in llm_with_tools.astream(messages):
            content = chunk.content or ""
            if not content:
                continue
            buffer += content
            stripped = buffer.lstrip()

            # Начало tool_call — ждём закрывающий тег
            if stripped.startswith("<tool_call"):
                is_tool_call = True
                if "</tool_call>" in buffer:
                    break
                continue

            # Текст без tool_call — обрываем
            if len(stripped) >= _ABORT_THRESHOLD:
                logger.info(
                    "Aborted text stream after %d chars — likely text response",
                    len(stripped),
                )
                break
    except Exception as e:
        logger.exception("Stream error during intent detection: %s", e)
        if retrieval_task and not retrieval_task.done():
            retrieval_task.cancel()
        yield "Произошла ошибка при обработке запроса."
        return

    # ─── Ветка 1: tool_call ─────────────────────────────────
    if is_tool_call:
        if retrieval_task and not retrieval_task.done():
            retrieval_task.cancel()

        tool_calls = parse_tool_calls_fallback(buffer)
        if not tool_calls:
            logger.error("Cannot parse tool_call from: %r", buffer[:300])
            yield "Не удалось обработать вызов инструмента."
            return

        # Добавляем AIMessage с tool_calls и выполняем
        messages.append(AIMessage(
            content="",
            tool_calls=[
                {
                    "name": tc["name"],
                    "args": tc.get("args", {}),
                    "id": tc.get("id", f"call_{tc['name']}"),
                }
                for tc in tool_calls
            ],
        ))

        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args", {})
            logger.info("▶ Tool call: %s(%s)", name, args)

            func = TOOL_REGISTRY.get(name)
            if func is None:
                result = {"error": f"Unknown tool: {name}"}
            else:
                try:
                    result = await func(**(args or {}))
                except Exception as e:
                    logger.exception("Tool '%s' failed: %s", name, e)
                    result = {"error": str(e)}

            preview = json.dumps(result, ensure_ascii=False)
            logger.info(
                "◀ Tool result %s: %s",
                name,
                preview[:300] + ("…" if len(preview) > 300 else ""),
            )
            messages.append(ToolMessage(
                content=json.dumps(result, ensure_ascii=False),
                tool_call_id=tc.get("id", f"call_{name}"),
            ))

        # Стримим финальный ответ после tool
        logger.info("── streaming answer (after tool) ──")
        async for chunk in llm_with_tools.astream(messages):
            if chunk.content:
                yield chunk.content
        return

    # ─── Ветка 2: текстовый ответ — используем параллельный retrieval ───
    context = ""
    if retrieval_task:
        try:
            result = await retrieval_task
            context = result.get("context", "") or ""
            logger.info(
                "Parallel retrieval finished, context_len=%d", len(context)
            )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("Parallel retrieval failed: %s", e)

    # Заменяем последний HumanMessage — инъектим контекст
    if context:
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                messages[i] = HumanMessage(
                    content=f"Контекст:\n{context}\n\nВопрос: {question}"
                )
                break
    else:
        # Контекста нет — просим LLM честно ответить
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                messages[i] = HumanMessage(
                    content=(
                        f"Контекст не найден в базе знаний. Вопрос: {question}\n"
                        "Если не можешь ответить без контекста — "
                        "скажи «К сожалению, у меня нет информации по этому запросу.»"
                    )
                )
                break

    # Стримим финальный ответ с контекстом
    logger.info("── streaming answer (with context) ──")
    async for chunk in llm_with_tools.astream(messages):
        if chunk.content:
            yield chunk.content