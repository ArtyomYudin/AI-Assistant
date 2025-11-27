import time, json
from typing import AsyncGenerator, Dict, List, Tuple

def _coverage(answer: str, expected_keywords: List[str]) -> Tuple[int, int, float, Dict[str,bool]]:
    text = (answer or "").lower()
    found_map = {k: (k.lower() in text) for k in expected_keywords}
    found = sum(1 for v in found_map.values() if v)
    total = max(1, len(expected_keywords))
    score = round(100.0 * found / total, 2)
    return found, total, score, found_map

async def stream_answer_and_evaluate(core, question: str, expected_keywords: List[str], session_id: str = "test") -> AsyncGenerator[str, None]:
    start = time.time()
    buffer = ""
    async for chunk in core.qa_chain_with_history(question, session_id=session_id):
        # buffer += chunk
        # yield chunk  # пробрасываем стрим наружу

        # --- FIX: chunk может быть dict ---
        text = chunk.get("text") if isinstance(chunk, dict) else str(chunk)
        buffer += text
        yield text  # пробрасываем наружу только строку
    dur = time.time() - start
    found, total, score, fmap = _coverage(buffer, expected_keywords or [])
    metrics = {
        "duration_sec": round(dur, 3),
        "keywords_total": total,
        "keywords_found": found,
        "coverage_percent": score,
        "found_map": fmap,
    }
    # финальный маркер метрик (парсится на клиенте)
    yield f"\n\n[METRICS]{json.dumps(metrics, ensure_ascii=False)}"
