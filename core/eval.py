# core/eval.py

# Copyright 2025 Artyom Yudin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time, json
from typing import AsyncGenerator, Dict, List, Tuple

def _coverage(answer: str, expected_keywords: List[str]) -> Tuple[int, int, float, Dict[str,bool]]:
    text = (answer or "").lower()
    found_map = {k: (k.lower() in text) for k in expected_keywords}
    found = sum(1 for v in found_map.values() if v)
    total = max(1, len(expected_keywords))
    score = round(100.0 * found / total, 2)
    return found, total, score, found_map

async def stream_answer_and_evaluate(
    core, question: str, expected_keywords: List[str], session_id: str = "test"
) -> AsyncGenerator[str, None]:
    start = time.time()
    buffer = ""
    async for chunk in core.qa_chain_with_history(question, user_id=session_id):
        if isinstance(chunk, dict):
            # Служебные/ошибочные — не идут в coverage
            if chunk.get("type") in ("error", "status", "info"):
                yield chunk.get("text", "")
                continue
            text = chunk.get("text", "")
        else:
            text = str(chunk)
        buffer += text
        yield text
    dur = time.time() - start
    found, total, score, fmap = _coverage(buffer, expected_keywords or [])
    yield f"\n\n[METRICS]{json.dumps({
        'duration_sec': round(dur, 3),
        'keywords_total': total,
        'keywords_found': found,
        'coverage_percent': score,
        'found_map': fmap,
    }, ensure_ascii=False)}"