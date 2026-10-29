# core/rate_limiter.py

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

import asyncio
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Deque, Dict

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Per-user rate limit или таймаут очереди на LLM."""


class LLMRateLimiter:
    """
    Двухуровневый rate-limit на LLM.

    1. Concurrent — Semaphore ограничивает число одновременных стримов.
       Лишние запросы ждут в очереди до max_queue_wait секунд.
       Защита vLLM от OOM и деградации TTFT.

    2. Per-user — sliding window на 60 сек по user_id (sub из JWT).
       Защита от «шумного» пользователя, который бомбит /ask/sse.

    Служебные эндпоинты (/health) не проходят через limiter.
    """

    def __init__(
        self,
        max_concurrent: int = 4,
        max_queue_wait: float = 30.0,
        max_per_user_per_min: int = 10,
    ):
        self._max_concurrent = max_concurrent
        self._sem = asyncio.Semaphore(max_concurrent)
        self._max_queue_wait = max_queue_wait
        self._max_per_user_per_min = max_per_user_per_min

        self._user_windows: Dict[str, Deque[float]] = defaultdict(deque)
        self._windows_lock = asyncio.Lock()

        # Счётчик активных стримов — для is_available()/stats().
        # Не требует lock: инкремент/декремент без await между ними,
        # в asyncio это атомарно (single-threaded event loop).
        self._active: int = 0

    # ─────────────────────────────────────────────────────────────
    # Per-user rate limit
    # ─────────────────────────────────────────────────────────────

    async def _check_user_limit(self, user_id: str) -> None:
        if self._max_per_user_per_min <= 0:
            return  # выключено

        now = time.monotonic()
        window_start = now - 60.0

        async with self._windows_lock:
            q = self._user_windows[user_id]
            while q and q[0] < window_start:
                q.popleft()
            if len(q) >= self._max_per_user_per_min:
                raise RateLimitExceeded(
                    f"Превышен лимит {self._max_per_user_per_min} "
                    f"запросов в минуту. Попробуйте через минуту."
                )
            q.append(now)

    # ─────────────────────────────────────────────────────────────
    # Concurrent limit
    # ─────────────────────────────────────────────────────────────

    @asynccontextmanager
    async def acquire(self, user_id: str = "unknown"):
        # 1) Per-user rate limit
        await self._check_user_limit(user_id)

        # 2) Concurrent limit с очередью
        try:
            await asyncio.wait_for(
                self._sem.acquire(), timeout=self._max_queue_wait
            )
        except asyncio.TimeoutError:
            logger.warning(
                "LLM queue timeout %.1fs (user=%s, max_concurrent=%d)",
                self._max_queue_wait, user_id, self._max_concurrent,
            )
            raise RateLimitExceeded(
                f"Сервис перегружен: очередь ожидания LLM превысила "
                f"{self._max_queue_wait:.0f} сек. Попробуйте через минуту."
            )

        self._active += 1
        try:
            yield
        finally:
            self._active -= 1
            self._sem.release()

    # ─────────────────────────────────────────────────────────────
    # Диагностика
    # ─────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Best-effort проверка для UX-статуса (racy, но не для логики)."""
        return self._active < self._max_concurrent

    def stats(self) -> dict:
        return {
            "max_concurrent": self._max_concurrent,
            "active": self._active,
            "available_slots": max(0, self._max_concurrent - self._active),
            "max_queue_wait_s": self._max_queue_wait,
            "max_per_user_per_min": self._max_per_user_per_min,
        }

    def reset_user(self, user_id: str) -> None:
        """Сбрасывает окно rate-limit пользователя (для тестов/админ-действий)."""
        self._user_windows.pop(user_id, None)