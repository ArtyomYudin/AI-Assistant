# core/rate_limiter.py
import asyncio
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Deque, Dict

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Лимит не получен за отведённое время или превышен per-session лимит."""


class LLMRateLimiter:
    """
    Двухуровневый rate-limit для LLM:

    1. Concurrent — Semaphore ограничивает число одновременных LLM-стримов.
       Лишние запросы ждут в очереди до max_queue_wait секунд.
       Защищает vLLM от OOM и деградации TTFT под нагрузкой.

    2. Per-session — sliding window на 60 сек по session_id.
       Защищает от «шумного» пользователя, который бомбит /ask/sse.

    Служебные эндпоинты (/health, /index) не проходят через limiter.
    """

    def __init__(
        self,
        max_concurrent: int = 4,
        max_queue_wait: float = 30.0,
        max_per_session_per_min: int = 10,
    ):
        self._max_concurrent = max_concurrent
        self._sem = asyncio.Semaphore(max_concurrent)
        self._max_queue_wait = max_queue_wait
        self._max_per_session_per_min = max_per_session_per_min
        self._session_windows: Dict[str, Deque[float]] = defaultdict(deque)
        self._session_lock = asyncio.Lock()

    async def _check_session_limit(self, session_id: str) -> None:
        if self._max_per_session_per_min <= 0:
            return  # выключено

        now = time.monotonic()
        window_start = now - 60.0

        async with self._session_lock:
            q = self._session_windows[session_id]
            while q and q[0] < window_start:
                q.popleft()
            if len(q) >= self._max_per_session_per_min:
                raise RateLimitExceeded(
                    f"Превышен лимит {self._max_per_session_per_min} "
                    f"запросов в минуту. Попробуйте через минуту."
                )
            q.append(now)

    @asynccontextmanager
    async def acquire(self, session_id: str = "global"):
        # 1) Per-session rate limit
        await self._check_session_limit(session_id)

        # 2) Concurrent limit с очередью
        try:
            await asyncio.wait_for(
                self._sem.acquire(), timeout=self._max_queue_wait
            )
        except asyncio.TimeoutError:
            logger.warning(
                "LLM queue timeout %.1fs (session=%s, max_concurrent=%d)",
                self._max_queue_wait, session_id, self._max_concurrent,
            )
            raise RateLimitExceeded(
                f"Сервис перегружен: очередь ожидания LLM превысила "
                f"{self._max_queue_wait:.0f} сек. Попробуйте через минуту."
            )

        try:
            yield
        finally:
            self._sem.release()

    def is_available(self) -> bool:
        """Best-effort проверка для UX (racy, но для статуса хватает)."""
        return self._sem._value > 0

    def stats(self) -> dict:
        return {
            "max_concurrent": self._max_concurrent,
            "available_slots": self._sem._value,
            "max_queue_wait_s": self._max_queue_wait,
            "max_per_session_per_min": self._max_per_session_per_min,
        }