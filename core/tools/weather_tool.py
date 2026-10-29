# core/tools/weather_tool.py

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

import logging
import httpx
import time
import re

from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Возвращает текущую погоду в указанном городе: температуру, "
                "ощущаемую температуру, влажность, скорость ветра. "
                "Используй, когда пользователь спрашивает про погоду, "
                "температуру на улице, осадки, ветер в конкретном городе."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": (
                            "Город, например 'СПб', 'Санкт-Петербург', 'Москва'. "
                            "Используй как написал пользователь, не переводи."
                        ),
                    }
                },
                "required": ["city"],
            },
        },
    },
]

# ── Конфигурация ──────────────────────────────────────────────────
WTTR_URL = "https://wttr.in"
# Custom-format: t=temp, f=feels, h=humidity, w=wind, p=precip, C=condition
WTTR_FORMAT = "%t|%f|%h|%w|%p|%C"

# Маленький ответ — таймаут можно держать коротким
_TIMEOUT = httpx.Timeout(connect=3.0, read=6.0, write=2.0, pool=2.0)

_CACHE: Dict[str, Tuple[dict, float]] = {}
_CACHE_TTL = 600.0  # 10 минут

# Circuit breaker: после N подряд ошибок — пауза
_CB_FAILS = 0
_CB_OPEN_UNTIL = 0.0
_CB_THRESHOLD = 2
_CB_COOLDOWN = 60.0


def _cb_is_open() -> bool:
    return time.monotonic() < _CB_OPEN_UNTIL


def _cb_fail() -> None:
    global _CB_FAILS, _CB_OPEN_UNTIL
    _CB_FAILS += 1
    if _CB_FAILS >= _CB_THRESHOLD:
        _CB_OPEN_UNTIL = time.monotonic() + _CB_COOLDOWN
        logger.warning("wttr.in: circuit breaker OPEN на %.0fс", _CB_COOLDOWN)


def _cb_ok() -> None:
    global _CB_FAILS, _CB_OPEN_UNTIL
    _CB_FAILS = 0
    _CB_OPEN_UNTIL = 0.0


# ── Парсинг ───────────────────────────────────────────────────────
_NUM_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def _num(s: str) -> Optional[float]:
    """Извлекает первое число из строки ('+10°C' → 10.0, '85%' → 85.0)."""
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group().replace(",", "."))
    except ValueError:
        return None


def _parse_wttr(line: str, city: str) -> Optional[dict]:
    """
    Парсит строку формата '%t|%f|%h|%w|%p|%C'.
    Пример: '+10°C|+7°C|85%|↖14km/h|0.0mm|Пасмурно'
    """
    parts = [p.strip() for p in line.strip().split("|")]
    if len(parts) < 6:
        logger.warning("wttr.in: неожиданный формат (%d полей): %r", len(parts), line)
        return None

    temp = _num(parts[0])
    feels = _num(parts[1])
    if temp is None:
        # '%t' вернул что-то нечисловое (например, 'Unknown') — плохой ответ
        logger.warning("wttr.in: не удалось разобрать температуру из %r", parts[0])
        return None

    return {
        "city": city,
        "temperature_c": temp,
        "feels_like_c": feels if feels is not None else temp,
        "humidity_pct": _num(parts[2]),
        "wind_speed_kmh": _num(parts[3]),
        "precipitation_mm": _num(parts[4]) or 0.0,
        "condition": parts[5] or "—",
    }


# ── Основная функция ──────────────────────────────────────────────
async def get_weather(city: str) -> dict:
    """
    Текущая погода через wttr.in (custom format).
    При любой ошибке — {'error': '...'}, никогда не бросает.
    """
    city_clean = city.strip()
    if not city_clean:
        return {"error": "Не указан город"}

    if _cb_is_open():
        return {"error": "Сервис погоды временно недоступен, попробуйте позже"}

    now = time.monotonic()
    cached = _CACHE.get(city_clean)
    if cached and now - cached[1] < _CACHE_TTL:
        logger.info("Погода из кэша для '%s'", city_clean)
        return cached[0]

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                f"{WTTR_URL}/{city_clean}",
                params={"format": WTTR_FORMAT, "lang": "ru"},
            )
            resp.raise_for_status()
            raw = resp.text.strip()

        # wttr.in на неизвестный город возвращает что-то вроде
        # "Unknown location; please try ~<lat>,<lon>"
        if "unknown" in raw.lower() or "not found" in raw.lower():
            logger.info("wttr.in: город не найден: %r", city_clean)
            return {"error": f"Город '{city_clean}' не найден"}

        result = _parse_wttr(raw, city_clean)
        if result is None:
            _cb_fail()
            return {"error": "Сервис погоды вернул некорректный ответ"}

        _CACHE[city_clean] = (result, now)
        _cb_ok()
        logger.info(
            "Погода '%s': %s°C, %s",
            city_clean, result["temperature_c"], result["condition"],
        )
        return result

    except httpx.TimeoutException:
        _cb_fail()
        logger.warning("wttr.in: таймаут для '%s'", city_clean)
        return {"error": "Сервис погоды не отвечает"}
    except httpx.HTTPStatusError as e:
        _cb_fail()
        logger.warning("wttr.in: HTTP %s для '%s'", e.response.status_code, city_clean)
        return {"error": f"Сервис погоды вернул ошибку {e.response.status_code}"}
    except Exception as e:
        _cb_fail()
        logger.exception("wttr.in: ошибка для '%s': %s", city_clean, e)
        return {"error": "Не удалось получить погоду"}