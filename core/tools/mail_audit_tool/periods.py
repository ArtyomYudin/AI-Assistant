# core/tools/mail_audit_tool/periods.py

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

"""Границы периода → аргументы --since/--until для journalctl."""
from datetime import datetime, timedelta, timezone

from .config import QUERY_TIMEZONE

VALID_PERIODS = frozenset({"today", "yesterday", "7days", "all"})


def _parse_exact_date(period: str) -> datetime | None:
    """Пробует распарсить строку строгого формата YYYY-MM-DD."""
    if len(period) != 10:
        return None
    try:
        return datetime.strptime(period, "%Y-%m-%d")
    except ValueError:
        return None


def is_valid_period(period: str) -> bool:
    """Проверяет, является ли строка допустимым периодом или датой YYYY-MM-DD."""
    return period in VALID_PERIODS or _parse_exact_date(period) is not None


def _get_time_args(period: str) -> list[str]:
    """Возвращает аргументы journalctl для --since/--until.

    'all' → без ограничений (весь доступный журнал). Для второго trace-запроса
    окно сужается в check_mail_logs по oldest/latest из первого ответа.
    """
    if period == "all":
        return []

    exact_date = _parse_exact_date(period)

    if QUERY_TIMEZONE is None:
        if period == "today":
            return ["--since", "today 00:00:00"]
        if period == "yesterday":
            return [
                "--since", "yesterday 00:00:00",
                "--until", "today 00:00:00",
            ]
        if period == "7days":
            return ["--since", "7 days ago"]
        if exact_date is not None:
            next_day = exact_date + timedelta(days=1)
            return [
                "--since", exact_date.strftime("%Y-%m-%d 00:00:00"),
                "--until", next_day.strftime("%Y-%m-%d 00:00:00"),
            ]
        return []

    from zoneinfo import ZoneInfo
    tz = ZoneInfo(QUERY_TIMEZONE)
    now = datetime.now(tz)

    def _utc(dt: datetime) -> str:
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if period == "today":
        since = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return ["--since", _utc(since)]
    if period == "yesterday":
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return ["--since", _utc(today - timedelta(days=1)),
                "--until", _utc(today)]
    if period == "7days":
        return ["--since", _utc(now - timedelta(days=7))]
    if exact_date is not None:
        since = exact_date.replace(tzinfo=tz)
        until = since + timedelta(days=1)
        return [
            "--since", _utc(since),
            "--until", _utc(until),
        ]

    return []