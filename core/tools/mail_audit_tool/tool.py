# core/tools/mail_audit_tool/tool.py

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


"""Schema + entrypoint check_mail_logs."""
import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from .client import _get_client, AsyncJournalClient
from .config import (
    QUERY_TIMEZONE, RSPAMD_TIMEOUT_S,
    POSTFIX_IDENTIFIERS, KNOWN_FINAL_STATUSES,
    VALID_PERIODS, PERIOD_ALIASES, MAX_MESSAGES_TOKENS, MAX_UNIQUE_ADDRESSES, MAIL_ATTACH_TABLE_THRESHOLD
)
from .parser import _parse_full_logs, _localize_timestamps
from .patterns import RE_QID
from .periods import _get_time_args, is_valid_period

# count_tokens — реальная метрика для vLLM (лимит 12288 токенов).
# Байты и символы некоррелированы с токенами: кириллица 2 байта/символ
# в UTF-8, но ~2.5 символа/токен; английский ~4 символа/токен. Поэтому
# бюджет messages[] считаем в токенах, а не в байтах.
from core.common.text import count_tokens
from ..context import attach_table

logger = logging.getLogger(__name__)

MAIL_AUDIT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "check_mail_logs",
            "description": (
                "Ищет письма в логах почтового сервера (Postfix + Rspamd) по "
                "отправителю и/или получателю. Для каждого письма возвращает: "
                "timestamp, queue_id, from, to (список получателей), status "
                "(delivered / bounced / deferred / rejected_by_filter / "
                "rejected_at_connection), reason отказа, rspamd-данные (если "
                "запрошены), а также latest_any / latest_delivered / latest_rejected "
                "— для ответа «когда было последнее».\n\n"
                "Подходит для любых вопросов про почту: «приходило ли от X», "
                "«кому отправлял X», «что на info@», «когда последнее», "
                "«почему заблокировано», «какой spam-score»."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sender": {
                        "type": "string",
                        "description": (
                            "Email-адрес или домен отправителя. Заполняй, когда "
                            "пользователь называет автора письма: «от X», «что "
                            "отправлял X», «кому писал X». Оставь пустым, если "
                            "вопрос только про получателя."
                        ),
                    },
                    "recipient": {
                        "type": "string",
                        "description": (
                            "Email-адрес или домен получателя. Заполняй, когда "
                            "пользователь называет адресата: «на info@», «приходило "
                            "ли X-у», «писал ли кто-то на X». Оставь пустым, если "
                            "вопрос только про отправителя."
                        ),
                    },
                    "period": {
                        "type": "string",
                        "pattern": r"^(today|yesterday|7days|all|\d{4}-\d{2}-\d{2})$",
                        "default": "all",
                        "description": (
                            "Период поиска. Выбирай строго по смыслу вопроса:\n"
                            "• 'today' / 'yesterday' / '7days' — ТОЛЬКО если пользователь "
                            "явно назвал период: «сегодня», «вчера», «за неделю», "
                            "«за последние 7 дней».\n"
                            "• 'YYYY-MM-DD' — конкретная дата (например, '2026-11-16'), если в вопросе названо число.\n"
                            "• 'all' — ВСЕГДА, если в вопросе нет явной даты. Сюда относятся:\n"
                            "  – «приходило ли письмо от X»\n"
                            "  – «когда последний раз приходило письмо от X»\n"
                            "  – «есть ли письма от X»\n"
                            "  – «было ли что-то от X»\n"
                            "  – «последнее письмо от X»\n"
                            "  НЕ используй 'today' по умолчанию — это главная ошибка. "
                            "Если сомневаешься — выбирай 'all'."
                        ),
                    },
                    "count_only": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Если true — возвращает ТОЛЬКО счётчики (total_found, "
                            "delivered_count, rejected_count) и latest_*, без массива messages. "
                            "Используй для вопросов «сколько?», «есть ли?», «было ли?», "
                            "«приходило ли?» — когда нужен факт/количество, а не список писем. "
                            "Экономит 10–15K токенов. Для «покажи последние» / «что приходило» "
                            "оставляй false."
                        ),
                    },
                    "include_rspamd": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Включать ли данные Rspamd (score, symbols) в ответ. "
                            "Устанавливай true ТОЛЬКО если пользователь явно спрашивает про "
                            "спам: «какой spam-score», «какие rules сработали», «почему "
                            "rspamd заблокировал», «какой вес у письма». "
                            "Для всех остальных вопросов («приходило ли», «когда последнее», "
                            "«было ли что-то», «доставлено ли») — оставляй false. "
                            "Это в 5–10 раз быстрее."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "default": 15,
                        "minimum": 1,
                        "maximum": 50,
                        "description": (
                            "Максимум писем в messages[]. ВАЖНО: если вопрос про "
                            "заблокированные/недоставленные — ставь limit≥5, иначе "
                            "в messages[] попадёт только первое (часто delivered), "
                            "а нужный rejected придётся брать из latest_rejected. "
                            "Для «покажи список» ставь limit=10."
                        ),
                    },
                },
                "required": [],
            },
        },
    }
]



async def _query_qids_batched(
    ssh: AsyncJournalClient,
    units: list[str],
    time_args: list[str],
    qids: list[str],
    batch_size: int = 12,
    identifiers: Optional[list[str]] = None,
) -> list[str]:
    """Разбивает qid'ы на батчи и выполняет query параллельно.

    PCRE с 50 альтернативами стоит ~6с, с 12 — ~1.5с. 4 батча по 12
    параллельно = max(1.5с) вместо 6с.
    """
    if len(qids) <= batch_size:
        return await ssh.query(
            units=units, time_args=time_args,
            grep_pattern="|".join(qids),
            identifiers=identifiers,
        )
    batches = [
        qids[i:i + batch_size] for i in range(0, len(qids), batch_size)
    ]
    results = await asyncio.gather(
        *[ssh.query(
            units=units, time_args=time_args,
            grep_pattern="|".join(b),
            identifiers=identifiers,
        ) for b in batches],
        return_exceptions=True,
    )
    lines: list[str] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.warning("trace batch %d failed: %s", i, r)
            continue
        lines.extend(r)
    return lines


def _collect_unique_addresses(
    messages: list[dict],
    field: str,
    limit: int = MAX_UNIQUE_ADDRESSES,
) -> tuple[list[dict], int]:
    """Собирает уникальные адреса из поля field ('from' или 'to').

    Возвращает (топ-N по частоте, всего уникальных).

    Сортировка: count desc. При равенстве — порядок вставки, а messages
    уже отсортированы desc по timestamp, значит при равной частоте
    самый свежий отправитель будет выше.

    Зачем: на вопрос «от кого эти письма?» без dedup модель получает
    15 писем от одного support@ и отвечает «15 писем от support@...,
    support@..., support@...» — дубли. С unique_senders — одна запись
    с count=15.

    Используется в full-flow ответе. В count_only не нужен — там нет
    messages[] и вопрос «сколько», а не «от кого».
    """
    counts: dict[str, int] = {}
    latest: dict[str, str] = {}
    for m in messages:
        val = m.get(field)
        if not val:
            continue
        # from — строка, to — список; нормализуем к списку
        items = val if isinstance(val, list) else [val]
        for item in items:
            if not item:
                continue
            counts[item] = counts.get(item, 0) + 1
            if item not in latest:
                # первое вхождение = самое свежее (messages sorted desc)
                latest[item] = m["timestamp"]

    total = len(counts)
    sorted_addrs = sorted(counts.keys(), key=lambda k: -counts[k])
    top = [
        {"addr": addr, "count": counts[addr], "last_timestamp": latest[addr]}
        for addr in sorted_addrs[:limit]
    ]
    return top, total


def _compact_message(m: Optional[dict], symbols_top: int = 5) -> Optional[dict]:
    """Компактизирует одно сообщение для отправки в LLM.

    - Убирает null/пустые поля (reason, rspamd_score, rspamd_symbols).
    - rspamd_symbols обрезает до top-N по |weight|; полное число — в
      rspamd_symbols_total.
    - reason обрезается до 200 символов.
    - Экономит 60–80% токенов на письмо без потери смысла.
    """
    if m is None:
        return None
    out: dict = {
        "timestamp": m["timestamp"],
        "queue_id": m["queue_id"],
    }
    if m.get("from"):
        out["from"] = m["from"]
    if m.get("to"):
        out["to"] = m["to"]
    out["status"] = m["status"]

    if m.get("reason"):
        reason = m["reason"]
        out["reason"] = reason[:200] + ("…" if len(reason) > 200 else "")

    if m.get("rspamd_score") is not None:
        out["rspamd_score"] = m["rspamd_score"]
    if m.get("rspamd_action"):
        out["rspamd_action"] = m["rspamd_action"]

    syms = m.get("rspamd_symbols") or []
    if syms:
        def _weight(s: str) -> float:
            try:
                return abs(float(s.rsplit("(", 1)[1].rstrip(")")))
            except Exception:
                return 0.0
        top = sorted(syms, key=_weight, reverse=True)[:symbols_top]
        out["rspamd_symbols"] = top
        if len(syms) > symbols_top:
            out["rspamd_symbols_total"] = len(syms)
    return out


def _build_summary(
    total: int,
    delivered: int,
    rejected: int,
    deferred: int = 0,
    bounced: int = 0,
) -> str:
    """
    Готовая фраза-сводка для LLM. Снижает риск галлюцинации на числовых
    полях: модель не «складывает» total/delivered/rejected в голове,
    а читает готовый текст.
    """
    if total == 0:
        return "Писем не найдено"

    parts = []
    if delivered:
        parts.append(f"{delivered} доставлено")
    if rejected:
        parts.append(f"{rejected} отклонено")
    if deferred:
        parts.append(f"{deferred} отложено")
    if bounced:
        parts.append(f"{bounced} возвращено")

    if not parts:
        return f"Всего {total} писем, статусы не определены"
    if delivered == total:
        return f"Всего {total} писем, все доставлены"
    if rejected == total:
        return f"Всего {total} писем, все отклонены"
    return f"Всего {total} писем: " + ", ".join(parts)


# ---------- Tool entrypoint ----------

async def check_mail_logs(
    sender: Optional[str] = None,
    recipient: Optional[str] = None,
    period: str = "all",
    include_rspamd: bool = False,
    count_only: bool = False,
    limit: int = 15,
) -> str:
    """Ищет письма в логах Postfix и Rspamd на удаленном сервере."""
    # Normalize period
    p = (period or "").strip().lower()
    p = PERIOD_ALIASES.get(p, p)
    if not is_valid_period(p):
        return json.dumps({
            "error": (
                f"invalid period: {period!r}. "
                f"Allowed: {sorted(VALID_PERIODS)} or 'YYYY-MM-DD'"
            )
        }, ensure_ascii=False)
    period = p

    # Normalize limit
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 15
    limit = max(1, min(50, limit))

    time_args = _get_time_args(period)

    # (?i) — PCRE2-флаг journalctl -g, регистронезависимый поиск.
    # re.escape нужен, чтобы точки/плюсы в email не трактовались как regex.
    search_term = sender or recipient
    safe_grep = f"(?i){re.escape(search_term)}" if search_term else None

    # Итоговый статус запроса к rspamd. Устанавливается в разных ветках ниже
    # и попадает в ответ, чтобы модель знала, есть ли score/symbols и почему.
    #   not_requested     — include_rspamd=False
    #   skipped_no_qids   — include_rspamd=True, но нет qid'ов для трассировки
    #   skipped_count_only — count_only=True, trace не запускался (см. ниже)
    #   included          — данные rspamd получены (могут быть пустыми)
    #   timeout           — rspamd-trace не уложился в RSPAMD_TIMEOUT_S
    #   error             — rspamd-trace упал с другой ошибкой
    rspamd_status = "not_requested"

    try:
        ssh = _get_client()

        # --- ЭТАП 1: быстрый поиск упоминаний в Postfix ---

        # Для period='all' сканируем журнал с конца и берём только последние
        # --reverse -n N останавливает journald после N матчей.
        # N = limit * 10 (по умолчанию 150) — этого хватает на «когда последнее»
        # и «первые N писем». Для «сколько всего за 22 дня» — невозможно
        # без полного скана (30+с), но такой вопрос и не предполагает
        # period='all' без явного уточнения.
        # Минимум 50 — на случай limit=1, чтобы было что показать.
        # Максимум 500 — чтобы не повторить 5-секундный скан.
        initial_max_lines = (
            min(max(limit * 10, 50), 500) if period == "all" else None
        )

        initial_lines = await ssh.query(
            units=["postfix@-.service"],
            time_args=time_args,
            grep_pattern=safe_grep,
            max_lines=initial_max_lines,
            identifiers=POSTFIX_IDENTIFIERS,
        )
        if not initial_lines:
            resp = {
                "total_found": 0,
                "delivered_count": 0,
                "rejected_count": 0,
                "returned": 0,
                "truncated": False,
                "scan_truncated": False,
                "period": period,
                "timezone": QUERY_TIMEZONE or "UTC",
                "rspamd_status": rspamd_status,
                "latest_any": None,
                "latest_delivered": None,
                "latest_rejected": None,
                "messages": [],
            }
            if period != "all":
                resp["hint"] = (
                    f"Период '{period}' не дал результатов. Если пользователь "
                    f"не ограничивал поиск явно ('сегодня'/'вчера'/...), "
                    f"повтори вызов с period='all'."
                )
            return json.dumps(resp, ensure_ascii=False)

        # --- ЭТАП 2: подготовка trace-запроса ---
        # Собираем qid'ы и временные границы найденных писем.
        qids: set[str] = set()
        oldest_ts: Optional[str] = None
        latest_ts: Optional[str] = None
        for line in initial_lines:
            m = RE_QID.search(line)
            if m:
                qids.add(m.group(1))
            ts = line.split(maxsplit=1)[0] if line else ""
            if ts:
                if oldest_ts is None or ts < oldest_ts:
                    oldest_ts = ts
                if latest_ts is None or ts > latest_ts:
                    latest_ts = ts

        # Для period='all' окно trace-запроса сужается по реальным границам
        # найденных писем. Без этого rspamd-trace сканировал бы весь журнал
        # (7+ дней на тесте, месяцы на проде) — это и было 8.5 сек.
        trace_time_args = time_args
        if period == "all" and oldest_ts and latest_ts:
            try:
                oldest_dt = datetime.strptime(oldest_ts, "%Y-%m-%dT%H:%M:%S%z")
                latest_dt = datetime.strptime(latest_ts, "%Y-%m-%dT%H:%M:%S%z")
                trace_time_args = [
                    "--since", (oldest_dt - timedelta(hours=1)).astimezone(
                        timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "--until", (latest_dt + timedelta(hours=1)).astimezone(
                        timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                ]
            except ValueError:
                pass  # не распарсили timestamp — оставляем окно как было

        # Парсим первый результат ДО второго запроса: если все статусы
        # уже финальные (delivered/bounced/deferred/...), второй запрос
        # не нужен, и мы экономим 3–4 секунды.
        parsed_initial = _parse_full_logs(
            initial_lines, sender, recipient, known_qids=qids,
        )

        # ★ COUNT_ONLY FAST PATH: пропускаем весь trace-этап.
        #   Для вопросов «сколько?», «есть ли?», «было ли?» данные о
        #   конкретных письмах не нужны — только счётчики (parsed_initial
        #   уже даёт delivered_count / rejected_count по статусам из
        #   первого запроса) и latest_* (первый элемент по каждому статусу).
        #
        #   Экономия:
        #     - по времени: 3–5с (не запускаем trace по qid'ам)
        #     - по контексту: 12K+ токенов → ~200–500 токенов
        #   Раньше именно этот сценарий ронял vLLM с 400 «max context
        #   length is 12288 tokens»: 50 писем × 30 rspamd-symbols = 12K+.
        #
        #   include_rspamd при count_only игнорируется: score/symbols —
        #   per-message данные, они не агрегируются в «сколько». Если
        #   пользователю нужен spam-score — модель должна вызвать
        #   count_only=false, include_rspamd=true, limit=1..3.
        if count_only:
            parsed = parsed_initial
            if include_rspamd:
                rspamd_status = "skipped_count_only"
        else:
            # ─── Кого обогащать trace'ом ────────────────────────────────
            # 1) unknown_qids — промежуточный статус, нужно дотянуть статус
            unknown_qids = [
                m["queue_id"]
                for m in parsed_initial
                if m["queue_id"] and m["status"] not in KNOWN_FINAL_STATUSES
            ]

            # 2) no_from_qids — recipient-запросы: grep по to=<> не матчит
            #    qmgr-строки, где живёт from=<>. Обогащаем только те, что
            #    реально попадут в ответ (limit) или в latest_*, а не все 50 —
            #    иначе trace по 50 qid'ам = 6+ секунд.
            visible_qids = {
                m["queue_id"]
                for m in parsed_initial[:limit]
                if m["queue_id"] and not m["from"]
            }
            # latest_any/latest_delivered считаются после финального parsed,
            # поэтому здесь — по parsed_initial (те же поля, тот же порядок).
            # Если нужен from у latest — добавь его qid.
            if parsed_initial:
                la = parsed_initial[0]
                if la["queue_id"] and not la["from"]:
                    visible_qids.add(la["queue_id"])
                ld = next(
                    (m for m in parsed_initial if m["status"] == "delivered"),
                    None,
                )
                if ld and ld["queue_id"] and not ld["from"]:
                    visible_qids.add(ld["queue_id"])
            no_from_qids = list(visible_qids)

            if include_rspamd:
                # Для rspamd обогащаем ещё и топ-N свежих (score может быть нужен
                # и для delivered-писем). parsed_initial уже отсортирован desc.
                sorted_with_qid = sorted(
                    (m for m in parsed_initial if m["queue_id"]),
                    key=lambda m: m["timestamp"],
                    reverse=True,
                )
                enrich_qids = [m["queue_id"] for m in sorted_with_qid[:50]]
                all_qids = list(dict.fromkeys(
                    unknown_qids + no_from_qids + enrich_qids
                ))
            else:
                all_qids = list(dict.fromkeys(unknown_qids + no_from_qids))

            needs_trace = bool(all_qids)

            if needs_trace and all_qids:
                recent_qids = all_qids
                qid_pattern = "|".join(recent_qids)

                if include_rspamd:
                    # Запускаем оба trace параллельно: asyncssh открывает
                    # независимые channel'ы на одном TCP, время = max(postfix, rspamd),
                    # а не сумма. return_exceptions=True — чтобы падение одного
                    # не убивало второй (graceful degradation).
                    postfix_task = _query_qids_batched(
                        ssh,
                        units=["postfix@-.service"],
                        time_args=trace_time_args,
                        qids=all_qids,
                        batch_size=12,
                        identifiers=POSTFIX_IDENTIFIERS,
                    )
                    # rspamd-trace ограничен RSPAMD_TIMEOUT_S снаружи. Внутри
                    # query() есть свой SSH_QUERY_TIMEOUT=20 на conn.run; если
                    # он сработает первым — вернётся RuntimeError("timeout"),
                    # если внешний — asyncio.TimeoutError. Оба считаем таймаутом.
                    rspamd_task = asyncio.wait_for(
                        ssh.query(
                            units=["rspamd.service"],
                            time_args=trace_time_args,
                            # Простой литерал — PCRE справляется за один проход.
                            # Лишние строки (чужие qid) отфильтрует _parse_full_logs
                            # по known_qids. Alternation вида ".*(qid1|qid2|...)"
                            # здесь был бы в разы медленнее.
                            grep_pattern="rspamd_task_write_log",
                        ),
                        timeout=RSPAMD_TIMEOUT_S,
                    )
                    postfix_trace, rspamd_trace = await asyncio.gather(
                        postfix_task, rspamd_task, return_exceptions=True,
                    )

                    trace_lines: list[str] = []
                    if isinstance(postfix_trace, Exception):
                        logger.warning("postfix trace failed: %s", postfix_trace)
                    else:
                        trace_lines.extend(postfix_trace)

                    if isinstance(rspamd_trace, Exception):
                        if isinstance(rspamd_trace, (asyncio.TimeoutError, TimeoutError)):
                            rspamd_status = "timeout"
                        else:
                            rspamd_status = "error"
                        logger.warning(
                            "rspamd trace failed (%s): %s",
                            rspamd_status, rspamd_trace,
                        )
                    else:
                        rspamd_status = "included"
                        trace_lines.extend(rspamd_trace)
                else:
                    trace_lines = await _query_qids_batched(
                        ssh,
                        units=["postfix@-.service"],
                        time_args=trace_time_args,
                        qids=all_qids,
                        batch_size=12,
                        identifiers=POSTFIX_IDENTIFIERS,
                    )

                if trace_lines:
                    unique_lines = list(dict.fromkeys(
                        list(initial_lines) + list(trace_lines)
                    ))
                    parsed = _parse_full_logs(
                        unique_lines, sender, recipient, known_qids=qids,
                    )
                else:
                    # Оба trace пусты/упали — fallback на первый запрос.
                    parsed = parsed_initial
            else:
                # needs_trace=False — все статусы уже известны.
                parsed = parsed_initial
                if include_rspamd and not qids:
                    # include_rspamd=True, но qid'ов нет (только instant_rejects
                    # без queue ID) — rspamd-trace физически нечего искать.
                    rspamd_status = "skipped_no_qids"

        _localize_timestamps(parsed, QUERY_TIMEZONE)

        delivered_count = sum(1 for m in parsed if m["status"] == "delivered")
        rejected_count = sum(
            1 for m in parsed if "rejected" in m["status"]
        )
        # Поля для ответа на «когда последнее». messages уже отсортированы
        # по timestamp desc — берём первый по нужному критерию.
        latest_any = parsed[0] if parsed else None
        latest_delivered = next(
            (m for m in parsed if m["status"] == "delivered"), None,
        )
        latest_rejected = next(
            (m for m in parsed if "rejected" in m["status"]), None,
        )

        # scan_truncated = True только если journalctl реально упёрся в -n N:
        # вернулось ровно N строк. Если вернулось меньше — это все матчи
        # в журнале, обрезки не было.
        scan_truncated = (
            initial_max_lines is not None
            and len(initial_lines) >= initial_max_lines
        )

        # ★ COUNT_ONLY RETURN: только счётчики + latest_*, без messages[].
        #   ~200–500 токенов вместо 12K+ для full flow на busy mailbox
        #   (у нас был реальный кейс: 1499 писем × rspamd symbols = 12289
        #   токенов, что ровно на 1 токен больше лимита vLLM 12288).
        #   latest_* отдаём компактными (top-5 symbols, без null-полей).
        if count_only:
            return json.dumps({
                "summary": _build_summary(
                    total=len(parsed),
                    delivered=delivered_count,
                    rejected=rejected_count,
                ),
                "total_found": len(parsed),
                "delivered_count": delivered_count,
                "rejected_count": rejected_count,
                "period": period,
                "timezone": QUERY_TIMEZONE or "UTC",
                "rspamd_status": rspamd_status,
                "scan_truncated": scan_truncated,
                "latest_any": _compact_message(latest_any),
                "latest_delivered": _compact_message(latest_delivered),
                "latest_rejected": _compact_message(latest_rejected),
                "count_only": True,
                # "note": (
                #     "Передай пользователю счётчики. Если он попросит список — "
                #     "вызови check_mail_logs ещё раз с count_only=false."
                # ),
            }, ensure_ascii=False)

        # ★ COMPACTION + TOKEN BUDGET для messages[]:
        #   1. _compact_message убирает null/пустые поля и режет
        #      rspamd_symbols до top-5 по |weight| (полное число —
        #      в rspamd_symbols_total). Экономит 60–80% токенов на письмо.
        #   2. Затем режем по количеству писем, чтобы уложиться в
        #      MAX_MESSAGES_TOKENS. Считаем именно токены, а не байты/
        #      символы: кириллица в UTF-8 занимает 2 байта на символ,
        #      но ~2.5 символа на токен; байты не коррелируют с лимитом
        #      vLLM (12288 токенов).
        #   3. Минимум одно сообщение отдаём всегда — даже если оно
        #      превышает бюджет (лучше один большой, чем пустой ответ).
        compact_messages: list[dict] = []
        used_tokens = 0
        for m in parsed[:limit]:
            cm = _compact_message(m)
            if cm is None:
                continue
            cm_tokens = count_tokens(json.dumps(cm, ensure_ascii=False))
            if compact_messages and used_tokens + cm_tokens > MAX_MESSAGES_TOKENS:
                break
            compact_messages.append(cm)
            used_tokens += cm_tokens

        # dedup по адресам
        senders_top, senders_total = _collect_unique_addresses(
            compact_messages, "from",
        )
        recipients_top, recipients_total = _collect_unique_addresses(
            compact_messages, "to",
        )

        # latest_* не входят в бюджет messages[], но тоже занимают место
        # в промпте (~100–300 токенов каждый после _compact_message).
        # Логируем реальный расклад — пригодится при отладке «почему 400».
        latest_tokens = sum(
            count_tokens(json.dumps(_compact_message(x), ensure_ascii=False))
            for x in (latest_any, latest_delivered, latest_rejected)
            if x is not None
        )
        logger.debug(
            "check_mail_logs: messages=%d (used_tokens=%d), latest_tokens=%d, "
            "unique_senders=%d/%d, total_payload≈%d tokens",
            len(compact_messages), used_tokens, latest_tokens,
            len(senders_top), senders_total,
            used_tokens + latest_tokens,
        )

        # ── ★ SIDE-CHANNEL TABLE для больших списков ──────────────
        # Как в gateway_status: >5 писем → таблица прикрепляется к UI
        # напрямую через side-channel, а модели отдаём только счётчики.
        #
        # Почему это решает галлюцинации: модель больше не видит 14 разных
        # строк с timestamp/from/status — нечему перепутать. Она читает
        # summary ("Всего 14: 9 доставлено, 5 отклонено") + latest_any —
        # этого достаточно для ответа. Детали пользователь видит в таблице.
        #
        # Что убираем из payload: messages[] (главный источник ошибок и
        # токенов). Что оставляем: latest_* (для «когда последнее»),
        # unique_senders/recipients (для «от кого / кому»), by_status
        # через summary — всё, что нужно для короткого точного ответа.
        if len(compact_messages) > MAIL_ATTACH_TABLE_THRESHOLD:
            # Собираем Markdown-таблицу.
            # Колонки фиксированные + опционально rspamd, если включён.
            use_rspamd_cols = include_rspamd and any(
                m.get("rspamd_score") is not None for m in compact_messages
            )
            headers = ["Время", "От", "Кому", "Статус", "Причина"]
            if use_rspamd_cols:
                headers += ["Score", "Символы"]

            rows = []
            for m in compact_messages:
                row = [
                    m.get("timestamp", ""),
                    m.get("from", "") or "—",
                    ", ".join(m.get("to") or []) or "—",
                    m.get("status", ""),
                    (m.get("reason") or "—")[:100],
                ]
                if use_rspamd_cols:
                    symbols = m.get("rspamd_symbols") or []
                    row.append(str(m.get("rspamd_score", "—")))
                    row.append(", ".join(symbols) if symbols else "—")
                rows.append(row)

            attach_table(
                title=f"Письма ({period}, {QUERY_TIMEZONE or 'UTC'})",
                headers=headers,
                rows=rows,
                max_rows=100,
            )

            logger.info(
                "check_mail_logs: attach_table triggered "
                "(%d messages > threshold=%d), messages[] removed from LLM payload",
                len(compact_messages), MAIL_ATTACH_TABLE_THRESHOLD,
            )

            # Возвращаем payload без messages[].
            # Модель получает: summary + счётчики + latest_* + unique_*.
            # Этого достаточно для ответа на любой «покажи/сколько/когда/
            # от кого», кроме «объясни конкретное письмо X».
            return json.dumps({
                "summary": _build_summary(
                    total=len(parsed),
                    delivered=delivered_count,
                    rejected=rejected_count,
                ),
                "total_found": len(parsed),
                "delivered_count": delivered_count,
                "rejected_count": rejected_count,
                "returned": len(compact_messages),
                "truncated": len(parsed) > len(compact_messages),
                "scan_truncated": scan_truncated,
                "period": period,
                "timezone": QUERY_TIMEZONE or "UTC",
                "rspamd_status": rspamd_status,
                "unique_senders": senders_top,
                "unique_senders_total": senders_total,
                "unique_recipients": recipients_top,
                "unique_recipients_total": recipients_total,
                "latest_any": _compact_message(latest_any),
                "latest_delivered": _compact_message(latest_delivered),
                "latest_rejected": _compact_message(latest_rejected),
                "display_mode": "attached_automatically",
                "instruction_for_assistant": (
                    f"Найдено {len(compact_messages)} писем. Полная таблица со "
                    "списком БУДЕТ АВТОМАТИЧЕСКИ ПРИКРЕПЛЕНА системой сразу "
                    "после твоего сообщения. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО перечислять "
                    "отдельные письма в тексте — сообщи только сводку "
                    "(сколько доставлено/отклонено, от кого последнее) и напиши, "
                    "что полный список представлен в таблице ниже.\n\n"
                    "Если пользователю нужны детали конкретного письма или "
                    "объяснение причины отклонения — уточни у него и вызови "
                    "check_mail_logs повторно с limit=1..3 и count_only=false."
                ),
            }, ensure_ascii=False)

        return json.dumps({
            "summary": _build_summary(
                total=len(parsed),
                delivered=delivered_count,
                rejected=rejected_count,
            ),
            "total_found": len(parsed),
            "delivered_count": delivered_count,
            "rejected_count": rejected_count,
            "returned": len(compact_messages),
            "truncated": len(parsed) > len(compact_messages),
            # scan_truncated: true означает, что total_found — это «сколько
            # нашли за N матчей», а не «сколько всего в журнале». Например,
            # при total_found=465 в журнале может быть 926+ писем.
            "scan_truncated": scan_truncated,
            "period": period,
            "timezone": QUERY_TIMEZONE or "UTC",
            # rspamd_status отражает фактическое состояние rspamd-данных,
            # а не просто эхо параметра include_rspamd. Если "timeout" или
            # "error" — score/symbols в messages будут None/[].
            "rspamd_status": rspamd_status,
            # Сводка по адресам — для «от кого?» / «кому?» без дублей
            "unique_senders": senders_top,
            "unique_senders_total": senders_total,
            "unique_recipients": recipients_top,
            "unique_recipients_total": recipients_total,
            # Прямые ответы на «когда последнее». Модели не надо самой
            # искать первый delivered в messages. Компактные — без
            # rspamd_symbols (top-5) и без null-полей.
            "latest_any": _compact_message(latest_any),
            "latest_delivered": _compact_message(latest_delivered),
            "latest_rejected": _compact_message(latest_rejected),
            "messages": compact_messages,
        }, ensure_ascii=False)

    except Exception as e:
        logger.exception("check_mail_logs failed")
        return json.dumps({"error": str(e)}, ensure_ascii=False)