# core/tools/mail_audit_tool/parser.py

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


"""Парсинг сырых строк journalctl в структурированный список писем."""
from datetime import datetime
from typing import Optional

from .patterns import (
    RE_QID, RE_FROM, RE_TO,
    RE_RSPAMD_QID, RE_RSPAMD_SCORE, RE_RSPAMD_SYMBOL_ITEM,
    RE_RSPAMD_SYMBOLS_BLOCK, RE_RSPAMD_ACTION,
)


def _extract_qid(line: str, known_qids: Optional[set[str]] = None) -> Optional[str]:
    """Извлекает queue ID из postfix или rspamd строки.

    known_qids — qid'ы, уже собранные из первого postfix-запроса.
    Для rspamd-строк сначала ищем вхождение известного qid подстрокой:
    это надёжнее regex, потому что формат rspamd-логов меняется между
    версиями ("qid: <X>", "queue-id: <X>", "#X(mail)" и т.п.).
    """
    m = RE_QID.search(line)
    if m:
        return m.group(1)
    if "rspamd" in line.lower():
        if known_qids:
            for qid in known_qids:
                if qid in line:
                    return qid
        m = RE_RSPAMD_QID.search(line)
        if m:
            return m.group(1)
    return None


def _parse_full_logs(
    lines: list[str],
    sender_filter: Optional[str],
    recipient_filter: Optional[str],
    known_qids: Optional[set[str]] = None,
) -> list[dict]:
    """Парсит сырые строки journalctl в структурированный список писем.

    known_qids нужен для матчинга rspamd-строк, где qid может быть в
    произвольном формате, но всегда присутствует как подстрока.
    """
    messages: dict[str, dict] = {}
    instant_rejects: list[dict] = []

    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        timestamp = parts[0]

        # --- Отклонения без Queue ID (NOQUEUE: reject / milter-reject) ---
        # Postfix пишет их до постановки в очередь, поэтому qid отсутствует.
        # Это отдельная ветка, иначе такие письма теряются полностью.
        if not RE_QID.search(line) and ("reject:" in line or "status=reject" in line):
            m_from = RE_FROM.search(line)
            m_to = RE_TO.search(line)
            f_addr = m_from.group(1) if m_from else ""
            t_addr = m_to.group(1) if m_to else ""

            if sender_filter and sender_filter.lower() not in f_addr.lower():
                continue
            if recipient_filter and recipient_filter.lower() not in t_addr.lower():
                continue

            if "milter-reject:" in line:
                reason = line.split("milter-reject:", 1)[1].strip()
                status = "rejected_by_filter"
            elif "status=reject" in line:
                reason = line.split("status=reject", 1)[1].lstrip(":").strip()
                status = "rejected_at_connection"
            else:
                reason = line.split("reject:", 1)[1].strip()
                status = "rejected_at_connection"

            instant_rejects.append({
                "timestamp": timestamp,
                "queue_id": None,
                "from": f_addr,
                "to": [t_addr] if t_addr else [],
                "status": status,
                "reason": reason,
                "rspamd_score": None,
                "rspamd_symbols": [],
            })
            continue

        # --- События с Queue ID (postfix или rspamd) ---
        qid = _extract_qid(line, known_qids)
        if not qid:
            continue

        if qid not in messages:
            messages[qid] = {
                "timestamp": timestamp,
                "queue_id": qid,
                "from": "",
                "to": set(),
                "status": "processing",
                "reason": None,
                "rspamd_score": None,
                "rspamd_symbols": [],
            }

        msg = messages[qid]

        m_from = RE_FROM.search(line)
        if m_from and not msg["from"]:
            msg["from"] = m_from.group(1)

        m_to = RE_TO.search(line)
        if m_to:
            msg["to"].add(m_to.group(1))

        # --- Статусы Postfix ---
        # Порядок важен: milter-reject/status=bounced/deferred/reject идут
        # после status=sent, но взаимно исключают друг друга по строке.
        if "status=sent" in line:
            msg["status"] = "delivered"
        elif "milter-reject:" in line:
            msg["status"] = "rejected_by_filter"
            msg["reason"] = line.split("milter-reject:", 1)[1].strip()
        elif "status=bounced" in line:
            msg["status"] = "bounced"
            msg["reason"] = line.split("status=bounced", 1)[1].strip()
        elif "status=deferred" in line:
            msg["status"] = "deferred"
            msg["reason"] = line.split("status=deferred", 1)[1].strip()
        elif "status=reject" in line:
            msg["status"] = "rejected_at_connection"
            msg["reason"] = line.split("status=reject", 1)[1].lstrip(":").strip()

        # --- Вердикт Rspamd ---
        if "rspamd" in line and qid in line:
            score_m = RE_RSPAMD_SCORE.search(line)
            if score_m:
                # группа 1 — формат [x/y], группа 2 — формат "score: x"
                msg["rspamd_score"] = score_m.group(1) or score_m.group(2)

            # Ищем блок [SYM(weight){details},...]. Первый найденный блок —
            # это и есть symbols (score-блок "[-4.45/12.00]" не подходит:
            # начинается с цифры/минуса, а регулярка ждёт заглавную букву).
            for block_m in RE_RSPAMD_SYMBOLS_BLOCK.finditer(line):
                items = RE_RSPAMD_SYMBOL_ITEM.findall(block_m.group(1))
                if items:
                    msg["rspamd_symbols"] = [
                        f"{name}({weight})" if weight else name
                        for name, weight in items
                    ]
                    break

            action_m = RE_RSPAMD_ACTION.search(line)
            if action_m:
                msg["rspamd_action"] = action_m.group(2)

    # Финальная фильтрация + сортировка
    results: list[dict] = []
    for msg in messages.values():
        msg["to"] = sorted(msg["to"])
        if sender_filter and sender_filter.lower() not in msg["from"].lower():
            continue
        if recipient_filter and not any(
            recipient_filter.lower() in r.lower() for r in msg["to"]
        ):
            continue
        results.append(msg)

    results.extend(instant_rejects)
    results.sort(key=lambda x: x["timestamp"], reverse=True)
    return results


def _localize_timestamps(messages: list[dict], tz_name: Optional[str]) -> None:
    """Переводит timestamp из UTC (journalctl --utc) в локальный пояс пользователя.

    Вызывается ПОСЛЕ _parse_full_logs: сортировка там идёт по ISO-строкам UTC,
    а после локализации формат меняется на "%Y-%m-%d %H:%M:%S %Z" —
    лексикографический sort по нему сломался бы.
    """
    if not tz_name or not messages:
        return
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
    except Exception:
        return
    for m in messages:
        ts = m.get("timestamp")
        if not ts:
            continue
        try:
            dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S%z")
        except ValueError:
            continue
        m["timestamp"] = dt.astimezone(tz).strftime("%Y-%m-%d %H:%M:%S %Z")