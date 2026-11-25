# core/tools/gateway_status_tool/tool.py
import ipaddress
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

import json
import logging
from collections import Counter
from typing import Any

from core.tools.context import attach_table
from core.tools.mail_audit_tool.client import _get_client

logger = logging.getLogger(__name__)


INLINE_NETWORKS_THRESHOLD = 15

GATEWAY_STATUS_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "check_gateway_status",
            "description": (
                "Проверяет текущее состояние, очереди и настройки безопасности почтового "
                "шлюза (Postfix / Ingrid). Используй этот инструмент для вопросов о текущем "
                "состоянии сервера, а не об истории отдельных писем:\n"
                "• размер текущей очереди писем на отправку/обработку на шлюзе;\n"
                "• список доверенных IP-адресов и подсетей (mynetworks / relay), которым "
                "разрешена отправка писем без авторизации;\n"
                "• статус работы служб шлюза (Postfix, Rspamd, Redis)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": [
                            "queue_size",
                            "trusted_networks",
                            "services_health",
                        ],
                        "description": (
                            "Тип запрашиваемой информации со шлюза:\n"
                            "• 'queue_size' — количество писем в очереди на шлюзе прямо сейчас "
                            "(active, deferred, hold, bounce). Выбирай для вопросов «какой размер "
                            "очереди», «сколько писем висит на шлюзе», «есть ли затор в очереди».\n"
                            "• 'trusted_networks' — список IP-адресов и подсетей (mynetworks / "
                            "relay_hosts), которым разрешена отправка писем через шлюз без "
                            "авторизации. Выбирай для вопросов «каким IP разрешен релей», «кому "
                            "разрешена отправка без авторизации».\n"
                            "• 'services_health' — проверка, запущены ли службы Postfix и Rspamd."
                        ),
                    },
                    "include_queue_details": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Актуально только при metric='queue_size'. Если true — возвращает "
                            "разбивку очереди по доменам-получателям и причины задержки (deferred). "
                            "Оставляй false, если пользователь спросил просто «какой размер "
                            "очереди» или «сколько писем в очереди»."
                        ),
                    },
                    "network_filter": {
                        "type": "string",
                        "default": "",
                        "description": (
                            "Актуально только при metric='trusted_networks'. Необязательный "
                            "фильтр по IP-адресу, префиксу или подсети (например, '172.30.7', "
                            "'172.30.7.0/24', '10.115.1.121'). Заполняй ТОЛЬКО если пользователь "
                            "спросил про конкретный IP или подсеть. В остальных случаях оставляй пустым."
                        ),
                    },
                },
                "required": ["metric"],
            },
        },
    }
]


def _parse_queue(raw_stdout: str, include_details: bool) -> dict[str, Any]:
    lines = [line for line in raw_stdout.splitlines() if line.strip()]
    if not lines:
        return {
            "total_in_queue": 0,
            "breakdown": {"active": 0, "deferred": 0, "hold": 0, "incoming": 0},
            "status": "Очередь на шлюзе пуста.",
        }

    queue_counts: Counter[str] = Counter()
    recipient_domains: Counter[str] = Counter()
    delay_reasons: Counter[str] = Counter()

    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        q_name = item.get("queue_name", "unknown")
        queue_counts[q_name] += 1

        if include_details:
            for rcpt in item.get("recipients", []):
                addr = rcpt.get("address", "")
                if "@" in addr:
                    domain = addr.split("@", 1)[1].lower()
                    recipient_domains[domain] += 1
                reason = rcpt.get("delay_reason")
                if reason:
                    delay_reasons[reason[:140]] += 1

    result: dict[str, Any] = {
        "total_in_queue": sum(queue_counts.values()),
        "breakdown": dict(queue_counts),
    }

    if include_details:
        result["top_recipient_domains"] = dict(recipient_domains.most_common(5))
        if delay_reasons:
            result["top_delay_reasons"] = dict(delay_reasons.most_common(5))

    return result


def _matches_filter(network_str: str, filter_str: str) -> bool:
    flt = filter_str.strip()
    if not flt:
        return True

    # Если передан неполный IPv4-префикс без маски (например, '192.168.21' или '172.30'),
    # превращаем его в валидную подсеть ('192.168.21.0/24' или '172.30.0.0/16')
    parts = flt.rstrip(".").split(".")
    if "/" not in flt and all(p.isdigit() for p in parts):
        if len(parts) == 3:
            flt = f"{'.'.join(parts)}.0/24"
        elif len(parts) == 2:
            flt = f"{'.'.join(parts)}.0.0/16"
        elif len(parts) == 1:
            flt = f"{parts}.0.0.0/8"

    try:
        rule_net = ipaddress.ip_network(network_str, strict=False)
        query_net = ipaddress.ip_network(flt, strict=False)
        return rule_net.overlaps(query_net)
    except ValueError:
        return flt in network_str



def _parse_mynetworks(raw_stdout: str, network_filter: str = "") -> dict[str, Any]:
    entries: list[dict[str, str]] = []
    current_source = "main.cf:mynetworks"

    for raw_line in raw_stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# source:"):
            current_source = line.replace("# source:", "").strip()
            continue

        comment = ""
        if "#" in line:
            left_part, comment_part = line.split("#", 1)
            comment = comment_part.strip()
        else:
            left_part = line

        tokens = left_part.split()
        if not tokens:
            continue

        network = tokens[0]
        action = tokens[1] if len(tokens) > 1 else "PERMIT"

        if action.upper() == "REJECT":
            continue

        # Применяем фильтр, если пользователь указал IP или подсеть
        if network_filter and not _matches_filter(network, network_filter):
            continue

        entries.append(
            {
                "network": network,
                "comment": comment,
                "source": current_source,
            }
        )

    total = len(entries)

    if total == 0 and network_filter:
        return {
            "total_allowed_networks": 0,
            "filter_applied": network_filter,
            "status": f"Адресов или подсетей, попадающих под '{network_filter}', в списке разрешённых не найдено.",
        }

    # Малый список (после фильтрации или всего): отдаём модели целиком
    if total <= INLINE_NETWORKS_THRESHOLD:
        res: dict[str, Any] = {
            "total_allowed_networks": total,
            "allowed_networks": entries,
        }
        if network_filter:
            res["filter_applied"] = network_filter
        return res

    # Большой список (> 15): отправляем в side-channel таблицу,
    # а модели НЕ даём preview_first_5, чтобы она не перечисляла обрывок списка
    rows = [
        [f"`{e['network']}`", e["comment"] or "—", f"`{e['source']}`"]
        for e in entries
    ]
    title = (
        f"Доверенные IP-адреса и подсети (фильтр: {network_filter})"
        if network_filter
        else "Доверенные IP-адреса и подсети шлюза (отправка без авторизации)"
    )
    attach_table(
        title=title,
        headers=["IP / Подсеть", "Назначение / Комментарий", "Источник"],
        rows=rows,
    )

    return {
        "total_allowed_networks": total,
        "filter_applied": network_filter or None,
        "display_mode": "attached_automatically",
        "instruction_for_assistant": (
            f"Найдено {total} записей. Полная таблица со всеми адресами БУДЕТ "
            "АВТОМАТИЧЕСКИ ПРИКРЕПЛЕНА системой сразу после твоего сообщения. "
            "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО перечислять отдельные IP-адреса в тексте — "
            "сообщи только общее количество и напиши, что полный список представлен в таблице ниже."
        ),
    }


def _parse_health(raw_stdout: str) -> dict[str, Any]:
    services = {}
    for line in raw_stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        sep = "=" if "=" in line else (":" if ":" in line else None)
        if sep:
            unit, state = line.split(sep, 1)
            services[unit.strip()] = state.strip()

    return {
        "overall_healthy": bool(services) and all(
            state == "active" for state in services.values()
        ),
        "services": services,
    }



async def check_gateway_status(
    metric: str,
    include_queue_details: bool = False,
    network_filter: str = "",
) -> str:
    try:
        ssh = _get_client()
        if metric == "queue_size":
            stdout = await ssh.run_gateway_cmd("postfix postqueue -j")
            payload = _parse_queue(stdout, include_queue_details)
        elif metric == "trusted_networks":
            stdout = await ssh.run_gateway_cmd("postfix mynetworks")
            payload = _parse_mynetworks(stdout, network_filter=network_filter)
        elif metric == "services_health":
            stdout = await ssh.run_gateway_cmd("system health")
            payload = _parse_health(stdout)
        else:
            payload = {"error": f"Unknown metric: {metric}"}

    except Exception as e:
        logger.exception("check_gateway_status failed")
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    return json.dumps(payload, ensure_ascii=False)