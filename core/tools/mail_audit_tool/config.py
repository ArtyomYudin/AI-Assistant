# core/tools/mail_audit_tool/config.py

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


"""Конфигурация mail-audit инструмента."""
import os

SSH_HOST = os.getenv("MAIL_AUDIT_SSH_HOST", "ingrid.center-inform.ru")
SSH_USER = os.getenv("MAIL_AUDIT_SSH_USER", "mail_auditor")
SSH_KEY_PATH = os.getenv(
    "MAIL_AUDIT_SSH_KEY", "config/id_ed25519_mail_auditor"
)

SSH_CONNECT_TIMEOUT = int(os.getenv("MAIL_AUDIT_SSH_CONNECT_TIMEOUT", "10"))
SSH_QUERY_TIMEOUT = int(os.getenv("MAIL_AUDIT_SSH_QUERY_TIMEOUT", "20"))
SSH_KEEPALIVE = int(os.getenv("MAIL_AUDIT_SSH_KEEPALIVE", "30"))
SSH_KEEPALIVE_COUNT_MAX = int(
    os.getenv("MAIL_AUDIT_SSH_KEEPALIVE_COUNT_MAX", "3")
)

RSPAMD_TIMEOUT_S = float(os.getenv("MAIL_AUDIT_RSPAMD_TIMEOUT_S", "6.0"))

# None → относительные даты journalctl; строка → явные границы в этом поясе.
QUERY_TIMEZONE = os.getenv("MAIL_AUDIT_QUERY_TZ", "Europe/Moscow") or None

POSTFIX_IDENTIFIERS = [
    "postfix/qmgr", "postfix/cleanup", "postfix/smtp",
    "postfix/bounce", "postfix/error", "postfix/local",
    "postfix/virtual", "postfix/lmtp", "postfix/pipe",
]

KNOWN_FINAL_STATUSES = {
    "delivered", "bounced", "deferred",
    "rejected_at_connection", "rejected_by_filter",
}

VALID_PERIODS = {"today", "yesterday", "7days", "all"}
PERIOD_ALIASES = {
    "сегодня": "today",
    "вчера": "yesterday",
    "7дней": "7days",
    "7 дней": "7days",
    "неделя": "7days",
    "всё": "all",
    "все": "all",
    "all time": "all",
}

# Бюджет messages[] в токенах. vLLM: 12288 total.
# Расклад: schema ~250 + system ~250 + history до 3500 + completion до 2048
#          + buffer. ToolMessage не должен съедать больше ~2000, иначе
#          даже без RAG и истории останется впритык.
# 2000 покрывает ~15–20 писем с rspamd symbols или ~30–40 без rspamd.
# Если итог превышает — режем по количеству писем, а не по символам
# в середине JSON (иначе получим невалидный JSON и модель сломается).
MAX_MESSAGES_TOKENS = 2000

# Максимум уникальных адресов в unique_senders/unique_recipients.
# 15 × ~50 символов = ~750 chars = ~200 токенов — приемлемо.
# Если отправителей больше — модель увидит "unique_senders_total: N"
# и поймёт, что список неполный.
MAX_UNIQUE_ADDRESSES = 15

# Порог: до 5 писем — модель пересказывает сама (коротко, точно),
# больше 5 — таблица через side-channel, модель только summary.
MAIL_ATTACH_TABLE_THRESHOLD = int(os.getenv("MAIL_ATTACH_TABLE_THRESHOLD", "5"))