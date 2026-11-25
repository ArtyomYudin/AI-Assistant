# core/tools/mail_audit_tool/patterns.py

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

"""Регулярки для парсинга journalctl-логов Postfix и Rspamd."""
import re

# Postfix: "postfix/qmgr[227056]: D4A8371: from=<...>"
RE_QID = re.compile(r'postfix/\w+\[\d+\]:\s+([A-Za-z0-9]{5,}):')
RE_FROM = re.compile(r'from=<([^>]*)>')
RE_TO = re.compile(r'to=<([^>]*)>')

# Rspamd: "qid: <B49DB69>" / "queue-id: <B49DB69>"
RE_RSPAMD_QID = re.compile(r'(?:qid|queue-id):\s*<?([A-Za-z0-9]{5,})>?')

# Score в двух форматах: "[-4.45/12.00]" (новый) и "score: -4.45" (старый).
RE_RSPAMD_SCORE = re.compile(
    r'(?:\[\s*(-?\d+(?:\.\d+)?)\s*/\s*\d+(?:\.\d+)?\s*\]'
    r'|\bscore:\s*(-?\d+(?:\.\d+)?))'
)

# Один символ в блоке: NAME(weight){details?}
RE_RSPAMD_SYMBOL_ITEM = re.compile(
    r'([A-Z][A-Z0-9_]*)(?:\(([^)]*)\))?(?:\{[^}]*\})?'
)

# Блок символов: [SYM1(1.00){...},SYM2(-2.00){},...]
RE_RSPAMD_SYMBOLS_BLOCK = re.compile(r'\[([A-Z][^\]]*)\]')

# Action: "(default: F (no action)"
RE_RSPAMD_ACTION = re.compile(r'\(default:\s*([A-Z])\s*\(([^)]+)\)')