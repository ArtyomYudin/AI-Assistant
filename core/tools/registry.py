# core/tools/registry.py

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

from core.tools.weather_tool import WEATHER_TOOLS, get_weather
from .exchange_mailbox_tool.tool import EXCHANGE_MAILBOX_TOOL, get_exchange_mailbox_info
from .gateway_status_tool.tool import GATEWAY_STATUS_TOOL, check_gateway_status
from .mail_audit_tool import check_mail_logs, MAIL_AUDIT_TOOL

# Все схемы, которые передаём в LLM
ALL_TOOLS = [
    # *WEATHER_TOOLS,
    *MAIL_AUDIT_TOOL,
    *GATEWAY_STATUS_TOOL,
    *EXCHANGE_MAILBOX_TOOL,
]


# Имена → вызываемые функции
TOOL_REGISTRY = {
    "get_weather": get_weather,
    "check_mail_logs": check_mail_logs,
    "check_gateway_status": check_gateway_status,
    "get_exchange_mailbox_info": get_exchange_mailbox_info
    # "search_knowledge_base": search_knowledge_base,
    # "lookup_vlan": lookup_vlan,
}