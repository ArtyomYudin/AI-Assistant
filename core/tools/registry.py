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
from core.tools.kb_tool import KB_TOOLS, search_knowledge_base


# Все схемы, которые передаём в LLM
ALL_TOOLS = [
    *WEATHER_TOOLS,
    # *KB_TOOLS,
    # сюда добавляй другие *_TOOLS по мере роста
    # *VLAN_TOOLS,
    # *IP_TOOLS,
]


# Имена → вызываемые функции
TOOL_REGISTRY = {
    "get_weather": get_weather,
    # "search_knowledge_base": search_knowledge_base,
    # "lookup_vlan": lookup_vlan,
}