# core/tools/registry.py
from core.tools.weather_tool import WEATHER_TOOLS, get_weather
from core.tools.kb_tool import KB_TOOLS, search_knowledge_base


# Все схемы, которые передаём в LLM
ALL_TOOLS = [
    *WEATHER_TOOLS,
    *KB_TOOLS,
    # сюда добавляй другие *_TOOLS по мере роста
    # *VLAN_TOOLS,
    # *IP_TOOLS,
]


# Имена → вызываемые функции
TOOL_REGISTRY = {
    "get_weather": get_weather,
    "search_knowledge_base": search_knowledge_base,
    # "lookup_vlan": lookup_vlan,
}