# core/tools/exchange_mailbox_tool/tool.py

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

EXCHANGE_MAILBOX_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_exchange_mailbox_info",
            "description": (
                "Получает настройки, права доступа, квоты и статус почтового ящика из Exchange 2019 и Active Directory. "
                "Используй для вопросов про: переадресацию (forwarding), размер ящика и свободное место, "
                "сетевой архив (Online Archive), права доступа других пользователей к ящику, "
                "доступ через OWA и мобильные устройства (ActiveSync), статус учетной записи (активна/отключена) "
                "и дату последней смены пароля."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "Email пользователя (например, user@center-inform.ru). Оставь пустым, если запрос по всем пользователям сразу (например, «у кого включена переадресация» или «кому разрешен OWA»)."
                    },
                    "query_type": {
                        "type": "string",
                        "enum": [
                            "general_and_ad",     # статус УЗ (активна/заблокирована), дата смены пароля (вопросы 5, 7)
                            "forwarding",         # переадресация на другой адрес (вопрос 1)
                            "size_and_archive",   # размер ящика, лимиты, подключен ли архив и место в нем (вопросы 8, 9, 10)
                            "permissions",        # кто имеет доступ FullAccess / SendAs к ящику (вопрос 11)
                            "client_access"       # разрешен ли OWA и ActiveSync / мобильные клиенты (вопросы 6, 12)
                        ],
                        "description": "Тип запрашиваемой информации по ящику."
                    }
                },
                "required": ["query_type"]
            }
        }
    }
]

async def get_exchange_mailbox_info(
    query_type: str
) -> str:
    pass