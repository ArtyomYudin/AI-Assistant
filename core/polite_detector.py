# core/polite_detector.py

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

import re

class PoliteDetector:
    """
    Локальный детектор вежливых/служебных сообщений.
    Не вызывает LLM, не требует контекста.
    Работает на регэкспах.
    """

    POLITE_PATTERNS = [
        r"^спасибо[!.,\s]*$",
        r"^спасибо большое[!.,\s]*$",
        r"^большое спасибо[!.,\s]*$",
        r"^благодарю[!.,\s]*$",
        r"^благодарю за помощь[!.,\s]*$",

        r"^(ок|окей|ага|понял|понятно|ясно|хорошо|ладно)[!.,\s]*$",

        r"^привет[!.,\s]*$",
        r"^здравствуйте[!.,\s]*$",
        r"^здравствуй[!.,\s]*$",
        r"^добрый день[!.,\s]*$",
        r"^добрый вечер[!.,\s]*$",
        r"^доброе утро[!.,\s]*$",

        r"^пока[!.,\s]*$",
        r"^до свидания[!.,\s]*$",
        r"^до встречи[!.,\s]*$",
    ]

    @classmethod
    def is_polite(cls, text: str) -> bool:
        t = text.strip().lower()
        return any(re.match(p, t) for p in cls.POLITE_PATTERNS)

    @classmethod
    def reply(cls, text: str) -> str:
        t = text.strip().lower()

        if "спасиб" in t or "благодар" in t:
            return "Пожалуйста! Если будут ещё вопросы — обращайтесь."

        if t.startswith(("привет", "здравствуй")):
            return "Здравствуйте! Чем могу помочь?"

        if t.startswith("добрый"):
            return "Здравствуйте! Чем могу помочь?"

        if t in ("ок", "окей", "ага", "понял", "ясно", "хорошо", "ладно"):
            return "Хорошо. Если появятся вопросы — спрашивайте."

        if t in ("пока", "до свидания", "до встречи"):
            return "Всего доброго!"

        return "Я здесь, чтобы помочь. Задайте вопрос по системам или регламентам ИТО."