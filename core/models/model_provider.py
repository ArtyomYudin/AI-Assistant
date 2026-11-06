# core/models/model_provider.py

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

import logging

from langchain_core.messages import HumanMessage

from config.rag_config import RAGConfig
from core.tools.registry import ALL_TOOLS

logger = logging.getLogger(__name__)


class ModelProvider:
    """
    Ленивая инициализация embeddings / LLM / LLM+tools / tool_check_llm.
    Единая точка прогрева и закрытия клиентов.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self._embeddings = None
        self._llm = None
        self._llm_with_tools = None
        self._tool_check_llm = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            from core.models.local_embeddings import LocalEmbeddings
            self._embeddings = LocalEmbeddings(
                model=self.config.EMBEDDING_NAME,
                base_url=self.config.EMBEDDING_BASE_URL,
                timeout=60,
            )
        return self._embeddings

    @property
    def llm(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self.config.LLM_NAME,
                api_key="EMPTY",
                base_url=self.config.LLM_BASE_URL,
                max_tokens=self.config.LLM_MAX_TOKEN,
                temperature=self.config.LLM_TEMPERATURE,
                top_p=self.config.LLM_TOP_P,
                presence_penalty=self.config.LLM_PRESENCE_PENALTY,
                frequency_penalty=self.config.LLM_FREQUENCY_PENALTY,
                streaming=True,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "top_k": self.config.LLM_TOP_K,
                    "bad_words": [
                        "Если нужно — могу",
                        "Если нужно, могу",
                        "Если нужно – могу",
                        "Если нужно, я могу",
                        "Если нужно — я могу",
                        "Если потребуется — могу",
                        "Если потребуется, могу",
                        "Если необходимо — могу",
                        "Если необходимо, могу",
                        "Если понадобится — могу",
                        "Если понадобится, могу",
                        "Дайте знать, если",
                        "Дайте знать если",
                        "Обращайтесь, если",
                        "Обращайтесь если",
                        "Готов помочь",
                        "Готова помочь",
                        "Рад помочь",
                        "Рада помочь",
                        "могу экспортировать",
                        "могу прислать",
                        "могу отфильтровать",
                        "могу создать",
                        "могу сделать",
                        "могу предоставить",
                        "могу уточнить",
                        "могу показать",
                        "могу предложить",
                    ],
                },
            )
        return self._llm

    @property
    def llm_with_tools(self):
        """LLM с привязанными инструментами (ленивая инициализация)."""
        if self._llm_with_tools is None:
            self._llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
            logger.info(
                "LLM bind_tools: %d инструментов → %s",
                len(ALL_TOOLS),
                [t["function"]["name"] for t in ALL_TOOLS],
            )
        return self._llm_with_tools

    @property
    def tool_check_llm(self):
        """Лёгкий LLM только для решения «нужен ли инструмент»."""
        if self._tool_check_llm is None:
            self._tool_check_llm = self.llm_with_tools.bind(
                max_tokens=64,
                temperature=0.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
            )
        return self._tool_check_llm

    async def warmup(self) -> None:
        try:
            await self.embeddings.embed_query("warmup")
            logger.info("✅ Embeddings прогреты")
        except Exception as e:
            logger.warning("⚠️ Embeddings warmup failed: %s", e)

        try:
            _ = self.llm
            _ = self.llm_with_tools
            _ = self.tool_check_llm
            await self.llm.bind(max_tokens=1).ainvoke([HumanMessage(content=".")])
            logger.info("✅ LLM прогреты: %d инструментов", len(ALL_TOOLS))
        except Exception as e:
            logger.warning("⚠️ LLM warmup failed: %s", e)

    async def aclose(self) -> None:
        if self._embeddings is not None and hasattr(self._embeddings, "client"):
            try:
                await self._embeddings.client.aclose()
            except Exception as e:
                logger.warning("Ошибка при закрытии embeddings: %s", e)

        if self._llm is not None:
            for attr in ("_async_client", "client"):
                c = getattr(self._llm, attr, None)
                if c is not None and hasattr(c, "aclose"):
                    try:
                        await c.aclose()
                    except Exception as e:
                        logger.warning("Ошибка при закрытии LLM-клиента: %s", e)
                    break