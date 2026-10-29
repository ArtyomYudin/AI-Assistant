
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

import asyncio, logging
from config.rag_config import RAGConfig
from core.rag_core import RAGCore
from core.collection_manager import CollectionManager


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s: %(message)s")
logger = logging.getLogger(__name__)

async def demo_question(core: RAGCore, question: str, session_id: str = "demo") -> None:
    if core.qa_chain_with_history is None:
        core.create_qa_generator()
    print("\nQ:", question)
    print("A:", end=" ", flush=True)
    async for token in core.qa_chain_with_history(question, user_id=session_id):
        print(token, end="", flush=True)
    print("\n")

async def build_and_index(core: RAGCore, data_dir=None) -> None:
    docs = core.load_documents(directory=data_dir or core.config.DATA_DIR)
    if not docs:
        logger.warning("Документы не найдены. Добавьте файлы в scraped_data/")
        return
    await core.setup_vectorstore(docs)

async def main():
    cfg = RAGConfig()
    core = RAGCore(cfg)
    # cm = CollectionManager(core)

    # Строим все коллекции по папкам
    await core.collection_manager.build_all_collections()

    # # Строим маршрутизатор коллекций
    # await core.collection_manager.build_router()
    #
    # # await build_and_index(core)
    # core.create_retriever(k=cfg.K, fetch_k=cfg.FETCH_K)
    # await demo_question(core, "список камер видеонаблюдения?")
    # # resp = await core.collection_manager.routed_search("как получить доступ к камерам?")
    # # print(resp)

    await core.close()

if __name__ == "__main__":
    asyncio.run(main())