import asyncio, logging
from config.rag_config import RAGConfig
from core.rag_core import RAGCore

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s: %(message)s")
logger = logging.getLogger(__name__)

async def demo_question(core: RAGCore, question: str, session_id: str = "demo") -> None:
    if core.qa_chain_with_history is None:
        core.create_qa_generator()
    print("\nQ:", question)
    print("A:", end=" ", flush=True)
    async for token in core.qa_chain_with_history(question, session_id=session_id):
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
    await build_and_index(core)
    core.create_retriever(k=cfg.K, fetch_k=cfg.FETCH_K)
    await demo_question(core, "Коротко опиши цели документации из источников.")
    await core.close()

if __name__ == "__main__":
    asyncio.run(main())