import logging, os, pathlib
from typing import List, Optional
import PyPDF2
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def load_documents_from_directory(directory_path: str, file_extensions: Optional[List[str]] = None) -> List[Document]:
    directory = pathlib.Path(directory_path)
    if file_extensions is None:
        file_extensions = ["*.pdf", "*.txt", "*.md"]
    if not directory.exists():
        logger.warning("Директория %s не существует", directory)
        return []
    docs = []
    for ext in file_extensions:
        for path in directory.rglob(f"**/{ext}"):   # рекурсивный обход
            try:
                if path.suffix.lower() == ".pdf":
                    with open(path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        text = "\n".join((p.extract_text() or "") for p in reader.pages).strip()
                    if text:
                        docs.append(Document(page_content=text, metadata={"source": str(path)}))
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        docs.append(Document(page_content=content, metadata={"source": str(path)}))
            except Exception as e:
                logger.exception("Ошибка при загрузке %s: %s", path, e)
    logger.info("Загружено документов: %d", len(docs))
    return docs
