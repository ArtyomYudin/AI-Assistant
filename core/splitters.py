import logging
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from core.utils import clean_md_content, hash_text, remove_bm25_comments, extract_global_bm25_text

logger = logging.getLogger(__name__)

HEADERS_TO_SPLIT_ON: List[Tuple[str, str]] = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
    ("####", "Header4"),
]

class SplitterManager:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.recursive = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ".", ",", ""], length_function=len
        )
        self.header = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)

    def split_by_headers(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
        chunks: List[Document] = []
        for doc in documents:
            raw_content = doc.page_content
            # Извлекаем ГЛОБАЛЬНЫЙ bm25_text для всего документа
            global_bm25 = extract_global_bm25_text(raw_content)
            # Удаляем комментарий из контента
            content_wo_bm25 = remove_bm25_comments(raw_content)
            logger.info("Извлечён global_bm25: %r", global_bm25)
            content = clean_md_content(content_wo_bm25)
            if not content:
                continue
            try:
                header_docs = self.header.split_text(content)
            except Exception as e:
                logger.warning("Ошибка при разбиении по заголовкам: %s", e)
                header_docs = [Document(page_content=content, metadata={})]
            for h_doc in header_docs:
                # Собираем заголовки в одну строку
                headers = [h_doc.metadata.get(k) for _, k in HEADERS_TO_SPLIT_ON if k in h_doc.metadata]
                title = " > ".join(filter(None, headers)) if headers else doc.metadata.get("title", "")

                # Формируем bm25_text для чанка:
                #    - если есть глобальный — используем его + заголовок секции
                #    - иначе — fallback на заголовок
                if global_bm25:
                    # bm25_text = f"{title}. {global_bm25}"
                    bm25_text = global_bm25
                else:
                    bm25_text = title

                # Дробим дальше, если нужно
                if len(h_doc.page_content) > self.recursive._chunk_size:  # noqa: SLF001
                    try:
                        subs = self.recursive.split_documents([h_doc])
                        for s in subs:
                            s.metadata["title"] = title
                            s.metadata["bm25_text"] = bm25_text
                            logger.info("Рекурсивно установлен bm25_text для чанка с заголовком %r: %r", title, bm25_text)
                        chunks.extend(subs)
                    except Exception as e:
                        logger.warning("Ошибка при рекурсивном разбиении: %s", e)
                        h_doc.metadata["title"] = title
                        h_doc.metadata["bm25_text"] = bm25_text
                        chunks.append(h_doc)
                else:
                    h_doc.metadata["title"] = title
                    logger.info("Для чанка с заголовком %r установлен bm25_text: %r", title, bm25_text)
                    h_doc.metadata["bm25_text"] = bm25_text
                    chunks.append(h_doc)
        for c in chunks:
            c.metadata["hash"] = hash_text(c.page_content)
        return chunks
