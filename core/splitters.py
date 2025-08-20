import logging
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from core.utils import clean_md_content, hash_text

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
            content = clean_md_content(doc.page_content)
            if not content:
                continue
            try:
                header_docs = self.header.split_text(content)
            except Exception as e:
                logger.warning("Ошибка при разбиении по заголовкам: %s", e)
                header_docs = [Document(page_content=content, metadata={})]
            for h_doc in header_docs:
                headers = [h_doc.metadata.get(k) for _, k in HEADERS_TO_SPLIT_ON if k in h_doc.metadata]
                prefix = " > ".join(filter(None, headers)) + "\n" if headers else ""
                if len(h_doc.page_content) > self.recursive._chunk_size:  # noqa: SLF001
                    try:
                        subs = self.recursive.split_documents([h_doc])
                        for s in subs:
                            s.page_content = prefix + s.page_content
                        chunks.extend(subs)
                    except Exception as e:
                        logger.warning("Ошибка при рекурсивном разбиении: %s", e)
                        h_doc.page_content = prefix + h_doc.page_content
                        chunks.append(h_doc)
                else:
                    h_doc.page_content = prefix + h_doc.page_content
                    chunks.append(h_doc)
        for c in chunks:
            c.metadata["hash"] = hash_text(c.page_content)
        return chunks
