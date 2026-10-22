# core/splitters.py

import logging
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from core.utils import clean_md_content, hash_text, remove_bm25_comments, extract_global_bm25_text
import logging
from typing import List, Tuple, Optional
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

# Во сколько раз документ может превышать chunk_size,
# чтобы его всё ещё держали одним чанком.
# 3 = документ до 3 * chunk_size символов остаётся целым.
WHOLE_DOC_MAX_CHUNKS = 4

# Максимум символов тела, попадающих в bm25_text.
# Milvus VARCHAR max_length у bm25_text — 65535, но большие куски
# замедляют индексацию и размывают IDF. 2000–3000 символов хватает.
BM25_BODY_LIMIT = 2500

# Максимум символов глобального bm25-комментария, добавляемого в каждый чанк.
# Без ограничения длинный комментарий размывает IDF.
GLOBAL_BM25_LIMIT = 300


class SplitterManager:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self.recursive = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ".", ",", ""],
            length_function=len,
        )
        self.header = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)


    @staticmethod
    def _make_bm25_text(
        title: str,
        body: str,
        global_bm25: Optional[str],
        limit: int = BM25_BODY_LIMIT,
    ) -> str:
        """
        Формирует bm25_text для чанка:
          - заголовок (для веса ключевых слов секции),
          - глобальный bm25 (если был в файле, обрезанный),
          - первые limit символов тела (чтобы IP, модели, имена попали в BM25).
        """
        parts: List[str] = []
        if title:
            parts.append(title)
        if global_bm25:
            parts.append(global_bm25[:GLOBAL_BM25_LIMIT])
        body = (body or "").strip()
        if body:
            parts.append(body[:limit])
        return ". ".join(parts)


    @staticmethod
    def _first_header(content: str) -> str:
        """Возвращает текст первого заголовка H1/H2 в документе."""
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("#"):
                # Убираем решётки и пробелы
                return line.lstrip("#").strip()
        return ""


    def split_by_headers(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        chunks: List[Document] = []

        for doc in documents:
            raw_content = doc.page_content

            source = doc.metadata.get("source", "N/A")

            # Извлекаем ГЛОБАЛЬНЫЙ bm25_text для всего документа
            global_bm25 = extract_global_bm25_text(raw_content)

            # Удаляем bm25-комментарии из контента, чтобы они не попали в тело чанков
            content_wo_bm25 = remove_bm25_comments(raw_content)
            logger.debug("Извлечён global_bm25: %r", global_bm25)

            content = clean_md_content(content_wo_bm25)
            if not content:
                continue

            # ────────────────────────────────────────────────────────────
            # НОВОЕ: если весь документ небольшой — оставляем одним чанком,
            # не дробим по заголовкам. Это спасает справочники вроде
            # списков камер, где фрагментация ломает контекст.
            # ────────────────────────────────────────────────────────────
            # if len(content) <= self._chunk_size * WHOLE_DOC_MAX_CHUNKS:
            #     title = doc.metadata.get("title") or self._first_header(content)
            #     chunk = Document(page_content=content)
            #     chunk.metadata["title"] = title
            #     chunk.metadata["source"] = source
            #     chunk.metadata["bm25_text"] = self._make_bm25_text(
            #         title=title,
            #         body=content,
            #         global_bm25=global_bm25,
            #     )
            #     chunk.metadata["hash"] = hash_text(content)
            #     chunks.append(chunk)
            #     logger.debug(
            #         "Документ %s (%d символов) сохранён одним чанком: %r",
            #         source, len(content), title,
            #     )
            #     continue
            # # ────────────────────────────────────────────────────────────

            try:
                header_docs = self.header.split_text(content)
            except Exception as e:
                logger.warning("Ошибка при разбиении по заголовкам: %s", e)
                header_docs = [Document(page_content=content, metadata={})]

            for h_doc in header_docs:
                # Собираем заголовки в одну строку: "Header1 > Header2 > Header3"
                headers = [
                    h_doc.metadata.get(k)
                    for _, k in HEADERS_TO_SPLIT_ON
                    if k in h_doc.metadata
                ]
                title = " > ".join(filter(None, headers)) if headers else doc.metadata.get("title", "")

                # Финализация чанка: title + bm25_text считаются
                # от ТЕЛА ИМЕННО ЭТОГО чанка (важно для sub-чанков!)
                def _finalize(chunk: Document) -> Document:
                    chunk.metadata["title"] = title
                    chunk.metadata["source"] = source
                    chunk.metadata["bm25_text"] = self._make_bm25_text(
                        title=title,
                        body=chunk.page_content,
                        global_bm25=global_bm25,
                    )
                    logger.debug(
                        "bm25_text для %r: %r",
                        title,
                        chunk.metadata["bm25_text"][:120],
                    )
                    return chunk

                # Дробим дальше, если секция больше chunk_size
                if len(h_doc.page_content) > self._chunk_size:
                    try:
                        subs = self.recursive.split_documents([h_doc])
                        for s in subs:
                            chunks.append(_finalize(s))
                    except Exception as e:
                        logger.warning("Ошибка при рекурсивном разбиении: %s", e)
                        chunks.append(_finalize(h_doc))
                else:
                    chunks.append(_finalize(h_doc))

        for c in chunks:
            c.metadata["hash"] = hash_text(c.page_content)

        return chunks