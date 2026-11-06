# core/ingestion/document_indexer.py

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
from typing import List, Optional

from langchain_core.documents import Document

from config.rag_config import RAGConfig
from core.ingestion.document_loader import load_documents_from_directory
from core.storage.milvus_manager import MilvusManager
from core.models.model_provider import ModelProvider
from core.ingestion.splitter import SplitterManager

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """Загрузка документов и каноничная индексация в коллекцию Milvus."""

    def __init__(
        self,
        config: RAGConfig,
        splitters: SplitterManager,
        models: ModelProvider,
        milvus: MilvusManager,
    ):
        self.config = config
        self.splitters = splitters
        self.models = models
        self.milvus = milvus

    def load_documents(self, directory: Optional[str] = None) -> List[Document]:
        directory = directory or self.config.DATA_DIR
        return load_documents_from_directory(directory)

    async def setup_vectorstore(self, documents: List[Document]) -> None:
        total = await self.index_documents(self.config.COLLECTION_NAME, documents)
        if total == 0:
            logger.warning("setup_vectorstore: вставлено 0 фрагментов")

    async def index_documents(
        self,
        collection_name: str,
        documents: List[Document],
    ) -> int:
        """
        Возвращает число вставленных фрагментов.

        Правила:
          - text       = тело чанка БЕЗ title
          - text_full  = title + body (MilvusManager.insert_records)
          - dense      = эмбеддинг ТОЛЬКО от body
          - bm25_text  = title + global_bm25 + body[:2500]
        """
        if not documents:
            return 0

        processed = self.splitters.split_by_headers(documents)
        if not processed:
            logger.warning("Нет чанков после сплиттинга для %s", collection_name)
            return 0

        texts = [d.page_content for d in processed]
        titles = [d.metadata.get("title", "") for d in processed]
        bm25_texts = [d.metadata.get("bm25_text", "") for d in processed]
        sources = [d.metadata.get("source", "N/A") for d in processed]
        hashes = [d.metadata.get("hash", "") for d in processed]

        try:
            dense_vectors = await self.models.embeddings.embed_documents(texts)
        except Exception as e:
            logger.exception("Ошибка эмбеддинга для %s: %s", collection_name, e)
            raise

        if not dense_vectors:
            logger.error("Пустой список эмбеддингов для %s", collection_name)
            return 0
        dense_dim = len(dense_vectors[0])

        milvus = MilvusManager(
            uri=self.config.MILVUS_URI,
            collection_name=collection_name,
            recreate=self.config.RECREATE_COLLECTION,
        )
        await milvus.create_collection_if_needed(dense_dim=dense_dim)

        rows = [
            {
                "text": t,
                "title": tit,
                "bm25_text": bm25,
                "source": s,
                "hash": h,
                "dense_vector": dv,
                "chunk_index": idx,
            }
            for idx, (t, tit, bm25, s, h, dv) in enumerate(zip(
                texts, titles, bm25_texts, sources, hashes, dense_vectors
            ))
        ]

        unique = await milvus.ensure_not_duplicate_rows(rows)
        if not unique:
            logger.info("Все фрагменты уже в %s", collection_name)
            return 0

        bs = self.config.INDEX_BATCH_SIZE
        total = 0
        for i in range(0, len(unique), bs):
            total += await milvus.insert_records(unique[i:i + bs])

        await milvus.client.load_collection(collection_name)
        logger.info(
            "Коллекция %s проиндексирована: %d фрагментов (dim=%d)",
            collection_name, total, dense_dim,
        )
        return total