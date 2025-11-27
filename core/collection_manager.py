# core/collection_manager.py
"""
CollectionManager — модуль, который:
- загружает подпапки DATA_DIR
- индексирует каждую папку как отдельную коллекцию Milvus
- строит embedding каждой коллекции
- выбирает коллекцию для запроса по cosine similarity
- вызывает Milvus hybrid search в выбранной коллекции

Используется RAGCore, MilvusManager, твой embedder и сплиттеры.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from langchain_core.documents import Document

from core.milvus_manager import MilvusManager

logger = logging.getLogger(__name__)


# ==============================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ
# Строит embedding коллекции
# ==============================

async def build_collection_embedding(
    docs: List[Document],
    embed_query_fn,
) -> np.ndarray:
    """
    Строит embedding коллекции как взвешенный центроид всех документов.
    Использует bm25_text если есть.
    """

    vecs = []
    weights = []

    for doc in docs:
        # лучший текст для общей семантики
        txt = doc.metadata.get("bm25_text") or doc.page_content[:300]
        emb = await embed_query_fn(txt)

        # вес = sqrt длины
        w = np.sqrt(len(txt))

        vecs.append(emb)
        weights.append(w)

    vecs = np.array(vecs)
    weights = np.array(weights)

    centroid = np.average(vecs, axis=0, weights=weights)
    centroid = centroid / np.linalg.norm(centroid)

    return centroid


# ==============================
# ROUTER КОЛЛЕКЦИЙ
# ==============================

class CollectionRouter:
    """
    Хранит embedding каждой коллекции.
    Позволяет определить, куда отправить запрос.
    """

    def __init__(self, collections: Dict[str, np.ndarray]):
        self.collections = collections

    def pick_collection(self, query_vec: np.ndarray) -> str:
        """Возвращает имя коллекции с максимальной cosine similarity."""

        sims = {
            name: float(np.dot(query_vec, vec))
            for name, vec in self.collections.items()
        }

        # наибольшая косинусная близость
        return max(sims, key=sims.get)


class CollectionManager:
    """
    Управляет жизненным циклом всех коллекций:
    - загрузка документов по папкам
    - создание Milvus-коллекций
    - индексирование
    - построение роутера
    - выбор коллекции
    """

    def __init__(self, core):
        """
        core — инстанс RAGCore
        Содержит:
            - config
            - embeddings (property)
            - splitters
            - milvus (основной, можно создавать новые для каждой коллекции)
        """
        self.core = core
        self.router: Optional[CollectionRouter] = None
        self.collections: dict[str, MilvusManager] = {}  # имена коллекций -> менеджеры Milvus

    # ----------------------------------------------------------
    # Загрузка коллекций с диска и индексирование
    # ----------------------------------------------------------

    async def build_all_collections(self):
        """
        Сканирует DATA_DIR, индексирует каждую подпапку как отдельную коллекцию Milvus.
        """

        root = Path(self.core.config.DATA_DIR)
        if not root.exists():
            logger.error("DATA_DIR не существует: %s", root)
            return

        subdirs = [p for p in root.iterdir() if p.is_dir()]
        logger.info("Найдено коллекций: %d", len(subdirs))

        for folder in subdirs:
            cname = folder.name
            logger.info("📂 Индексация коллекции: %s", cname)

            docs = self.core.load_documents(str(folder))
            if not docs:
                logger.warning("Папка %s пуста — пропускаю", folder)
                continue

            await self._index_collection(cname, docs)

    # ----------------------------------------------------------
    # Реальная индексация документов в коллекцию Milvus
    # ----------------------------------------------------------

    async def _index_collection(self, collection_name: str, docs: List[Document]):
        """
        Создаёт коллекцию Milvus и индексирует туда документы.
        """

        # from core.milvus_manager import MilvusManager

        milvus = MilvusManager(
            uri=self.core.config.MILVUS_URI,
            collection_name=collection_name,
            recreate=self.core.config.RECREATE_COLLECTION,
        )

        # Чанкинг
        chunks = self.core.splitters.split_by_headers(docs)
        if not chunks:
            return

        texts = [d.page_content for d in chunks]
        titles = [d.metadata.get("title") for d in chunks]
        bm25_list = [d.metadata.get("bm25_text") for d in chunks]
        sources = [d.metadata.get("source", "N/A") for d in chunks]
        hashes = [d.metadata.get("hash") for d in chunks]

        dense = await self.core.embeddings.embed_documents(texts)
        dim = len(dense[0])

        await milvus.create_collection_if_needed(dense_dim=dim)

        # сбор записей
        rows = [
            {
                "text": t,
                "title": tit,
                "bm25_text": b25,
                "source": src,
                "hash": h,
                "dense_vector": dv,
            }
            for t, tit, b25, src, h, dv in zip(
                texts, titles, bm25_list, sources, hashes, dense
            )
        ]

        # удаление дублей
        rows = await milvus.ensure_not_duplicate_rows(rows)

        batch = self.core.config.INDEX_BATCH_SIZE
        for i in range(0, len(rows), batch):
            await milvus.insert_records(rows[i:i + batch])

        await milvus.client.load_collection(collection_name)
        logger.info("Коллекция %s загружена в память.", collection_name)

    # ----------------------------------------------------------
    # Построение Collection Router
    # ----------------------------------------------------------

    async def build_router(self):
        """
        Создаёт router: embedding каждой коллекции.
        """

        root = Path(self.core.config.DATA_DIR)
        subdirs = [p for p in root.iterdir() if p.is_dir()]

        embeddings = {}

        for folder in subdirs:
            name = folder.name
            docs = self.core.load_documents(str(folder))
            if not docs:
                continue

            centroid = await build_collection_embedding(
                docs, self.core.embeddings.embed_query
            )
            embeddings[name] = centroid

        self.router = CollectionRouter(embeddings)
        logger.info("Маршрутизатор создан: %d коллекций", len(embeddings))

    # ----------------------------------------------------------
    # Выбор коллекции для запроса
    # ----------------------------------------------------------

    async def route_query(self, query: str) -> str:
        """
        Возвращает имя коллекции, куда отправить запрос.
        """

        if not self.router:
            raise RuntimeError("Router не создан: вызови build_router().")

        qvec = await self.core.embeddings.embed_query(query)
        return self.router.pick_collection(qvec)

    # ----------------------------------------------------------
    # Выполнить поиск с роутингом
    # ----------------------------------------------------------

    async def routed_search(self, query: str):
        """
        Автоматически выбирает коллекцию и выполняет поиск в ней.
        """

        cname = await self.route_query(query)
        logger.info("Роутер выбрал коллекцию: %s", cname)

        qvec = await self.core.embeddings.embed_query(query)

        return await self.core.milvus.hybrid_search(
            query_text=query,
            query_dense=qvec,
            fetch_k=self.core.config.FETCH_K,
            top_k=self.core.config.K,
            collection_name=cname,
            reranker_endpoint=self.core.config.RERANKER_BASE_URL,
        )
