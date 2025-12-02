import logging
from typing import Any, Dict, Iterable, List

from langchain_core.documents import Document
from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusException
from pymilvus.client.types import LoadState

from core.utils import hash_text, remove_stopwords_russian

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s: %(message)s")
logger = logging.getLogger(__name__)

class MilvusManager:
    """
    Класс-менеджер для работы с Milvus (векторной БД).
    Поддерживает:
      - Создание коллекции с нужной схемой
      - Индексацию документов
      - Проверку дубликатов
      - Гибридный поиск (dense + sparse)
      - Использование reranker-а (например, vLLM)
    """
    def __init__(self, uri: str, collection_name: str, recreate: bool = False):
        """
         :param uri: адрес подключения к Milvus
         :param collection_name: название коллекции
         :param recreate: если True — пересоздать коллекцию (удалить старую)
        """
        from pymilvus import AsyncMilvusClient
        self.client = AsyncMilvusClient(uri=uri)
        self.collection_name = collection_name
        self._recreate = recreate

    async def close(self) -> None:
        """Закрытие соединения с Milvus (если клиент поддерживает close)."""
        if hasattr(self.client, "close"):
            try:
                await self.client.close()
            except (MilvusException, ConnectionError) as e:
                logger.warning("Ошибка при закрытии Milvus: %s", e)
            except Exception as e:
                logger.error("Неожиданная ошибка при закрытии Milvus: %s", e)
                raise

    async def create_collection_if_needed(self, dense_dim: int) -> None:
        """
        Создание коллекции в Milvus, если её ещё нет.
        Если recreate=True, то удаляется старая коллекция.
        В схему добавляются:
            - текст
            - источник
            - hash (для проверки дубликатов)
            - dense-вектор (для семантического поиска)
            - sparse-вектор (BM25 для поиска по ключевым словам)
        """
        if await self.client.has_collection(self.collection_name):
            if self._recreate:
                logger.info("Удаление существующей коллекции %s", self.collection_name)
                await self.client.drop_collection(self.collection_name)
            else:
                logger.info("Используется существующая коллекция %s", self.collection_name)
                return

        # Создание схемы коллекции
        schema = self.client.create_schema(auto_id=True, description="CentrInform RAG")
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True) # чисто тело
        schema.add_field(field_name="text_full", datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True)  # title + тело
        schema.add_field(field_name="bm25_text", datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="hash", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dense_dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # Добавляем BM25 для sparse-вектора (по text_full)
        schema.add_function(Function(
            name = "text_bm25_emb",
            # input_field_names = ["text_full"],
            input_field_names=["bm25_text"],
            output_field_names = ["sparse_vector"],
            function_type = FunctionType.BM25,
        ))

        # Индексы
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name = "dense_vector",
            index_type = "HNSW",
            metric_type = "COSINE",
            params = {"M": 16, "efConstruction": 200}
        )
        index_params.add_index(
            field_name = "sparse_vector",
            index_type = "SPARSE_INVERTED_INDEX",
            metric_type = "BM25",
            params={
                "bm25_k1": 1.6,
                "bm25_b": 0.75,  # 0.0 отключаем нормализацию по длине
            }
        )

        await self.client.create_collection(self.collection_name, schema = schema, index_params = index_params)
        logger.info("Коллекция %s создана (dense_dim=%d)", self.collection_name, dense_dim)

    async def ensure_collection_loaded(self, collection_name: str) -> None:
        """
        Проверка, загружена ли коллекция в память. Если нет загружаем
        """
        try:
            state_response = await self.client.get_load_state(collection_name)
            if state_response != LoadState.Loaded:
                await self.client.load_collection(collection_name)
                logger.info("Коллекция %s успешно загружена", collection_name)
            else:
                logger.info("Коллекция %s уже загружена", collection_name)
        except Exception as e:
            logger.error("Ошибка при загрузке коллекции %s: %s", collection_name, str(e))


    async def is_duplicate_hash(self, h: str) -> bool:
        """
        Проверка, есть ли документ с данным hash в коллекции.
        :param h: hash текста
        :return: True если уже существует
        """
        try:
            res = await self.client.query(
                collection_name = self.collection_name,
                filter = f"hash == '{h}'",
                output_fields = ["id"],
                limit = 1
            )
            return isinstance(res, list) and len(res) > 0
        except Exception as e:
            logger.warning("Ошибка при проверке дубликата: %s", e)
            return False

    async def ensure_not_duplicate_rows(self, rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Фильтрует список документов, убирая дубликаты по hash.
        :param rows: документы в формате dict
        :return: только уникальные документы
        """
        unique: List[Dict[str, Any]] = []
        for row in rows:
            h = row.get("hash") or hash_text(row.get("text", ""))
            if not await self.is_duplicate_hash(h):
                row["hash"] = h
                unique.append(row)
        return unique

    def filter_documents(self, docs: List[Document], query: str, score_threshold: float = 0.01) -> List[Document]:
        """
        Фильтрация документов после поиска.
        - Оставляем только те, что прошли порог score
        - Дополнительно можно добавить проверку на наличие ключевых слов из запроса
        """
        filtered = [d for d in docs if d.metadata.get("score", 1) <= score_threshold]

        # Если после фильтрации ничего не осталось — возвращаем топ-1 как fallback
        if not filtered and docs:
            filtered = [docs[0]]

        return filtered

    async def insert_records(self, rows: List[Dict[str, Any]]) -> int:
        """
        Вставка записей в Milvus.
        :param rows: список документов
        :return: число успешно вставленных документов
        """
        if not rows:
            return 0
        try:
            for row in rows:
                title = row.get("title", "")
                body = row.get("text", "")
                row["text_full"] = f"{title}. {body}" if title else body
                # row["bm25_text"] = row.get("bm25_text")
            await self.client.insert(self.collection_name, rows)
            await self.client.flush(collection_name=self.collection_name)
            return len(rows)
        except Exception as e:
            logger.exception("Ошибка при вставке батча: %s", e)
            return 0

    async def hybrid_search(self,
                            query_text: str,
                            query_dense: List[float],
                            fetch_k: int,
                            top_k: int,
                            collection_name: str = "",
                            reranker_endpoint: str = ""
                            ) -> List[Document]:
        """
        Гибридный поиск: dense + sparse + (опционально) reranker.
        :param query_text: исходный текстовый запрос
        :param query_dense: вектор эмбеддинга запроса
        :param fetch_k: сколько кандидатов брать до rerank
        :param top_k: сколько вернуть в ответе
        :param collection_name: имя коллекции
        :param reranker_endpoint: если указан — использовать reranker (например, vLLM)
        :return: список документов LangChain Document
        """
        try:
            # Поиск по dense-вектору
            req_dense = AnnSearchRequest(
                data = [query_dense],
                anns_field = "dense_vector",
                limit = fetch_k,
                param = {"ef": 100}
            )

            # Поиск по sparse-вектору (BM25)
            query_for_sparse = remove_stopwords_russian(query_text) # удаление стоп-слов
            req_sparse = AnnSearchRequest(
                data = [query_for_sparse],
                anns_field = "sparse_vector",
                limit = fetch_k,
                param = {"drop_ratio_search": 0.2}
            )

            # Подключаем reranker, если указан endpoint
            ranker = None
            if reranker_endpoint:
                ranker = Function(
                    name = "vllm_semantic_ranker",
                    input_field_names = ["text"],
                    function_type = FunctionType.RERANK,
                    params = {
                        "reranker": "model",
                        "provider": "vllm",
                        "queries": [query_text],
                        "endpoint": reranker_endpoint,
                        "maxBatch": 64,
                        "truncate_prompt_tokens": 256
                    }
                )

            # Выполняем гибридный поиск
            results = await self.client.hybrid_search(
                collection_name = collection_name, #self.collection_name,
                reqs = [req_dense, req_sparse],
                ranker = ranker,
                output_fields = ["text","title", "bm25_text", "source","hash"],
                limit = top_k
            )

            # Преобразуем результаты в LangChain Document
            docs: List[Document] = []
            for hits in results:  # для каждой query
                for res in hits:
                    ent = res.get("entity", None) if isinstance(res, dict) else getattr(res, "entity", None)
                    if ent:
                        docs.append(
                            Document(
                                page_content = ent.get("text",""),
                                metadata = {
                                    "title": ent.get("title", ""),
                                    "bm25_text": ent.get("bm25_text"),
                                    "source": ent.get("source","N/A"),
                                    "hash": ent.get("hash",""),
                                    "score": getattr(res,"score",0)
                                }
                            )
                        )

            docs = self.filter_documents(docs, query_text, score_threshold=0.01)
            return docs
        except Exception as e:
            logger.exception("Ошибка при гибридном поиске: %s", e)
            return []
