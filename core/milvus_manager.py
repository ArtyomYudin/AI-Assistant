import logging
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.documents import Document
from pymilvus import AnnSearchRequest, DataType, Function, FunctionType

from core.utils import hash_text

logger = logging.getLogger(__name__)

class MilvusManager:
    def __init__(self, uri: str, collection_name: str, recreate: bool = False):
        from pymilvus import MilvusClient
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name
        self._recreate = recreate

    def create_collection_if_needed(self, dense_dim: int) -> None:
        if self.client.has_collection(self.collection_name):
            if self._recreate:
                logger.info("Удаление существующей коллекции %s", self.collection_name)
                self.client.drop_collection(self.collection_name)
            else:
                logger.info("Используется существующая коллекция %s", self.collection_name)
                return

        schema = self.client.create_schema(auto_id=True, description="CentrInform RAG")
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="hash", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dense_dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        schema.add_function(Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["sparse_vector"],
            function_type=FunctionType.BM25,
        ))

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="dense_vector", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 200})
        index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")

        self.client.create_collection(self.collection_name, schema=schema, index_params=index_params)
        logger.info("Коллекция %s создана (dense_dim=%d)", self.collection_name, dense_dim)

    def is_duplicate_hash(self, h: str) -> bool:
        try:
            res = self.client.query(collection_name=self.collection_name, filter=f"hash == '{h}'", output_fields=["id"], limit=1)
            return bool(res)
        except Exception as e:
            logger.warning("Ошибка при проверке дубликата: %s", e)
            return False

    def ensure_not_duplicate_rows(self, rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique: List[Dict[str, Any]] = []
        for row in rows:
            h = row.get("hash") or hash_text(row.get("text", ""))
            if not self.is_duplicate_hash(h):
                row["hash"] = h
                unique.append(row)
        return unique

    def insert_records(self, rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        try:
            self.client.insert(self.collection_name, rows)
            self.client.flush(self.collection_name)
            return len(rows)
        except Exception as e:
            logger.exception("Ошибка при вставке батча: %s", e)
            return 0

    def hybrid_search(self, query_text: str, query_dense: List[float], fetch_k: int, top_k: int, reranker_endpoint: str = "") -> List[Document]:
        try:
            req_dense = AnnSearchRequest(data=[query_dense], anns_field="dense_vector", limit=fetch_k, param={"ef": 100})
            req_sparse = AnnSearchRequest(data=[query_text], anns_field="sparse_vector", limit=fetch_k, param={"drop_ratio_search": 0.2})

            ranker = None
            if reranker_endpoint:
                ranker = Function(name="vllm_semantic_ranker", input_field_names=["text"], function_type=FunctionType.RERANK, params={
                    "reranker": "model", "provider": "vllm", "queries": [query_text], "endpoint": reranker_endpoint, "maxBatch": 64, "truncate_prompt_tokens": 256
                })

            results = self.client.hybrid_search(collection_name=self.collection_name, reqs=[req_dense, req_sparse], ranker=ranker, output_fields=["text","source","hash"], limit=top_k)
            docs: List[Document] = []
            if results and results[0]:
                for res in results[0]:
                    ent = getattr(res, "entity", None)
                    if ent:
                        docs.append(Document(page_content=ent.get("text",""), metadata={"source": ent.get("source","N/A"), "hash": ent.get("hash",""), "score": getattr(res,"score",0)}))
            return docs
        except Exception as e:
            logger.exception("Ошибка при гибридном поиске: %s", e)
            return []
