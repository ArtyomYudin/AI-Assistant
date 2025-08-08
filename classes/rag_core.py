import hashlib
import logging
import pathlib
import re
import httpx
import glob
import os
import PyPDF2

from typing import List, Dict, Any, Callable
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

from pymilvus import MilvusClient, Function, FunctionType, DataType, AnnSearchRequest

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

root = pathlib.Path(__file__).parent.parent.resolve()

# Константы
HEADERS_TO_SPLIT_ON = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
    ("####", "Header4"),
]
DENSE_DIM = 1024
SPARSE_DIM = 30522


# Простой аналог Document из LangChain---
class SimpleDoc:
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata


# Основной класс
class RAGCore:
    def __init__(self,
                 llm_name: str = "Qwen3-8B-AWQ",
                 llm_base_url: str = "http://172.20.4.50:8001/v1",
                 embedding_name: str = "multilingual-e5-large",
                 embedding_base_url: str = "http://172.20.4.50:8000/v1",
                 reranker_name: str = "bge-reranker-v2-m3",
                 reranker_base_url: str = "http://172.20.4.50:8002/v1/rerank",
                 milvus_uri: str = "http://10.3.0.5:19530",
                 collection_name: str = "demo_ci_rag",
                 ):

        self.global_unique_hashes = set()
        self.collection_name = collection_name
        self.embedding_name = embedding_name
        self.llm_name = llm_name
        self.llm_base_url = llm_base_url
        self.embedding_client = httpx.Client(base_url=embedding_base_url, timeout=60.0)
        self.http_client = httpx.Client(base_url=llm_base_url, timeout=60.0)
        self.chat_histories: Dict[str, List[Dict[str, str]]] = {}
        self.milvus_client = MilvusClient(uri=milvus_uri)
        self.reranker_name = reranker_name
        self.reranker_base_url = reranker_base_url

        # TODO: Избавиться полностью от зависимости LongChain
        # Рекурсивное разбиение с учетом структуры MD
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=[
                # "\n#{1,6} ",  # Заголовки
                # "```\n",  # Блоки кода
                # "\n\n",  # Параграфы
                # "\n",  # Строки
                # ". ",  # Предложения
                # " ",  # Пробелы
                # ""  # Символы
                "\n\n",  # параграфы
                "\n",  # строки
                " ",  # пробелы
                ".",  # предложения
                ",",  # запятые
                ""
            ],
            length_function=len,
            # is_separator_regex=False,
        )

        # TODO: Избавиться полностью от зависимости LongChain
        # Инициализация сплиттера по заголовкам
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON
        )

        # Инициализация векторной базы данных
        self.retriever = None
        self.qa_chain_with_history = None

    def __del__(self):
        if hasattr(self, 'http_client'):
            self.http_client.close()

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self.embedding_client.post("/embeddings", json={
                "model": self.embedding_name,
                "input": texts
            })
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            logger.error(f"Ошибка при получении эмбеддингов: {e}")
            return []

    def _call_llm(self, messages: List[dict], max_tokens: int = 8192, temperature: float = 0.7) -> str:
        """
        Прямой вызов LLM через vLLM-совместимый API.
        :param messages: Список сообщений в формате OpenAI: [{"role": "user", "content": "..."}, ...]
        :return: Ответ модели как строка
        """
        payload = {
            "model": self.llm_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}
        }
        try:
            response = self.http_client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Ошибка при вызове LLM: {e}")
            return "Извините, произошла ошибка при генерации ответа."

    def _stream_llm_response(self, messages: List[dict], on_token: Callable[[str], None]):
        """
        Stream-ответ от LLM через SSE
        """
        payload = {
            "model": self.llm_name,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 8192,
            "chat_template_kwargs": {"enable_thinking": False}
        }
        try:
            with httpx.stream("POST", f"{self.llm_base_url}/chat/completions", json=payload, timeout=60.0) as response:
                response.raise_for_status()
                full_response = ""
                for line in response.iter_lines():
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            import json
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                full_response += delta
                                on_token(delta)  # Вызываем колбэк
                        except:
                            continue
                return full_response
        except Exception as e:
            logger.error(f"Ошибка при streaming: {e}")
            on_token("Извините, произошла ошибка при генерации ответа.")
            return "Ошибка"

    def _hash_text(self, text) -> str:
        """
        Функция принимает строковое значение, кодирует его в байты и вычисляет SHA-256 хэш.
        Хэш будет необходим для сравнения и выявления повторяющейся информации.
        :param text:
        :return: Безопасный хеш(дайджест) как строковый объект двойной длины, содержащий только шестнадцатеричные цифры.
        """
        hash_object = hashlib.sha256(text.encode())
        return hash_object.hexdigest()

    def _get_header_values(self, split_doc: SimpleDoc, ) -> list[str]:
        """
        Получите текстовые значения заголовков в документе, полученном в результате разделения.
        """
        header_keys = [header_key for _, header_key in HEADERS_TO_SPLIT_ON]

        return [
            header_value
            for header_key in header_keys
            if (header_value := split_doc.metadata.get(header_key)) is not None
        ]

    def _clean_md_content(self, content: str) -> str:
        """
        Очистка MD-контента от лишних элементов
        """
        # Удаление HTML комментариев
        content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
        # Удаление ссылок в формате [text](url) оставляя текст
        content = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", content)
        # Удаление изображений
        content = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", "", content)
        # Нормализация пробелов
        # content = re.sub(r"\s+", " ", content)
        # Удаление множественных пустых строк
        content = re.sub(r"\n\s*\n", "\n", content)
        return content.strip()

    def load_documents_from_directory(self,
                                      directory_path: str = f"{root}/scraped_data",
                                      file_extensions: List[str] = None
                                      ) -> List[SimpleDoc]:
        """
        Загрузка всех документов из директории
        """
        if file_extensions is None:
            file_extensions = ['*.pdf', '*.txt', '*.md']

        load_documents = []

        for extension in file_extensions:
            pattern = os.path.join(directory_path, extension)
            for path in glob.glob(pattern):
                try:
                    if path.endswith(".pdf"):
                        with open(path, "rb") as f:
                            reader = PyPDF2.PdfReader(f)
                            text = "\n".join([page.extract_text() for page in reader.pages])
                        load_documents.append(SimpleDoc(text, {"source": path}))
                    else:
                        with open(path, "r", encoding="utf-8") as f:
                            load_documents.append(SimpleDoc(f.read(), {"source": path}))
                except Exception as e:
                    logger.error(f"Ошибка при загрузке {path}: {e}")

        logger.info(f"Загружено {len(load_documents)} документов из директории {directory_path}")

        return load_documents

    def split_by_headers(self, load_documents: List[SimpleDoc]) -> List[SimpleDoc]:
        """
        Разбиение по заголовкам с сохранением иерархии
        """
        chunks = []

        for doc in load_documents:
            # очистка документа
            clear_content = self._clean_md_content(doc.page_content)
            # Сначала разбиваем по заголовкам
            header_docs = self.header_splitter.split_text(clear_content)
            # Затем каждый раздел дополнительно разбиваем если нужно
            for header_doc in header_docs:
                # # Пропускаем только заголовки без содержания
                # if re.match(r'^#+\s+[^\n]*\s*$', header_doc.page_content):
                #     continue
                headers = self._get_header_values(header_doc)
                header_text = " > ".join(headers) + "\n" if headers else ""
                if len(header_doc.page_content) > self.recursive_splitter._chunk_size:
                    # Если раздел большой - дополнительно разбиваем
                    sub_docs = self.recursive_splitter.split_documents([header_doc])
                    # Добавляем заголовок, чтобы сохранить контекст и улучшить поиск
                    for sub_doc in sub_docs:
                        sub_doc.page_content = header_text + sub_doc.page_content
                    chunks.extend(sub_docs)
                else:
                    # Добавляем заголовок, чтобы сохранить контекст и улучшить поиск
                    header_doc.page_content = header_text + header_doc.page_content
                    chunks.append(header_doc)

        logger.info(f"Разделено {len(load_documents)} документов на {len(chunks)} фрагментов.")

        # Удаление дубликатов, на основе хэш
        unique_chunks = []
        for chunk in chunks:
            chunk_hash = self._hash_text(chunk.page_content)
            if chunk_hash not in self.global_unique_hashes:
                unique_chunks.append(chunk)
                self.global_unique_hashes.add(chunk_hash)

        logger.info(f"Количество Уникальных фрагментов {len(unique_chunks)}.")

        return unique_chunks

    def _create_collection_if_not_exists(self):
        """
        Создаем коллекцию с hybrid-схемой (dense + sparse), если её нет.
        """
        if self.milvus_client.has_collection(self.collection_name):
            logger.info(f"Коллекция {self.collection_name} уже существует")
            self.milvus_client.drop_collection(self.collection_name)

        schema = self.milvus_client.create_schema(auto_id=True, description="CentrInform Demo RAG Collection")
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=512),
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=DENSE_DIM),
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR),

        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["sparse_vector"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)

        # Создание индексов
        index_params = self.milvus_client.prepare_index_params()
        index_params.add_index(
            index_name="dense_index",
            field_name="dense_vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200}
        )

        index_params.add_index(
            index_name="sparse_index",
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25"
        )

        self.milvus_client.create_collection(self.collection_name, schema=schema, index_params=index_params)

        logger.info(f"Коллекция {self.collection_name} создана с индексами")

    def setup_vectorstore(self, documents: List[SimpleDoc]) -> None:
        """
        Настройка векторной базы данных Milvus.
        Создание коллекции
        Вставка документов
        """
        processed_docs = self.split_by_headers(documents)

        if not processed_docs:
            logger.warning("Нет документов для индексации")
            return

        self._create_collection_if_not_exists()

        texts = [doc.page_content for doc in processed_docs]
        sources = [doc.metadata.get("source", "N/A") for doc in processed_docs]

        try:
            dense_vectors = self._get_embeddings(texts)

            data = [
                {"text": text, "source": src, "dense_vector": dv}
                for text, src, dv in zip(texts, sources, dense_vectors)
            ]

            # Вставка
            self.milvus_client.insert(collection_name=self.collection_name, data=data)
            self.milvus_client.flush(collection_name=self.collection_name)
            logger.info(f"Проиндексировано {len(processed_docs)} документов")

        except Exception as e:
            logger.error(f"Ошибка при вставке в Milvus: {e}")
            raise

    def _hybrid_search(self, query: str, k: int = 7, fetch_k: int = 20) -> List[SimpleDoc]:
        """
        Выполняет hybrid-поиск (dense + sparse) с reranking.
        """
        try:
            # Получаем dense векторы запроса
            query_dense = self._get_embeddings([query])[0]

            # Выполнение поиска
            search_param_dense = {
                "data": [query_dense],
                "anns_field": "dense_vector",
                "limit": fetch_k,
                "param": {"ef": 100}
            }
            search_param_sparse = {
                "data": [query],
                "anns_field": "sparse_vector",
                "limit": fetch_k,
                "param": {"drop_ratio_search": 0.2}
            }
            request_1 = AnnSearchRequest(**search_param_dense)
            request_2 = AnnSearchRequest(**search_param_sparse)
            reqs = [request_1, request_2]

            # rerank = Function(
            #     name="weight",
            #     input_field_names=[],  # Must be an empty list
            #     function_type=FunctionType.RERANK,
            #     params={
            #         "reranker": "weighted",
            #         "weights": [0.6, 0.4],
            #         "norm_score": True  # Optional
            #     }
            # )

            vllm_ranker = Function(
                name="vllm_semantic_ranker",  # Choose a descriptive name
                input_field_names=["text"],  # Field containing text to rerank
                function_type=FunctionType.RERANK,  # Must be RERANK
                params={
                    "reranker": "model",  # Specifies model-based reranking
                    "provider": "vllm",  # Specifies vLLM service
                    "queries": [query],  # Query text
                    "endpoint": self.reranker_base_url,  # vLLM service address
                    "maxBatch": 64,  # Optional: batch size
                    "truncate_prompt_tokens": 256,  # Optional: Use last 256 tokens
                }
            )

            results = self.milvus_client.hybrid_search(
                collection_name=self.collection_name,
                reqs=reqs,
                # ranker=RRFRanker(k=60),
                ranker= vllm_ranker,
                output_fields=["text", "source"],
                limit=k
            )

            # Преобразуем в Document
            retrieved_docs = []
            for res in results[0]:
                doc = SimpleDoc(
                    page_content=res.entity.get("text"),
                    metadata={"source": res.entity.get("source"), "score": res.score}
                )
                retrieved_docs.append(doc)

            return retrieved_docs

        except Exception as e:
            logger.error(f"Ошибка при hybrid-поиске: {e}")
            return []

    def create_retriever(self, k: int = 7, fetch_k: int = 20) -> None:
        """
        Создание ретривера для поиска
        """
        def retrieve(query) -> List[SimpleDoc]:
            return self._hybrid_search(query, k=k, fetch_k=fetch_k)
        self.retriever = retrieve
        logger.info("Ретривер создан")

    def clear_session(self, session_id: str):
        self.chat_histories.pop(session_id, None)

    def get_chat_history(self, session_id: str) -> List[dict]:
        return self.chat_histories.get(session_id, [])

    def setup_qa_chain(self) -> None:
        """
        Настройка QA-логики без LangChain — только pymilvus, ручной промпт, httpx.
        """
        if not self.retriever:
            logger.error("Ретривер не создан. Вызовите create_retriever().")
            return

        # def qa_chain(question: str, chat_history: str = "") -> str:
        #     #   Получаем релевантные документы
        #     docs = self.retriever(question)
        #     if not docs:
        #         return "Информация не найдена."
        #
        #     # Форматируем контекст
        #     context = "\n\n".join([
        #         f"[Источник: {doc.metadata.get('source', 'N/A')}] {doc.page_content.strip()}"
        #         for doc in docs
        #     ])
        #
        #     # 3. Формируем промпт вручную
        #     prompt = (
        #         "Вы — экспертный ассистент организации АО «ЦентрИнформ». Отвечайте ТОЛЬКО на основе контекста.\n\n"
        #         "Контекст:\n"
        #         f"{context}\n\n"
        #         "История диалога:\n"
        #         f"{chat_history}\n\n"
        #         f"Текущий вопрос: {question}\n\n"
        #         "Правила:\n"
        #         "1. Используйте ТОЛЬКО контекст и историю.\n"
        #         "2. Отвечайте чётко, на русском языке.\n"
        #         "3. Если информации нет — скажите: 'Информация не найдена'.\n"
        #         "4. Не добавляйте предположений.\n\n"
        #         "Ответ:"
        #     )
        #
        #     # 4. Генерируем ответ через прямой HTTP-вызов
        #     messages = [{"role": "user", "content": prompt}]
        #     return self._call_llm(messages)

        def qa_chain(question: str, chat_history: str = "", on_token: Callable[[str], None] = print) -> str:
            docs = self.retriever(question)
            if not docs:
                on_token("Информация не найдена.")
                return "Информация не найдена."

            context = "\n\n".join([
                f"[Источник: {doc.metadata.get('source', 'N/A')}] {doc.page_content.strip()}"
                for doc in docs
            ])
            prompt = (
                "Вы — экспертный ассистент АО «ЦентрИнформ». Отвечайте ТОЛЬКО на основе контекста.\n\n"
                f"Контекст:\n{context}\n\n"
                f"История диалога:\n{chat_history}\n\n"
                f"Текущий вопрос: {question}\n\n"
                "Правила:\n"
                "1. Только на основе контекста.\n"
                "2. На русском языке.\n"
                "3. Если нет — 'Информация не найдена'.\n"
                "4. Без предположений.\n\n"
                "Ответ:"
            )
            messages = [{"role": "user", "content": prompt}]
            return self._stream_llm_response(messages, on_token)

        # Сохраняем как внутренний метод
        self.qa_chain_with_history = qa_chain
        logger.info("Цепочка QA настроена")

    def search_similar_documents(self, query: str, k: int = 5, fetch_k=20) -> List[Dict]:
        """
        Поиск похожих документов
        """
        if not self.vectorstore:
            return []

        try:
            docs = self.vectorstore.similarity_search_with_score(
                query, k=k,
                fetch_k=fetch_k,
                ranker_type="weighted",
                ranker_params={"weights": [0.6, 0.4]}
            )
            results = []

            for doc, score in docs:
                results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "N/A"),
                    "score": float(score),
                    "metadata": doc.metadata
                })

            return results

        except Exception as e:
            logger.error(f"Ошибка при поиске документов: {e}")
            return []

    # def ask_question_rag(self, question: str, session_id: str = "default") -> Dict[str, Any]:
    #     """
    #         Ответ на вопрос с RAG и историей чата, без LangChain Runnables.
    #         Только pymilvus + ручной промпт + httpx.
    #         """
    #     if not self.qa_chain_with_history:
    #         return {"error": "Цепочка QA не настроена", "question": question}
    #
    #     try:
    #         # 1. Получаем историю чата
    #         history = self.get_chat_history(session_id)
    #         chat_history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history])
    #
    #         # Генерируем ответ
    #         answer = self.qa_chain_with_history(question, chat_history_text)
    #
    #         # Сохраняем в историю
    #         self.chat_histories.setdefault(session_id, []).append({"role": "user", "content": question})
    #         self.chat_histories[session_id].append({"role": "ai", "content": answer})
    #
    #         # 4. Получаем источники
    #         relevant_docs = self.retriever(question)
    #         sources = []
    #         for doc in relevant_docs:
    #             sources.append({
    #                 "source": doc.metadata.get("source", "Неизвестный источник"),
    #                 "content": doc.page_content.strip(),
    #                 "score": doc.metadata.get("score", None)
    #             })
    #
    #         # 5. Возвращаем результат
    #         return {
    #             "question": question,
    #             "answer": answer,
    #             "sources": sources[:3],
    #             "session_id": session_id,
    #             "chat_history": self.get_chat_history(session_id)
    #         }
    #
    #     except Exception as e:
    #         logger.error(f"Ошибка при обработке вопроса: {e}")
    #         return {
    #             "error": f"Ошибка при генерации ответа: {str(e)}",
    #             "question": question,
    #             "session_id": session_id
    #         }

    def ask_question_rag(self, question: str, session_id: str = "default", on_token: Callable[[str], None] = None) -> \
    Dict[str, Any]:
        if not hasattr(self, 'qa_chain_with_history'):
            return {"error": "QA не настроена", "question": question}

        if on_token is None:
            on_token = lambda x: print(x, end="", flush=True)

        try:
            history = self.get_chat_history(session_id)
            chat_history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history])

            full_answer = []

            def token_handler(token: str):
                on_token(token)
                full_answer.append(token)

            answer = self.qa_chain_with_history(question, chat_history_text, on_token=token_handler)
            final_answer = "".join(full_answer)

            # Сохраняем в историю
            self.chat_histories.setdefault(session_id, []).append({"role": "user", "content": question})
            self.chat_histories[session_id].append({"role": "ai", "content": final_answer})

            # Источники
            relevant_docs = self.retriever(question)
            sources = [
                {
                    "source": doc.metadata.get("source", "N/A"),
                    "content": doc.page_content.strip(),
                    "score": doc.metadata.get("score")
                }
                for doc in relevant_docs[:3]
            ]

            return {
                "question": question,
                "answer": final_answer,
                "sources": sources,
                "session_id": session_id,
                "chat_history": self.get_chat_history(session_id)
            }

        except Exception as e:
            logger.error(f"Ошибка: {e}")
            on_token("Произошла ошибка при генерации ответа.")
            return {"error": str(e), "question": question}
