import hashlib
import logging
import pathlib
import re

from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.retrievers import MultiQueryRetriever

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage


# from classes.custom_compressor import CustomBGERerankerCompressor
# from langchain.retrievers import ContextualCompressionRetriever


from classes.e5_mistrall_embeddings import E5MistralEmbeddings

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

root = pathlib.Path(__file__).parent.parent.resolve()

HEADERS_TO_SPLIT_ON = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
    ("####", "Header4"),
]

class RAGCore:
    def __init__(self,
                 llm_name: str = "Qwen3-8B-AWQ",
                 llm_base_url: str = "http://172.20.4.50:8001/v1",
                 embedding_name: str = "multilingual-e5-large",
                 embedding_base_url: str = "http://172.20.4.50:8000/v1",
                 #reranker_name: str = "BAAI/bge-reranker-large",
                 milvus_base_uri: str = "http://10.1.0.2:19530",
                 collection_name: str = "demo_ci_rag",
                 ):

        self.global_unique_hashes = set()

        self.collection_name = collection_name
        self.milvus_connection_args = {"uri": milvus_base_uri}

        # Хранение истории чатов по сессиям
        self.chat_histories: Dict[str, InMemoryChatMessageHistory] = {}

        # Dense Embeddings (через vLLM)
        self.embeddings = OpenAIEmbeddings(
            model=embedding_name,
            api_key="EMPTY",
            base_url=embedding_base_url,
            embedding_ctx_length=512,
            timeout=60,
        )

        # LLM для генерации и multiquery (через vLLM)
        self.llm = ChatOpenAI(
            model=llm_name,
            api_key="EMPTY",
            base_url=llm_base_url,
            max_tokens=8192,
            temperature=0.7,
            streaming=True,
            # callbacks=[StreamingStdOutCallbackHandler()],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

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

        # Инициализация сплиттера по заголовкам
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON
        )

        # Инициализация векторной базы данных
        self.vectorstore = None
        self.retriever = None
        self.qa_chain_with_history = None

    def hash_text(self, text) -> str:
        """
        Функция принимает строковое значение, кодирует его в байты и вычисляет SHA-256 хэш.
        Хэш будет необходим для сравнения и выявления повторяющейся информации.
        :param text:
        :return: Безопасный хеш(дайджест) как строковый объект двойной длины, содержащий только шестнадцатеричные цифры.
        """
        hash_object = hashlib.sha256(text.encode())
        return hash_object.hexdigest()

    def get_header_values(self, split_doc: Document,) -> list[str]:
        """
        Получите текстовые значения заголовков в документе, полученном в результате разделения.
        """
        header_keys = [header_key for _, header_key in HEADERS_TO_SPLIT_ON]

        return [
            header_value
            for header_key in header_keys
            if (header_value := split_doc.metadata.get(header_key)) is not None
        ]


    def clean_md_content(self, content: str) -> str:
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
                                      ) -> List[Document]:
        """
        Загрузка всех документов из директории
        """
        if file_extensions is None:
            file_extensions = ['*.pdf', '*.txt', '*.md', '*.doc', '*.docx']

        load_documents = []

        for extension in file_extensions:
            try:
                loader = DirectoryLoader(
                    directory_path,
                    glob=extension,
                    loader_cls=PyPDFLoader if extension == '*.pdf' else TextLoader,
                    loader_kwargs={'encoding': 'utf-8'} if extension in ['*.txt', '*.md'] else {}
                )
                docs = loader.load()
                load_documents.extend(docs)
            except Exception as e:
                logger.error(f"Ошибка при загрузке файлов {extension}: {e}")

        logger.info(f"Загружено {len(load_documents)} документов из директории {directory_path}")

        return load_documents


    def split_by_headers(self, load_documents: List[Document]) -> List[Document]:
        """
        Разбиение по заголовкам с сохранением иерархии
        """
        chunks = []

        for doc in load_documents:
            # очистка документа
            clear_content = self.clean_md_content(doc.page_content)
            # Сначала разбиваем по заголовкам
            header_docs = self.header_splitter.split_text(clear_content)
            # Затем каждый раздел дополнительно разбиваем если нужно
            for header_doc in header_docs:
                # # Пропускаем только заголовки без содержания
                # if re.match(r'^#+\s+[^\n]*\s*$', header_doc.page_content):
                #     continue
                headers = self.get_header_values(header_doc)
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
            chunk_hash = self.hash_text(chunk.page_content)
            if chunk_hash not in self.global_unique_hashes:
                unique_chunks.append(chunk)
                self.global_unique_hashes.add(chunk_hash)

        logger.info(f"Количество Уникальных фрагментов {len(unique_chunks)}.")

        return unique_chunks


    def setup_vectorstore(self, documents: List[Document]) -> None:
        """
        Настройка векторной базы данных Milvus
        """
        processed_docs = self.split_by_headers(documents)

        if not processed_docs:
            logger.warning("Нет документов для индексации")
            return

        # Если используем Mitral- подобные, то добавляем инструкции
        # e5_embeddings = E5MistralEmbeddings(self.embeddings)

        # Параметры индекса (нужно поиграться бля большей производительности)
        dense_index_param= {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {
                "M": 16,
                "efConstruction": 200,
                "ef": 100  # чем выше — точнее, но медленнее
                # ef — влияет на качество поиска. В поиске можно поставить ef=100, при построении — efConstruction=200.
            },
        }

        sparse_index_param = {
            "metric_type": "BM25",
            "index_type": "SPARSE_INVERTED_INDEX",
            # "params": {"drop_ratio_build": 0.2},
        }

        try:
            # Создание или подключение к коллекции Milvus
            self.vectorstore = Milvus.from_documents(
                documents=processed_docs,
                embedding=self.embeddings,
                builtin_function=BM25BuiltInFunction(output_field_names="sparse"),
                # embedding=e5_embeddings,
                connection_args=self.milvus_connection_args,
                collection_name=self.collection_name,
                drop_old=True,  # Только для dev. В prod — реализовать проверку изменения
                consistency_level="Strong",
                enable_dynamic_field=True,  # Поддержка metadata
                # auto_id=True,
                vector_field=["dense", "sparse"],
                index_params= [dense_index_param, sparse_index_param],
            )

            logger.info(f"Векторная база данных настроена с {len(processed_docs)} документами")

        except Exception as e:
            logger.error(f"Ошибка при настройке векторной базы: {e}")
            raise


    def create_retriever(self, k: int = 7, fetch_k: int = 20) -> None:
        """
        Создание ретривера для поиска
        """
        if not self.vectorstore:
            logger.error("Векторная база данных не настроена")
            return

        base_retriever = self.vectorstore.as_retriever(
            search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "ranker_type": "weighted",  # или "rrf, weighted"
            "ranker_params": {"weights": [0.6, 0.4], "lambda": 0.5}, # lambda=0.5 — баланс между точностью и полнотой.
                # Можно поиграть с lambda от 0.1 (больше внимания точности) до 0.9 (полнота).
        })

        # MultiQuery: генерация нескольких версий запроса
        multi_query_prompt = PromptTemplate.from_template("""
        Вы — помощник поиска организации АО «ЦентрИнформ». Ваша задача — сгенерировать несколько различных формулировок одного и того же запроса на русском языке, чтобы улучшить поиск в базе знаний.
        Оригинальный вопрос: {question}

        Создайте 3 альтернативные формулировки этого вопроса, каждую с новой строки.
        Семантика должна сохраняться. Не нумеруйте строки. Не добавляйте пояснений.
        """.strip())

        llm_with_queries = self.llm.bind(n=2)
        multiquery_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm_with_queries,
            prompt=multi_query_prompt,
            include_original=True,  # включаем оригинал
        )
        self.retriever = multiquery_retriever

        # self.retriever = base_retriever

        logger.info("Ретривер создан")


    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """Получение истории чата для сессии"""
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = InMemoryChatMessageHistory()
        return self.chat_histories[session_id]


    def clear_session(self, session_id: str):
        self.chat_histories.pop(session_id, None)


    def get_chat_history(self, session_id: str) -> List[dict]:
        history = self.get_session_history(session_id)
        return [
            {"role": "user" if isinstance(m, HumanMessage) else "ai", "content": m.content}
            for m in history.messages
        ]


    def setup_qa_chain(self) -> None:
        """
        Настройка цепочки вопрос-ответ
        """
        if not self.retriever:
            logger.error("Ретривер не создан")
            return

        # Продвинутый промпт для точных ответов
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Вы — экспертный ассистент организации АО «ЦентрИнформ». Отвечайте ТОЛЬКО на основе контекста.\n"
             "Контекст:\n{context}\n\n"
             "История диалога:\n{chat_history}\n\n"
             "Текущий вопрос: {question}\n\n"
             "Правила:\n"
             "1. Используйте ТОЛЬКО контекст и историю.\n"
             "2. Отвечайте чётко, на русском языке.\n"
             "3. Если информации нет — скажите: 'Информация не найдена'.\n"
             "4. Не добавляйте предположений.\n"
             "Ответ:"),
        ])

        # def format_docs(docs):
        #     return "\n\n".join(doc.page_content.strip() for doc in docs)

        def format_docs(docs: List[Document], max_docs: int = 10) -> str:
            return "\n".join(
                f"[Источник: {doc.metadata.get('source', 'N/A')}] {doc.page_content.strip()}"
                for doc in docs[:max_docs]
            )

        chain = (
                {
                    "context": lambda x: format_docs(self.retriever.get_relevant_documents(x["question"])),
                    "question": lambda x: x["question"],
                    "chat_history": lambda x: x.get("chat_history", ""),
                }
                | prompt
                | self.llm
                | StrOutputParser()
        )

        # Обёртка с поддержкой истории
        self.qa_chain_with_history = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=self.get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

        logger.info("Цепочка QA настроена")


    def search_similar_documents(self, query: str, k: int = 5, fetch_k = 20) -> List[Dict]:
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


    def ask_question_rag(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Ответ на вопрос с использованием RAG и истории чата

        Args:
            question: Текст вопроса
            session_id: Уникальный ID сессии (для поддержки нескольких пользователей)

        Returns:
            Словарь с ответом, источниками и историей
        """
        if not self.qa_chain_with_history:
            return {"error": "Цепочка не настроена", "question": question}

        try:
            # Получаем релевантные документы
            relevant_docs = self.retriever.get_relevant_documents(question)

            # Генерируем ответ с историей
            response = self.qa_chain_with_history.invoke(
                {"question": question},
                config={"configurable": {"session_id": session_id}}
            )

            # Форматируем источники
            sources = []
            for doc in relevant_docs:
                sources.append({
                    "source": doc.metadata.get("source", "Неизвестный источник"),
                    "content": doc.page_content.strip(),
                    "score": getattr(doc, "score", None)
                })

            return {
                "question": question,
                "answer": response.strip(),
                "sources": sources[:3],
                "session_id": session_id,
            }

        except Exception as e:
            logger.error(f"Ошибка при ответе на вопрос (RAG): {e}")
            return {"error": str(e), "question": question, "session_id": session_id}