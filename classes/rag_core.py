import hashlib
import logging
import pathlib
import re

from typing import List, Dict, Any

from langchain.retrievers import MultiQueryRetriever
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_milvus import Milvus
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

root = pathlib.Path(__file__).parent.parent.resolve()

class RAGCore:
    def __init__(self,
                 llm_name: str = "Qwen3-8B-AWQ",
                 llm_base_url: str = "http://172.20.4.50:8001/v1",
                 embedding_name : str = "Qwen3-Embedding-4B",
                 embedding_base_url: str ="http://172.20.4.50:8000/v1",
                 milvus_base_uri: str = f"{root}/milvus_db/test.db",
                 collection_name: str = "demo_ci_rag",
                 ):

        self.collection_name = collection_name
        self.milvus_connection_args = {
            "uri": milvus_base_uri
        }

        # Инициализация эмбеддингов
        self.embeddings = OpenAIEmbeddings(
            model=embedding_name,
            api_key="EMPTY",
            base_url=embedding_base_url,
            # embedding_ctx_length=2048,
        )

        # Инициализация LLM
        self.llm = ChatOpenAI(
            model=llm_name,
            api_key="EMPTY",
            base_url=llm_base_url,
            extra_body={"chat_template_kwargs": {"enable_thinking":False}},
            max_tokens=512,
            temperature=0.9,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )

        # Рекурсивное разбиение с учетом структуры MD
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=[
                "\n#{1,6} ",  # Заголовки
                "```\n",  # Блоки кода
                "\n\n",  # Параграфы
                "\n",  # Строки
                ". ",  # Предложения
                " ",  # Пробелы
                ""  # Символы
            ],
            length_function=len,
            is_separator_regex=False,
        )

        # Инициализация сплиттера по заголовкам
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header1"),
                 ("##", "Header2"),
                 ("###", "Header3"),
                 ("####", "Header4"),
            ], strip_headers=False
        )

        # Инициализация векторной базы данных
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None


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


    def clean_md_content(self, content: str) -> str:
        """
        Очистка MD-контента от лишних элементов
        """
        # Удаление HTML комментариев
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

        # Удаление ссылок в формате [text](url) оставляя текст
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)

        # Удаление изображений
        content = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', content)

        # Нормализация пробелов
        # content = re.sub(r'\s+', ' ', content)

        # Удаление множественных пустых строк
        content = re.sub(r'\n\s*\n', '\n\n', content)

        return content.strip()


    def split_by_headers(self, load_documents: List[Document]) -> List[Document]:
        """
        Разбиение по заголовкам с сохранением иерархии
        """
        chunks = []

        try:
            for doc in load_documents:
                # очистка документа
                clear_content = self.clean_md_content(doc.page_content)
                # Сначала разбиваем по заголовкам
                header_docs = self.header_splitter.split_text(clear_content)
                # Затем каждый раздел дополнительно разбиваем если нужно
                final_docs = []
                for header_doc in header_docs:
                    if len(header_doc.page_content) > self.recursive_splitter._chunk_size:
                        # Если раздел большой - дополнительно разбиваем
                        sub_docs = self.recursive_splitter.split_documents([header_doc])
                        chunks.extend(sub_docs)
                    else:
                        chunks.append(header_doc)

            return chunks

        except Exception as e:
            logger.error(f"Ошибка при разбиении по заголовкам: {e}")
            # fallback на рекурсивное разбиение
            return self.recursive_splitter.split_documents(load_documents)


    def setup_vectorstore(self, documents: List[Document]) -> None:
        """
        Настройка векторной базы данных Milvus
        """
        processed_docs = self.split_by_headers(documents)

        if not processed_docs:
            logger.warning("Нет документов для индексации")
            return

        try:
            # Создание или подключение к коллекции Milvus
            self.vectorstore = Milvus.from_documents(
                documents=processed_docs,
                embedding=self.embeddings,
                connection_args=self.milvus_connection_args,
                collection_name=self.collection_name,
                drop_old=True,  # Удалять существующую коллекцию
                # consistency_level="Eventually",
                 index_params={
                     "metric_type": "COSINE",
                #     "index_type": "AUTOINDEX",
                     "params": {}}
            )

            logger.info(f"Векторная база данных настроена с {len(processed_docs)} документами")

        except Exception as e:
            logger.error(f"Ошибка при настройке векторной базы: {e}")
            raise


    def create_retriever(self, search_kwargs: Dict[str, Any] = None) -> None:
        """
        Создание ретривера для поиска
        """
        if not self.vectorstore:
            logger.error("Векторная база данных не настроена")
            return

        if search_kwargs is None:
            search_kwargs = {
                "k": 3,  # Количество результатов
                "score_threshold": 0.3  # Порог релевантности
                #"fetch_k": 50
            }

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

        logger.info("Ретривер создан")


    def setup_qa_chain(self) -> None:
        """
        Настройка цепочки вопрос-ответ
        """
        if not self.retriever:
            logger.error("Ретривер не создан")
            return

        # Продвинутый промпт для точных ответов
        template = """Вы - экспертный помощник по документам. Отвечайте ТОЛЬКО на основе следующего контекста:

                    Контекст:
                    {context}
            
                    Вопрос: {question}
            
                    Требования к ответу:
                    1. Используйте ТОЛЬКО информацию из контекста выше
                    2. Отвечайте четко и по существу на русском языке
                    3. Если информации нет - скажите "Информация не найдена в предоставленных документах"
                    4. Не добавляйте предположений или информацию вне контекста
                    5. Сохраняйте точность и объективность
            
                    Ответ:"""

        prompt = ChatPromptTemplate.from_template(template)

        # def format_docs(docs):
        #     """Форматирование документов для контекста"""
        #     formatted_docs = []
        #     for doc in docs:
        #         source = doc.metadata.get('source', 'Неизвестный источник')
        #         content = doc.page_content.strip()
        #         formatted_docs.append(f"Источник: {source}\nСодержание: {content}")
        #     return "\n\n".join(formatted_docs)
        def format_docs(docs):
            return "\n\n".join(doc.page_content.strip() for doc in docs)

        # Создание QA цепочки
        self.qa_chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        logger.info("Цепочка QA настроена")

    def search_similar_documents(self, query: str, k: int = 5) -> List[Dict]:
        """
        Поиск похожих документов
        """
        if not self.vectorstore:
            return []

        try:
            docs = self.vectorstore.similarity_search_with_score(query, k=k)
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


    def ask_question_rag(self, question: str) -> Dict[str, Any]:
        """
        Ответ на вопрос с использованием RAG цепочки
        """
        if not self.qa_chain:
            return {
                "question": question,
                "answer": "Система не настроена",
                "sources": [],
                "error": "RAG цепочка не настроена"
            }

        try:
            # Получение релевантных документов
            # relevant_docs = self.retriever.get_relevant_documents(question)

            # Генерация ответа
            answer = self.qa_chain.invoke(question)

            # # Подготовка источников
            # sources = []
            # for doc in relevant_docs:
            #     sources.append({
            #         "source": doc.metadata.get("source", "Неизвестный источник"),
            #         "content": doc.page_content,
            #         "score": doc.metadata.get("score", 0) if "score" in doc.metadata else 0
            #     })
            #
            return {
                "question": question,
                "answer": answer.strip(),
                #"sources": sources[:3]  # Ограничиваем 3 источниками
            }

        except Exception as e:
            logger.error(f"Ошибка при ответе на вопрос (RAG): {e}")
            return {
                "question": question,
                "answer": "Ошибка при обработке запроса",
                "sources": [],
                "error": str(e)
            }