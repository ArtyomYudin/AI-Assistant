import asyncio
import hashlib
import os
import pathlib
import time

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.summarize.refine_prompts import prompt_template
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain_core.messages import trim_messages, ChatMessage, HumanMessage, AIMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import MarkdownHeaderTextSplitter, MarkdownTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM

from pydantic import BaseModel, field_validator


class ChatMessage(BaseModel):
    question: str = ""

    @field_validator("question", mode="before")
    def truncate_question(cls, v):  # noqa
        if isinstance(v, str):
            return v[:128]
        return v

global_unique_hashes = set()
root = pathlib.Path(__file__).parent.parent.resolve()
CHROMA_DB_PATH = f"{root}/chroma_db"
DATA_PATH = f"{root}/scraped_data"
EMBEDDINGS = OllamaEmbeddings(base_url="http://172.21.92.99:11434", model="mxbai-embed-large")
#LLM_LOCAL = OllamaLLM(base_url="http://172.21.92.99:11434", model="qwen2.5-coder:14b")
LLM_LOCAL = OllamaLLM(base_url="http://172.21.92.99:11434", model="llama3.2:latest", temperature=0.1)

chroma_db_store = Chroma(persist_directory=CHROMA_DB_PATH,
                             embedding_function=EMBEDDINGS,
                             collection_name="Chroma-RAG")
chat_history = {}  # approach with AiMessage/HumanMessage

prompt_template = ChatPromptTemplate(
        [(
                "system",
                """
                    [INST]Ready for an online meeting?[/INST]
                    [INST]Answer the question based only on the following context:
                    {context}[/INST]
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ]
    )
# cached_chain = prompt_template | model
document_chain = create_stuff_documents_chain(llm=LLM_LOCAL, prompt=prompt_template)


# Функция принимает строковое значение, кодирует его в байты и вычисляет SHA-256 хэш.
# Хэш будет необходим для сравнения и выявления повторяющейся информации.
def hash_text(text):
    hash_object = hashlib.sha256(text.encode())
    return hash_object.hexdigest()

# Генерация списка файлов, соответствующих заданному расширению
def get_all_files_by_extension(path: str, file_extension='.md'):
    for (dir_path, dir_names, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith(file_extension):
                yield os.path.join(dir_path, filename)


# Загрузка документов
def load_documents(data_path: str, file_extension=".md") -> list:
    documents = []
    files_list = get_all_files_by_extension(data_path, file_extension)
    for f_name in files_list:
        document_loader = TextLoader(f_name, encoding="utf-8")
        documents.extend(document_loader.load())

    return documents


# Разделение содержимого документов небольшие фрагменты (chunk)
# text_splitter не задействован, используется разделение по заголовкам (header)

def split_text(documents: list[Document]):
    text_splitter = MarkdownTextSplitter(
        chunk_size=380,  # Размер каждого фрагмента в символах
        chunk_overlap=80,  # Перекрытие между последовательными фрагментами
        length_function=len,  # Функция для вычисления длины текста
    )

    headers = [("#", "Header 1"),
               ("##", "Header 2"),
               ("###", "Header 3")]

    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers, strip_headers=False)

    chunks = []
    for doc in documents:
        parsed_chunks = md_splitter.split_text(doc.page_content)
        for chunk in parsed_chunks:
            chunk.metadata['source'] = doc.metadata['source']
        chunks.extend(parsed_chunks)

    # Разделение документов на более мелкие части с помощью текстового разделителя
    #chunks = text_splitter.split_documents(documents)
    print(f"Разделено {len(documents)} документов на {len(chunks)} фрагментов.")

    # Удаление дубликатов, на основе хэш
    unique_chunks = []
    for chunk in chunks:
        chunk_hash = hash_text(chunk.page_content)
        if chunk_hash not in global_unique_hashes:
            unique_chunks.append(chunk)
            global_unique_hashes.add(chunk_hash)

    print(f"Количество Уникальных фрагментов {len(unique_chunks)}.")
    return unique_chunks  # Return the list of split text chunks


def delete_collection():
    chroma_db_store = Chroma(persist_directory=CHROMA_DB_PATH,
                             embedding_function=EMBEDDINGS,
                             collection_name="Chroma-RAG")
    chroma_db_store.delete_collection()


# Сохранение подготовленных фрагментов документов в ChromaDB
def generate_db(chunks: list[Document]):
    # Создайте новую базу данных Chroma из документов
    Chroma.from_documents(
        documents=chunks,
        collection_name="Chroma-RAG",
        embedding=EMBEDDINGS,
        persist_directory=CHROMA_DB_PATH
    )

    print(f"Записано {len(chunks)} фрагментов в {CHROMA_DB_PATH}.")

async def rag(message: ChatMessage, session_id: str = ""):
    if session_id not in chat_history:
        chat_history[session_id] = []

    messages = trim_messages(chat_history[session_id], strategy="last", token_counter=count_tokens_approximately,
                             max_tokens=2056, start_on="human", allow_partial=False)


    response_text = await document_chain.ainvoke({"context": chroma_db_store.similarity_search(message.question, k=3),
                                                  "question": message.question,
                                                  "chat_history": messages})

    chat_history[session_id].append(HumanMessage(content=message.question))
    chat_history[session_id].append(AIMessage(content=response_text))

    return response_text

async def run_rag_test():

    while True:
        user_input = input(
            "Ask a query about your company (or type 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        elif len(user_input) < 1:
            continue

        response = await rag(ChatMessage(question=user_input))
        print("Response: \n\n" + response)


if __name__ == "__main__":
    #tic = time.perf_counter()
    #ls = load_documents(DATA_PATH, ".md")
    #chunks=split_text(ls)
    #delete_collection()
    #generate_db(chunks)
    #print(f"Время выполнения {(time.perf_counter() - tic):.2f} сек.")

    asyncio.run(run_rag_test())