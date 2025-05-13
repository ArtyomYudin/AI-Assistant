import hashlib
import os
import pathlib
import shutil
import time

from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


global_unique_hashes = set()
root = pathlib.Path(__file__).parent.parent.resolve()

CHROMA_DB_PATH = f"{root}/chroma_db"
DATA_PATH = f"{root}/scraped_data"
EMBEDDINGS = OllamaEmbeddings(base_url="http://172.21.92.99:11434", model="mxbai-embed-large")
COLLECTION_NAME = "demo_ci_rag"


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
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=380,  # Размер каждого фрагмента в символах
    #     chunk_overlap=80,  # Перекрытие между последовательными фрагментами
    #     length_function=len,  # Функция для вычисления длины текста
    #     add_start_index=True,  # Флаг для добавления начального индекса к каждому фрагменту
    # )
    text_splitter = MarkdownTextSplitter(
        chunk_size=380,  # Размер каждого фрагмента в символах
        chunk_overlap=80,  # Перекрытие между последовательными фрагментами
        length_function=len,  # Функция для вычисления длины текста
    )

    # headers = [("#", "Header 1"),
    #            ("##", "Header 2"),
    #            ("###", "Header 3")]
    #
    # md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers, strip_headers=False)
    #
    # chunks = []
    # for doc in documents:
    #     parsed_chunks = md_splitter.split_text(doc.page_content)
    #     for chunk in parsed_chunks:
    #         chunk.metadata['source'] = doc.metadata['source']
    #     chunks.extend(parsed_chunks)

    # Разделение документов на более мелкие части с помощью текстового разделителя
    chunks = text_splitter.split_documents(documents)
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


# Сохранение подготовленных фрагментов документов в ChromaDB
def generate_db(chunks: list[Document]):
    # Очищаем существующий каталог базы данных, если он есть
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)

    # Создайте новую базу данных Chroma из документов
    vector_store = Chroma.from_documents(
        documents=chunks,
        persist_directory=CHROMA_DB_PATH,
        embedding=EMBEDDINGS,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"Записано {len(chunks)} фрагментов в {CHROMA_DB_PATH}.")


if __name__ == "__main__":
    tic = time.perf_counter()
    ls = load_documents(DATA_PATH, ".md")
    chunks=split_text(ls)
    generate_db(chunks)
    print(f"Время выполнения {(time.perf_counter() - tic):.2f} сек.")