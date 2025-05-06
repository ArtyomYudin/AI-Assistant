import hashlib
import os
import pathlib
import time

from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, MarkdownTextSplitter
from langchain.vectorstores import Chroma

root = pathlib.Path(__file__).parent.parent.resolve()
CHROMA_DB_PATH = f"{root}/chroma_db"
DATA_PATH = f"{root}/md_files"
global_unique_hashes = set()


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
    files_list = get_all_files_by_extension(DATA_PATH, file_extension)
    for f_name in files_list:
        document_loader = TextLoader(f_name, encoding="utf-8")
        documents.extend(document_loader.load())

    return documents


# Разделение содержимого документов небольшие фрагменты (chunk)
def split_text(documents: list[Document]):
    text_splitter = MarkdownTextSplitter(
        chunk_size=500,  # Размер каждого фрагмента в символах
        chunk_overlap=100,  # Перекрытие между последовательными фрагментами
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
    # Создайте новую базу данных Chroma из документов
    Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(base_url="http://172.21.92.99:11434", model="mxbai-embed-large"),
        persist_directory=CHROMA_DB_PATH
    )

    print(f"Saved {len(chunks)} chunks to {CHROMA_DB_PATH}.")

if __name__ == "__main__":
    tic = time.perf_counter()
    ls = load_documents(DATA_PATH, ".md")
    chunks=split_text(ls)
    generate_db(chunks)
