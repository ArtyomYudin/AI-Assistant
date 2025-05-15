import pathlib
import os
import pprint
import time
import requests

os.environ['USER_AGENT'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"

from markdownify import markdownify as md, MarkdownConverter
from urllib.parse import urlparse
from bs4 import BeautifulSoup, SoupStrainer
from langchain_community.document_loaders import AsyncHtmlLoader, WebBaseLoader
from langchain_community.document_transformers import Html2TextTransformer, BeautifulSoupTransformer

root = pathlib.Path(__file__).parent.parent.resolve()

FILE_TO_PARSE = f"{root}/source_data/links.txt"
DIR_TO_STORE = f"{root}/scraped_data"

# Исследуемый домен
DOMAIN = "center-inform.ru"
HOST = "https://" + DOMAIN
# Список префиксов для исключения ссылок по типу tel:, mainto: и т.д
EXCLUDED_PREFIX = ["#", "tel:", "mailto:"]
# множество всех внутренних ссылок сайта
internal_links = set()

def get_all_internal_links(url: str, max_depth=1):
    # список ссылок, от которых в конце мы рекурсивно запустимся
    links_to_handle_recursive = []
    # получаем html код страницы
    request = requests.get(url)
    # парсим его с помощью BeautifulSoup
    bf_soup = BeautifulSoup(request.content, "lxml")
    for tag_a in bf_soup.find_all('a'):
        #link = tag_a["href"]
        internal_link= tag_a.get("href", "")
        # исключаем ненужные ссылки
        # так как ссылка может быть вида /about/awards/#382
        # отделяем последнюю составляющую пути  internal_link.split('/')[-1]
        # в результате получаем #382 и проверяем ее на префиксы для исключения
        if all(not internal_link.split('/')[-1].startswith(prefix) for prefix in EXCLUDED_PREFIX):
            # проверяем относительная ли абсолютная ссылка
            # /about - относительная ссылка
            # https://domain.com/about - абсолютная ссылка
            if internal_link.startswith("/") and not internal_link.startswith("//"):
                # преобразуем относительную ссылку в абсолютную
                internal_link = HOST + internal_link
            # проверка ссылки на соответствие домену и добавлена ли она уже в список ссылок
            if urlparse(internal_link).netloc == DOMAIN and internal_link not in internal_links:
                internal_links.add(internal_link)
                links_to_handle_recursive.append(internal_link)
    if max_depth > 0:
        for internal_link in links_to_handle_recursive:
            get_all_internal_links(internal_link, max_depth=max_depth - 1)


# Получение списка ссылок, предварительно удалив лишние пробелы
# def get_links_to_parse() -> list:
#     try:
#         with open(FILE_TO_PARSE, "r") as f:
#             return [link.strip() for link in f.readlines()]
#     except:
#         return []

# Асинхронная загрузка ссылок и трансформация в чистый ASCII (MarkDown)
def async_loader(links):
    # TODO: необходимо избавиться от тегов <ul>
    loader = AsyncHtmlLoader(links)
    docs = loader.load()

    # Получение чистого HTML без лишних классов
    bs_transformer = BeautifulSoupTransformer()
    for doc in docs:
        doc.page_content = bs_transformer.remove_unwanted_classnames(
            doc.page_content,
            ["top-column", "breadcrumb-navigation",
             "dv-copy", "dv-contact-inc", "dv-botmenu"])

        doc.page_content = bs_transformer.remove_unwanted_tags(
            doc.page_content,
            unwanted_tags=["head", "script", "noscript", "ul"])

        #doc.page_content = bs_transformer.remove_unnecessary_lines(doc.page_content)

        #soup = BeautifulSoup(doc.page_content, 'lxml')
        # removals = soup.find_all('div', {'id': 'footer'})
        # for match in removals:
        #     match.decompose()


    #print(docs[0])
    # Преобразование в ASCII (MarkDown)
    html_to_text = Html2TextTransformer(ignore_links=True, ignore_images=True)
    docs_transformed = html_to_text.transform_documents(docs)


    # Сохранение преобразованных документов в файлы
    for idx, doc in enumerate(docs_transformed):
        with open(f"{DIR_TO_STORE}/document_{idx}.md", "w+", encoding="utf-8") as f:
            f.write(doc.page_content)
            print(f"Файл {DIR_TO_STORE}/document_{idx}.md сохранен")



if __name__ == "__main__":
    get_all_internal_links(HOST, max_depth=0)
    tic = time.perf_counter()
    pprint.pprint(internal_links)
    async_loader(list(internal_links))
    print(f'Список из {len(internal_links)} ссылок обработан за {(time.perf_counter() - tic):.2f} сек.')