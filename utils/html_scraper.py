import pathlib
#import os
import time

#os.environ['USER_AGENT'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer, BeautifulSoupTransformer

root = pathlib.Path(__file__).parent.parent.resolve()

FILE_TO_PARSE = f"{root}/source_data/links.txt"
DIR_TO_STORE = f"{root}/scraped_data"

# Получение списка ссылок, предварительно удалив лишние пробелы
def get_links_to_parse() -> list:
    try:
        with open(FILE_TO_PARSE, "r") as f:
            return [link.strip() for link in f.readlines()]
    except:
        return []

# Асинхронная загрузка ссылок и трансформация в чистый ASCII (MarkDown)
def async_loader(links):
    loader = AsyncHtmlLoader(links)
    docs = loader.load()

    # Получение чистого HTML без лишних классов
    bs_transformer = BeautifulSoupTransformer()
    for doc in docs:
        doc.page_content = bs_transformer.remove_unwanted_classnames(doc.page_content,
                                                                     ['top-column', 'breadcrumb-navigation',
                                                                      'dv-copy', 'dv-contact-inc', 'dv-botmenu'])

    # Преобразование в ASCII (MarkDown)
    html_to_text = Html2TextTransformer(ignore_links=True, ignore_images=True)
    docs_transformed = html_to_text.transform_documents(docs)

    # Сохранение преобразованных документов в файлы
    for idx, doc in enumerate(docs_transformed):
        with open(f"{DIR_TO_STORE}/document_{idx}.md", "w+", encoding="utf-8") as f:
            f.write(doc.page_content)
            print(f"Файл {DIR_TO_STORE}/document_{idx}.md сохранен")



if __name__ == "__main__":
    tic = time.perf_counter()
    ls = get_links_to_parse()
    async_loader(ls)
    print(f'Список из {len(ls)} ссылок обработан за {(time.perf_counter() - tic):.2f} сек.')