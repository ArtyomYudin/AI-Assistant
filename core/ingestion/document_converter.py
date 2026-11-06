# core/ngestion/document_converter.py

# Copyright 2025 Artyom Yudin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions #, DOCLING_DEFAULT_OPTIONS
from docling_core.types.doc.base import ImageRefMode


def create_universal_converter() -> DocumentConverter:
    """
    Создает и возвращает настроенный DocumentConverter,
    который умеет обрабатывать и PDF, и DOCX.
    """
    # --- Настройки для PDF ---
    pdf_pipeline_options = PdfPipelineOptions()
    pdf_pipeline_options.do_ocr = False # Docling Parse without EasyOCR
    pdf_pipeline_options.generate_page_images = True
    pdf_pipeline_options.do_table_structure = True
    pdf_pipeline_options.table_structure_options.do_cell_matching = True

    # --- Инициализация конвертера с опциями для обоих форматов ---
    conv = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_pipeline_options
            ),
            InputFormat.DOCX: WordFormatOption()
            # Для DOCX пока нет специфичных опций пайплайна,
            # но явное указание гарантирует использование нужного бэкенда.
        }
    )
    return conv


def universal_file_to_markdown(
        path: str, output_dir: str = "./converted_output"
) -> str:
    """
    Конвертирует PDF или DOCX в Markdown с сохранением изображений
    как отдельных файлов и ссылками на них в тексте.
    """
    doc_path = Path(path)
    if not doc_path.exists():
        raise FileNotFoundError(f"Файл {path} не найден")

    # Определяем формат по расширению (на всякий случай, хотя Docling умеет сам)
    suffix = doc_path.suffix.lower()
    if suffix not in (".pdf", ".docx"):
        raise ValueError(f"Неподдерживаемый формат файла: {suffix}. Ожидается .pdf или .docx")

    # Создаем выходную директорию, если её нет
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Инициализируем универсальный конвертер
    conv = create_universal_converter()

    # Конвертируем документ
    result = conv.convert(doc_path)

    # Сохраняем Markdown с изображениями в режиме REFERENCED.
    # save_as_markdown сам создаст подпапку (обычно 'artifacts') для картинок
    # и пропишет ссылки на них в итоговом файле.
    md_filename = out_dir / f"{doc_path.stem}.md"
    result.document.save_as_markdown(
        md_filename,
        image_mode=ImageRefMode.REFERENCED,
        include_annotations=True
    )

    print(f"Готово. Markdown сохранен в: {md_filename}")
    print(f"Изображения (если есть) находятся в подпапке рядом с файлом .md")

    # Возвращаем содержимое для удобства (опционально)
    return md_filename.read_text(encoding="utf-8")


if __name__ == "__main__":
    start_time = time.time()

    # --- Примеры использования ---
    # можно вызывать функцию для любого поддерживаемого файла
    try:
        # Пример для PDF
        # markdown_text = universal_file_to_markdown(
        #     "path/to/your/document.pdf",
        #     output_dir="./converted_output"
        # )

        # Пример для DOCX
        markdown_text = universal_file_to_markdown(
            "source_data/outlook_test.docx",
            output_dir="./converted_output"
        )

        print("\n--- Первые 500 символов Markdown ---")
        print(markdown_text[:500])

    except Exception as e:
        print(f"Произошла ошибка: {e}")

    end_time = time.time() - start_time
    print(f"\nОбщее время выполнения: {end_time:.2f} секунд.")