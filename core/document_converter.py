import time
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions #, DOCLING_DEFAULT_OPTIONS

# --- Настраиваем пайплайн ---
pdf_pipeline_options = PdfPipelineOptions()

pdf_pipeline_options.do_ocr = False # Docling Parse without EasyOCR

pdf_pipeline_options.generate_page_images = True   # извлекаем изображения (сохраняются рядом с md)
# pdf_pipeline_options.generate_preview_images = False

pdf_pipeline_options.do_table_structure = True # извлечение таблиц
pdf_pipeline_options.table_structure_options.do_cell_matching = True  # использует текстовые ячейки, предсказанные на основе модели структуры таблицы

# --- Конвертер ---
conv = DocumentConverter(
    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
    }
    # pipeline_options=DOCLING_DEFAULT_OPTIONS
)

def file_to_markdown(path: str) -> str:
    """
    Конвертирует PDF/DOCX в Markdown с помощью Docling.
    Поддерживает извлечение изображений из PDF.
    """
    doc_path = Path(path)
    if not doc_path.exists():
        raise FileNotFoundError(f"Файл {path} не найден")

    # Конвертация
    result = conv.convert(doc_path)

    # Экспорт в Markdown
    md_content = result.document.export_to_markdown()

    return md_content


# --- Пример использования ---
if __name__ == "__main__":
    start_time = time.time()
    md_text = file_to_markdown("../scraped_data/instructions/polygon.pdf_bak")
    print(md_text[:1000])  # печатаем первые 1000 символов Markdown
    end_time = time.time() - start_time
    print(f"Document converted in {end_time:.2f} seconds.")