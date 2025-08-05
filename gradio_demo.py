import gradio as gr
from classes.rag_core import RAGCore
import asyncio

# Применяем nest_asyncio, так как Gradio запускается в Jupyter/встроенный event loop
# nest_asyncio.apply()

# Инициализация RAGCore
rag = RAGCore(
    llm_base_url="http://172.20.4.50:8001/v1",
    embedding_base_url="http://172.20.4.50:8000/v1",
    collection_name="demo_ci_rag"
)


# Загрузка документов и настройка системы
def initialize_rag():
    print("Загрузка документов...")
    docs = rag.load_documents_from_directory()
    if docs:
        print("Настройка векторной БД...")
        rag.setup_vectorstore(docs)
        rag.create_retriever()
        rag.setup_qa_chain()
        print("RAG система готова.")
    else:
        print("Документы не найдены. Проверьте папку scraped_data.")


# Запускаем инициализацию
initialize_rag()


# Асинхронная функция с стримингом
async def rag_stream(question: str, history: list):
    session_id = "gradio_demo_session"  # Можно использовать hash от пользователя

    # Асинхронный поток генерации
    full_response = ""
    async for chunk in rag.qa_chain_with_history.astream(
            {"question": question},
            config={"configurable": {"session_id": session_id}}
    ):
        full_response += chunk
        # Постепенно отправляем ответ
        yield full_response

    # Дополнительно можно добавить конец
    # yield full_response


# Gradio ChatInterface
demo = gr.ChatInterface(
    fn=rag_stream,
    type="messages",  # Поддержка истории
    chatbot=gr.Chatbot(
        height=600,
        # avatar_images=["👤", "🤖"],  # Иконки
        show_copy_button=False,
        type="messages",
    ),
    textbox=gr.Textbox(
        placeholder="Введите ваш вопрос...",
        container=False,
        scale=7,
    ),

    title="RAG Ассистент — АО «ЦентрИнформ»",
    description="""
    <center>
        <b>Демо-версия RAG-системы</b><br>
        Основана на <code>Milvus + vLLM + LangChain</code><br>
        Поддержка стриминга ответов и истории.
    </center>
    """,
    examples=[
        "Что такое АО «ЦентрИнформ»?",
        "Какие продукты предоставляет компания?",
        "Опишите процесс обработки данных",
    ],
    theme="soft",
)

# Запуск
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Для локального доступа
        server_port=7860,
        share=False,  # True — если хотите публичную ссылку
        debug=True,
    )