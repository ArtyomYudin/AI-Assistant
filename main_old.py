from classes.rag_core import RAGCore

def main():
    # Инициализация системы
    rag_system = RAGCore()

    # Загрузка документов
    documents = rag_system.load_documents_from_directory()

    # Настройка векторной базы
    rag_system.setup_vectorstore(documents)
    rag_system.create_retriever(k=10, fetch_k=50)
    rag_system.setup_qa_chain()

    # # Поиск похожих документов
    similar_docs = rag_system.search_similar_documents("список партнеров", k=7)
    print("Похожие документы:")
    for doc in similar_docs:
        print(f"Score: {doc['score']:.3f} - {doc['content'][:100]}...")

    # Ответ на вопрос
    # answer = rag_system.ask_question_rag("покажи список директоров?")
    # print(f"\nВопрос: {answer['question']}")
    # print(f"Ответ: {answer['answer']}")
    # print("Источники:")
    # for source in answer['sources']:
    #     print(f"  - {source['content'][:100]}...")
    print("\nWelcome to Interactive Q&A System!")
    print("Enter 'q' or 'quit' to exit.")
    while True:
        question = input("\nPlease enter your question: ")
        if question.lower() in ["q", "quit"]:
            print("\nThank you for using! Goodbye!")
            break

        output =  rag_system.ask_question_rag(question, session_id="keeper")
        print(f"Ответ: {output['answer']}")

    # # Первая сессия
    # print("=== Сессия: user_001 ===")
    # rag_system.ask_question_rag("покажи название первого партнера?", session_id="user_001")
    # rag_system.ask_question_rag("и расскажи о нем?", session_id="user_001")

if __name__ == "__main__":
    main()