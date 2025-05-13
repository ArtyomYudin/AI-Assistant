import pathlib

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings

root = pathlib.Path(__file__).parent.parent.resolve()

CHROMA_DB_PATH = f"{root}/chroma_db"
LLM_LOCAL = OllamaLLM(base_url="http://172.21.92.99:11434", model="llama3.2:latest", temperature=0.1)
EMBEDDINGS = OllamaEmbeddings(base_url="http://172.21.92.99:11434", model="mxbai-embed-large")
COLLECTION_NAME = "demo_ci_rag"


# prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
# 1. If the question is to request links, please only return the source links with no answer.
# 2. If you don't know the answer, don't try to make up an answer. Just say **I can't find the final answer**.
# 3. If you find the answer, write the answer in a concise way.
#
# {context}
#
# Question: {question}
# Helpful Answer:"""

chat_history = {}  # approach with AiMessage/HumanMessage

vector_store = Chroma(persist_directory=CHROMA_DB_PATH,
                   embedding_function=EMBEDDINGS,
                   collection_name=COLLECTION_NAME
                   )


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                [INST]Ты — внутренний помощник ИТ отдела по имени ИИ-Бот. Отвечаешь по делу без лишних вступлений.
                Используй обычный текст без форматирования.
                Включай ссылки только если они есть в контексте.
                Говори от первого лица множественного числа: "Мы предоставляем", "У нас есть".
                На приветствия отвечай доброжелательно, на негатив — с легким юмором.
                При технических вопросах предлагай практические решения.[/INST]
                [INST]Отвечай на вопрос, основываясь только на следующем контексте:
                {context}[/INST]
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ]
)

# cached_chain = prompt_template | model
document_chain = create_stuff_documents_chain(llm=LLM_LOCAL, prompt=prompt_template)
session_id = ""

if session_id not in chat_history:
        chat_history[session_id] = []

messages = trim_messages(chat_history[session_id], strategy="last", token_counter=count_tokens_approximately,
                             max_tokens=2056, start_on="human", allow_partial=False)


response_text = document_chain.invoke({"context": vector_store.similarity_search("python palindrome", k=3),
                                                  "question": "python palindrome",
                                                  "chat_history": messages})

print(response_text)


