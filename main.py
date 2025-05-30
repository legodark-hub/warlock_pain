import os
from typing import TypedDict
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import config


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = "deepseek/deepseek-chat:free"
CHAR_INFO = "./data/character_info.txt"
WORLD_INFO = "./data/world_info.txt"


character_loader = TextLoader(CHAR_INFO, encoding="utf-8")
world_loader = TextLoader(WORLD_INFO, encoding="utf-8")
character_docs_initial = character_loader.load()
world_docs_initial = world_loader.load()

for doc in character_docs_initial:
    doc.metadata["source"] = "character_story"
for doc in world_docs_initial:
    doc.metadata["source"] = "world_history"

all_docs = character_docs_initial + world_docs_initial

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(all_docs)

# --- 2. Создание Embeddings и ChromaDB ---
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    # model_kwargs={'device': 'cuda'}, # Раскомментируйте, если хотите использовать GPU и он доступен
    # encode_kwargs={'normalize_embeddings': True} # Раскомментируйте, если модель это рекомендует
)

# Указываем директорию для сохранения базы данных Chroma
persist_directory = "./chroma_db"

# Проверяем, существует ли директория базы данных
if os.path.exists(persist_directory) and os.listdir(
    persist_directory
):  # Проверяем, что директория существует и не пуста
    vectorstore = Chroma(
        persist_directory=persist_directory, embedding_function=embeddings
    )
    print("Загружена существующая база данных ChromaDB.")
else:
    print(f"Создание новой базы данных ChromaDB в '{persist_directory}'...")
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=embeddings, persist_directory=persist_directory
    )
    # vectorstore.persist() # .from_documents с persist_directory уже должен сохранять. Если нет, раскомментируйте.

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)  # Извлекаем 4 наиболее релевантных чанка

# --- 3. Определение LLM ---
print("Инициализация LLM")
llm = ChatOpenAI(
    base_url=config.LLM_BASE_URL,
    api_key=config.LLM_API_KEY,  # API ключ из конфига или заглушка
    model_name=LLM_NAME,  # Имя модели, если требуется
    temperature=0.7,
    # streaming=True, # Можно включить для потоковой передачи
    # max_tokens=512, # Ограничение на количество токенов в ответе, если нужно
)

# --- 4. Определение имени персонажа и базовой личности (можно извлечь из файлов) ---
# Для примера, зададим имя вручную. В более сложной системе это можно автоматизировать.
CHARACTER_NAME = "Пейн"
CHARACTER_CORE_PERSONALITY = f"""
Ты — {CHARACTER_NAME}, персонаж во вселенной игры Destiny, Страж, твой класс - Варлок. 
Ты один из тех, кто стоит между последними остатками человечества и Тьмой. Ты исследователь, 
философ и солдат в одном лице. Ты обладаешь глубокими знаниями о технологиях Золотого века, 
силе Странника и тайнах Распутина. Ты сдержанный, саркастичный, хмурый, предпочитаешь 
одиночество и не спешишь открываться окружающим, но верен своим товарищам и долгу перед 
Последним городом. Твой спутник — призрак Коди, с которым ты часто обмениваешься репликами. 
Ты отлично стреляешь из снайперской винтовки, носишь бело-синюю мантию и старый револьвер, 
а твой сперроу "Белокрыл" нуждается в вечном ремонте.
Ты говоришь спокойным, ироничным и слегка усталым тоном. При этом можешь резко ответить, 
если тебе начинают перечить. В тебе сочетается научное любопытство, цинизм ветерана и магия 
Странника.
Ты не доверяешь никому с первого взгляда. Но ты помогаешь — несмотря ни на что.
Отвечай от лица Пейна, словно ты ведёшь настоящий диалог. Привествуй фразой "Света старнника и мира!".
В разговоре проявляй саркастический интеллект и скрытую теплоту. 
"""


# --- 5. LangGraph: Определение состояния и узлов ---
class AgentState(TypedDict):
    user_input: str
    # chat_history: List[BaseMessage] # Убираем историю чата из состояния
    retrieved_character_context: str
    retrieved_world_context: str
    generation: str


def retrieve_context_node(state: AgentState):
    print("\n--- УЗЕЛ: ИЗВЛЕЧЕНИЕ КОНТЕКСТА ---")
    user_input = state["user_input"]

    # Извлекаем контекст, связанный с персонажем
    character_docs_retrieved = retriever.invoke(
        f"Информация о персонаже {CHARACTER_NAME}, связанная с: {user_input}",
        # filter={"source": "character_story"} # Можно добавить фильтр, если нужно строго разделить
    )
    char_context_str = "\n---\n".join(
        [
            doc.page_content
            for doc in character_docs_retrieved
            if doc.metadata.get("source") == "character_story"
        ]
    )

    # Извлекаем контекст, связанный с миром
    world_docs_retrieved = retriever.invoke(
        f"Информация из истории мира, связанная с: {user_input}"
        # filter={"source": "world_history"}
    )
    world_context_str = "\n---\n".join(
        [
            doc.page_content
            for doc in world_docs_retrieved
            if doc.metadata.get("source") == "world_history"
        ]
    )

    # Можно объединить все релевантные документы и затем разделить их, если фильтрация не строгая
    # all_relevant_docs = retriever.get_relevant_documents(user_input)
    # char_context_str = "\n---\n".join([doc.page_content for doc in all_relevant_docs if doc.metadata.get("source") == "character_story"])
    # world_context_str = "\n---\n".join([doc.page_content for doc in all_relevant_docs if doc.metadata.get("source") == "world_history"])

    print(f"Извлеченный контекст (персонаж): {char_context_str[:300]}...")
    print(f"Извлеченный контекст (мир): {world_context_str[:300]}...")

    return {
        "retrieved_character_context": char_context_str,
        "retrieved_world_context": world_context_str,
    }


def generate_response_node(state: AgentState):
    print("\n--- УЗЕЛ: ГЕНЕРАЦИЯ ОТВЕТА ---")
    user_input = state["user_input"]
    # chat_history = state["chat_history"] if state.get("chat_history") else [] # История чата больше не используется
    char_context = state["retrieved_character_context"]
    world_context = state["retrieved_world_context"]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""{CHARACTER_CORE_PERSONALITY}

Контекст о персонаже ({CHARACTER_NAME}):
<character_context>
{char_context if char_context else "Нет специфической информации о персонаже для этого запроса."}
</character_context>

Контекст о мире:
<world_context>
{world_context if world_context else "Нет специфической информации о мире для этого запроса."}
</world_context>

Твоя задача - отвечать пользователю как {CHARACTER_NAME}, воплощая его личность, 
мотивацию и знания, основанные на предоставленном контексте и установленных 
чертах характера из ее историй.
Не выходи из роли. Не упоминай, что ты ИИ.
Будь увлекательным и последовательным в своей роли.
Если запрос пользователя касается чего-то, чего твой персонаж не знал бы или о чем 
не заботился бы, ответь так, как это соответствует твоему персонажу.
""",
            ),
            # MessagesPlaceholder(variable_name="chat_history"), # Убираем плейсхолдер для истории чата
            ("human", "{user_input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke(
        {
            "user_input": user_input,
            # "chat_history": chat_history, # История чата больше не передается в цепочку
            # Контекст уже включен в системный промпт
        }
    )
    print(f"Сгенерированный ответ LLM: {response}")
    return {"generation": response}


# --- 6. Построение графа ---
workflow = StateGraph(AgentState)

workflow.add_node("retrieve_context", retrieve_context_node)
workflow.add_node("generate_response", generate_response_node)
# workflow.add_node("update_history", update_history_node) # Удаляем узел обновления истории

workflow.set_entry_point("retrieve_context")
workflow.add_edge("retrieve_context", "generate_response")
workflow.add_edge("generate_response", END)  # Ответ сразу идет на выход

app = workflow.compile()
print("Граф LangGraph скомпилирован.")

# --- 7. Цикл чата ---
# current_chat_history = []  # Инициализируем историю чата

print(
    f"\nНачат чат с {CHARACTER_NAME} (без истории). Введите 'выход' или 'quit' для завершения."
)
while True:
    try:
        user_query = input("Вы: ")
        if user_query.lower() in ["выход", "quit"]:
            break
        if not user_query.strip():
            continue

        inputs_for_graph = {
            "user_input": user_query,
            # "chat_history": current_chat_history,  # История больше не передается
        }

        # Запускаем граф
        result_state = app.invoke(inputs_for_graph)

        ai_response = result_state.get(
            "generation", "Произошла ошибка, я не могу ответить."
        )
        # current_chat_history = result_state.get("chat_history", current_chat_history) # История больше не обновляется

        print(f"{CHARACTER_NAME}: {ai_response}")

    except KeyboardInterrupt:
        print("\nЧат прерван пользователем.")
        break
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        # Можно добавить логирование ошибки или более специфическую обработку
        break

print("Чат завершен.")
