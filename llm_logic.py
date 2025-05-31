from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

import config
from prompts import CHARACTER_NAME, CHARACTER_CORE_PERSONALITY
from vector_store import initialize_retriever

LLM_NAME = "deepseek/deepseek-chat:free"

retriever = initialize_retriever()

print("Инициализация LLM")
llm = ChatOpenAI(
    base_url=config.LLM_BASE_URL,
    api_key=config.LLM_API_KEY,
    model_name=LLM_NAME,
    temperature=0.7,
    streaming=True,
)


class AgentState(TypedDict):
    user_input: str
    retrieved_character_context: str
    retrieved_world_context: str
    generation: str


def retrieve_context_node(state: AgentState):
    print("\n--- УЗЕЛ: ИЗВЛЕЧЕНИЕ КОНТЕКСТА ---")
    user_input = state["user_input"]

    character_docs_retrieved = retriever.invoke(
        f"Информация о персонаже {CHARACTER_NAME}, связанная с: {user_input}",
    )
    char_context_str = "\n---\n".join(
        [
            doc.page_content
            for doc in character_docs_retrieved
            if doc.metadata.get("source") == "character_story"
        ]
    )

    world_docs_retrieved = retriever.invoke(
        f"Информация из истории мира, связанная с: {user_input}"
    )
    world_context_str = "\n---\n".join(
        [
            doc.page_content
            for doc in world_docs_retrieved
            if doc.metadata.get("source") == "world_history"
        ]
    )

    print(f"Извлеченный контекст (персонаж): {char_context_str[:300]}...")
    print(f"Извлеченный контекст (мир): {world_context_str[:300]}...")

    return {
        "retrieved_character_context": char_context_str,
        "retrieved_world_context": world_context_str,
    }


def generate_response_node(state: AgentState):
    print("\n--- УЗЕЛ: ГЕНЕРАЦИЯ ОТВЕТА ---")
    user_input = state["user_input"]
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
            ("human", "{user_input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    full_response = chain.invoke({"user_input": user_input})

    print(f"{CHARACTER_NAME}: ", end="", flush=True)
    print(full_response)
    return {"generation": full_response}


workflow = StateGraph(AgentState)
workflow.add_node("retrieve_context", retrieve_context_node)
workflow.add_node("generate_response", generate_response_node)
workflow.set_entry_point("retrieve_context")
workflow.add_edge("retrieve_context", "generate_response")
workflow.add_edge("generate_response", END)

app = workflow.compile()
print("Граф LangGraph скомпилирован.")
