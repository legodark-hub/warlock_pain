from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages

import config
from prompts import CHARACTER_NAME, CHARACTER_CORE_PERSONALITY
from vector_store import initialize_retriever
from loguru import logger

MAX_MESSAGES_HISTORY = 20  


retriever = initialize_retriever()

logger.info("Инициализация LLM...")
llm = ChatOpenAI(
    base_url=config.LLM_BASE_URL,
    api_key=config.LLM_API_KEY,
    model=config.LLM_NAME,
    temperature=0.7,
)
logger.info("LLM инициализирован.")


async def trim_history_node(state: MessagesState):
    logger.debug("--- УЗЕЛ: ОБРЕЗКА ИСТОРИИ ---")
    trimmed_messages = trim_messages(
        state["messages"],
        max_tokens=MAX_MESSAGES_HISTORY,
        strategy="last",
        token_counter=len,
        start_on="human",
        include_system=False,
        allow_partial=False,
    )
    logger.debug(f"История обрезана до {len(trimmed_messages)} сообщений.")
    if not trimmed_messages:
        logger.warning("После обрезки не осталось сообщений. Возвращаем исходное состояние.")
        return state
    return {"messages": trimmed_messages}

async def retrieve_and_generate_node(state: MessagesState):
    logger.debug("--- УЗЕЛ: ИЗВЛЕЧЕНИЕ КОНТЕКСТА И ГЕНЕРАЦИЯ ОТВЕТА ---")

    if not state["messages"] or not isinstance(state["messages"][-1], HumanMessage):
        logger.error("Ошибка: Ожидалось, что последнее сообщение будет HumanMessage.")
        return {
            "messages": [
                AIMessage(
                    content="Я не получил ваш последний запрос или произошла ошибка в последовательности сообщений."
                )
            ]
        }
    current_human_message = state["messages"][-1]
    user_input = current_human_message.content

    logger.debug(f"Получен пользовательский ввод: {user_input}")
    
    docs_retrieved = await retriever.ainvoke(
        f"Информация, связанная с: {user_input}",
    )
    
    char_context_str = "\n---\n".join(
        [
            doc.page_content
            for doc in docs_retrieved
            if doc.metadata.get("source") == "character_story"
        ]
    )
 
    world_context_str = "\n---\n".join(
        [
            doc.page_content
            for doc in docs_retrieved
            if doc.metadata.get("source") == "world_history"
        ]
    )

    logger.debug(f"Извлеченный контекст (персонаж) для запроса: {char_context_str[:200]}...")
    logger.debug(f"Извлеченный контекст (мир) для запроса: {world_context_str[:200]}...")

    logger.debug("--- (внутри узла) ГЕНЕРАЦИЯ ОТВЕТА ---")

    system_prompt_content = f"""{CHARACTER_CORE_PERSONALITY}

Контекст о персонаже ({CHARACTER_NAME}):
<character_context>
{char_context_str if char_context_str else "Нет специфической информации о персонаже для этого запроса."}
</character_context>

Контекст о мире:
<world_context>
{world_context_str if world_context_str else "Нет специфической информации о мире для этого запроса."}
</world_context>
"""

    llm_input_messages = [SystemMessage(content=system_prompt_content)] + state[
        "messages"
    ]

    prompt = ChatPromptTemplate.from_messages(llm_input_messages)
    chain = prompt | llm | StrOutputParser()

    full_response = await chain.ainvoke({})

    logger.info(f"Ответ {CHARACTER_NAME}: {full_response}")

    return {"messages": [AIMessage(content=full_response)]}


workflow = StateGraph(MessagesState)
workflow.add_node("trim_history", trim_history_node)
workflow.add_node("retrieve_and_generate", retrieve_and_generate_node)
workflow.set_entry_point("trim_history")
workflow.add_edge("trim_history", "retrieve_and_generate")
workflow.add_edge("retrieve_and_generate", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
logger.info("Граф LangGraph скомпилирован с MessagesState.")
