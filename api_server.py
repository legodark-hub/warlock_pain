from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm_logic import app
from prompts import CHARACTER_NAME
from langchain_core.messages import HumanMessage
from loguru import logger

class ChatRequest(BaseModel):
    thread_id: str
    message: str

class ChatResponse(BaseModel):
    character_name: str
    response: str

fastapi_app = FastAPI(
    title="Warlock Pain Chat API",
    description="API для взаимодействия с персонажем Warlock Pain.",
    version="1.0.0"
)

@fastapi_app.on_event("startup")
async def startup_event():
    logger.info("Запуск FastAPI приложения...")
    if hasattr(app, 'checkpointer') and app.checkpointer and hasattr(type(app.checkpointer), '__name__'):
        logger.info(f"Используется {type(app.checkpointer).__name__} для сохранения истории диалогов.")
    else:
        logger.warning("Checkpointer не настроен или не имеет имени типа.")


@fastapi_app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    logger.info(f"Получен запрос для thread_id: {request.thread_id}, сообщение: '{request.message}'")

    if not request.message.strip():
        logger.warning("Получено пустое сообщение.")
        raise HTTPException(status_code=400, detail="Сообщение не может быть пустым.")

    inputs_for_graph = {"messages": [HumanMessage(content=request.message)]}

    try:
        result = await app.ainvoke(
            inputs_for_graph,
            config={"configurable": {"thread_id": request.thread_id}}
        )


        if result and "messages" in result and isinstance(result["messages"], list) and result["messages"]:
            ai_message = result["messages"][-1]
            if hasattr(ai_message, 'content'):
                logger.info(f"Ответ ИИ для thread_id {request.thread_id}: {ai_message.content}")
                return ChatResponse(character_name=CHARACTER_NAME, response=ai_message.content)
            else:
                logger.error(f"Объект сообщения ИИ не имеет атрибута 'content' для thread_id {request.thread_id}. Сообщение: {ai_message}")
                raise HTTPException(status_code=500, detail="Ошибка формата сообщения от ИИ.")
        else:
            logger.error(f"Не удалось получить корректный ответ от графа для thread_id {request.thread_id}. Результат: {result}")
            raise HTTPException(status_code=500, detail="Не удалось получить ответ от ИИ.")
    except Exception as e:
        logger.exception(f"Ошибка во время вызова ИИ для thread_id {request.thread_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")