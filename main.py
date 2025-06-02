import asyncio
from llm_logic import app
from prompts import CHARACTER_NAME
from langchain_core.messages import HumanMessage

THREAD_ID = "warlock_pain_cli_chat_v1"

async def main_chat_loop():
    print(f"Используется {type(app.checkpointer).__name__} для сохранения истории.")
    print(f"\nНачат чат с {CHARACTER_NAME}. Введите 'выход' или 'quit' для завершения.")

    while True:
        try:
            user_query = await asyncio.to_thread(input, "Вы: ")
            if user_query.lower() in ["выход", "quit"]:
                break
            if not user_query.strip():
                continue

            inputs_for_graph = {"messages": [HumanMessage(content=user_query)]}
            await app.ainvoke(inputs_for_graph, config={"configurable": {"thread_id": THREAD_ID}})

        except KeyboardInterrupt:
            print("\nЧат прерван пользователем.")
            break
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            break

    print("Чат завершен.")

if __name__ == "__main__":
    asyncio.run(main_chat_loop())
