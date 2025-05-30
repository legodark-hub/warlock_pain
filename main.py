from llm_logic import app
from prompts import CHARACTER_NAME

print(
    f"\nНачат чат с {CHARACTER_NAME}. Введите 'выход' или 'quit' для завершения."
)
while True:
    try:
        user_query = input("Вы: ")
        if user_query.lower() in ["выход", "quit"]:
            break
        if not user_query.strip():
            continue

        inputs_for_graph = {"user_input": user_query}
        app.invoke(inputs_for_graph)

    except KeyboardInterrupt:
        print("\nЧат прерван пользователем.")
        break
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        break

print("Чат завершен.")
