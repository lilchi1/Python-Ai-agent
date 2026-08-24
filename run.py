#!/usr/bin/env python3
"""
Точка входа — запуск агента для решения задач программирования.
"""
import sys
import os
import argparse

# Добавляем src в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm.local_llm import LocalLLM
from src.memory.vector_store import LocalVectorStore
from src.agent.coding_agent import CodingAgent
from src.orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser(
        description="Локальный RAG-агент для программирования"
    )
    parser.add_argument(
        "task",
        type=str,
        nargs="?",
        default="Напиши функцию, которая вычисляет факториал числа",
        help="Задача для решения"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/deepseek-coder-1.3b-instruct",
        help="Модель для генерации"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Максимальное число попыток"
    )
    parser.add_argument(
        "--init-knowledge",
        action="store_true",
        help="Инициализировать базу знаний примерами"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🧠 ЛОКАЛЬНЫЙ RAG-АГЕНТ ДЛЯ ПРОГРАММИРОВАНИЯ")
    print("=" * 60)

    # 1. Инициализация LLM
    llm = LocalLLM(model_name=args.model)

    # 2. Инициализация памяти
    vector_store = LocalVectorStore()

    # 3. Инициализация агента
    agent = CodingAgent(llm, vector_store, max_attempts=args.max_attempts)

    # 4. Инициализация оркестратора
    orchestrator = Orchestrator(agent)

    # 5. Инициализация знаний (опционально)
    if args.init_knowledge:
        print("\n📚 Инициализация базы знаний...")
        init_knowledge(vector_store)

    # 6. Выполнение задачи
    try:
        result = orchestrator.execute(args.task)
    except KeyboardInterrupt:
        print("\n\n⚠️ Прервано пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 7. Статистика
    stats = orchestrator.get_stats()
    print("\n📊 СТАТИСТИКА:")
    print(f"   Всего задач: {stats['total_tasks']}")
    print(f"   Успешно: {stats['successful']}")
    print(f"   Успешность: {stats['success_rate']*100:.1f}%")


def init_knowledge(vector_store):
    """Инициализация базы знаний примерами."""
    examples = [
        """Задача: Напиши функцию, которая вычисляет факториал числа.
Решение:
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)""",

        """Задача: Напиши функцию, которая проверяет, является ли число простым.
Решение:
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True""",

        """Задача: Напиши функцию, которая сортирует список чисел.
Решение:
def sort_list(arr):
    return sorted(arr)""",

        """Задача: Напиши функцию, которая находит максимальный элемент в списке.
Решение:
def find_max(arr):
    if not arr:
        return None
    return max(arr)""",

        """Задача: Напиши функцию, которая реверсирует строку.
Решение:
def reverse_string(s):
    return s[::-1]"""
    ]

    vector_store.add_documents(examples)
    print(f"✅ Добавлено {len(examples)} примеров в базу знаний")


if __name__ == "__main__":
    main()