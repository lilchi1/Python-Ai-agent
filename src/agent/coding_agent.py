"""
Основной агент для решения задач программирования.
"""
import re
from typing import Dict, Any, List, Optional

from src.llm.local_llm import LocalLLM
from src.memory.vector_store import LocalVectorStore
from src.agent import prompts


class CodingAgent:
    def __init__(
        self,
        llm: LocalLLM,
        vector_store: LocalVectorStore,
        max_attempts: int = 3
    ):
        self.llm = llm
        self.vector_store = vector_store
        self.max_attempts = max_attempts
        self.history = []

    def solve(self, task: str) -> Dict[str, Any]:
        print(f"📌 Задача: {task}")

        # Поиск в базе знаний
        context_docs = self.vector_store.search(task, n_results=2)
        context = "\n\n".join([doc['text'] for doc in context_docs]) if context_docs else ""

        if context:
            print(f"📚 Найдено {len(context_docs)} релевантных документов")

        attempts = []
        code = ""
        feedback = ""
        success = False
        attempt = 0

        for attempt in range(self.max_attempts):
            print(f"\n🔄 Попытка {attempt + 1}/{self.max_attempts}")

            # Планирование
            plan = self._generate_plan(task, context, attempts)
            if plan:
                print(f"📋 План: {plan[:3]}..." if len(plan) > 3 else f"📋 План: {plan}")

            # Генерация кода
            code = self._generate_code(task, context, plan, attempts)
            if code:
                print(f"💻 Код сгенерирован ({len(code)} символов)")
            else:
                print(f"💻 Код не сгенерирован")

            # Оценка кода
            if code:
                success, feedback = self._evaluate_code(code)
            else:
                success = False
                feedback = "Код пуст"

            attempts.append({
                "code": code,
                "feedback": feedback,
                "success": success
            })

            if success:
                print(f"✅ Успешно! Попыток: {attempt + 1}")
                break

            print(f"❌ Ошибка: {feedback[:100]}...")

        # Самообучение
        self._learn(task, code, success, feedback)

        return {
            "success": success,
            "code": code,
            "attempts": attempt + 1,
            "feedback": feedback,
            "history": attempts
        }

    def _generate_plan(self, task: str, context: str, previous_attempts: List[Dict]) -> List[str]:
        """Генерация плана решения."""
        try:
            # Формируем промпт вручную, без использования format_prompt
            prompt = f"""Ты — ассистент по программированию на Python.
Составь план решения задачи.

Задача: {task}

"""
            if context:
                prompt += f"""Контекст (похожие решения):
{context}

"""
            prompt += """План решения (шаг за шагом):
"""

            response = self.llm.generate(prompt, max_new_tokens=256, temperature=0.3)

            # Парсим план
            lines = response.split("\n")
            plan = []
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-") or line.startswith("*")):
                    clean = re.sub(r'^[\d\-\*\.]+\s*', '', line)
                    if clean:
                        plan.append(clean)

            return plan if plan else [response]

        except Exception as e:
            print(f"⚠️ Ошибка планирования: {e}")
            return []

    def _generate_code(
        self,
        task: str,
        context: str,
        plan: List[str],
        previous_attempts: List[Dict]
    ) -> str:
        """Генерация кода для задачи."""
        try:
            # Формируем план в текст
            plan_text = ""
            if plan:
                plan_text = "План решения:\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(plan)]) + "\n"

            # Информация о предыдущих ошибках
            error_context = ""
            if previous_attempts and not previous_attempts[-1].get("success", False):
                error_context = f"\nПредыдущая попытка завершилась ошибкой:\n{previous_attempts[-1]['feedback']}\n"

            # Формируем полный промпт
            full_context = ""
            if context:
                full_context += f"Контекст (похожие решения):\n{context}\n\n"
            if plan_text:
                full_context += plan_text
            if error_context:
                full_context += error_context

            prompt = f"""Ты — ассистент по программированию на Python.
Реши задачу, написав только код.

Задача: {task}

{full_context}

Напиши код на Python. Используй только код, без пояснений.
```python
"""

            response = self.llm.generate(prompt, max_new_tokens=512, temperature=0.2)

            # Извлекаем код из ответа
            return self._extract_code(response)

        except Exception as e:
            print(f"⚠️ Ошибка генерации: {e}")
            return ""

    def _extract_code(self, text: str) -> str:
        """Извлечение кода из ответа."""
        if not text:
            return ""

        # Поиск блока ```python ... ```
        if "```python" in text:
            parts = text.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0].strip()
                if code:
                    return code

        if "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                code = parts[1].strip()
                if code and not code.startswith("python"):
                    return code

        # Если есть слово "def" или "class" — возвращаем весь текст
        if "def " in text or "class " in text:
            return text.strip()

        # Иначе — пустая строка
        return ""

    def _evaluate_code(self, code: str) -> tuple:
        """Оценка кода."""
        if not code.strip():
            return False, "Код пуст"

        # Проверка синтаксиса
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return False, f"Синтаксическая ошибка: {e}"

        # Проверка на заглушки
        if "def " in code and "pass" in code and "return" not in code:
            if "TODO" in code or "..." in code:
                return False, "Код содержит заглушки (TODO, ...)"

        return True, "Корректный синтаксис"

    def _learn(self, task: str, code: str, success: bool, feedback: str):
        """Самообучение на основе результата."""
        if not code:
            return

        if success:
            entry = f"Задача: {task}\nРешение:\n{code}"
            self.vector_store.add_documents(
                texts=[entry],
                metadatas=[{"task": task, "success": True}]
            )
            print("📚 Успешное решение сохранено в базу знаний")
        else:
            entry = f"Задача: {task}\nОшибка: {feedback}\nКод с ошибкой:\n{code}"
            self.vector_store.add_documents(
                texts=[entry],
                metadatas=[{"task": task, "success": False, "error": feedback[:50]}]
            )
            print("📚 Неудачная попытка сохранена в базу знаний")