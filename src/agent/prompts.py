"""
Шаблоны промптов для агента.
"""

SYSTEM_PROMPT = """Ты — ассистент по программированию на Python.
Твоя задача — решать задачи, генерируя чистый, рабочий код.

Правила:
1. Используй только стандартную библиотеку Python, если не указано иное.
2. Код должен быть готов к запуску.
3. Добавляй комментарии только для сложных участков.
4. Используй понятные имена переменных.
5. Обрабатывай возможные ошибки (try/except).
"""

# Шаблоны без использования {system} — подставляем SYSTEM_PROMPT напрямую
CODE_GENERATION_PROMPT = """
Ты — ассистент по программированию на Python.
Реши задачу, написав только код.

Задача: {task}

{context}

Напиши код на Python. Верни только код, без пояснений.
"""

PLAN_PROMPT = """
Ты — ассистент по программированию на Python.
Составь план решения задачи.

Задача: {task}

{context}

План решения (шаг за шагом):
"""

CODE_REVIEW_PROMPT = """
Ты — эксперт по код-ревью. Проверь следующий код.

Код:
{code}

Ошибки (если есть):
- Синтаксические ошибки
- Логические ошибки
- Потенциальные проблемы

Ответ:
"""


def format_prompt(template: str, **kwargs) -> str:
    """Форматирование промпта."""
    # Подставляем SYSTEM_PROMPT только если ключ есть
    if "system" in kwargs:
        kwargs["system"] = SYSTEM_PROMPT
    
    # Заменяем {task} и {context}
    result = template
    for key, value in kwargs.items():
        placeholder = "{" + key + "}"
        result = result.replace(placeholder, str(value))
    
    return result


PROMPTS = {
    "code": CODE_GENERATION_PROMPT,
    "plan": PLAN_PROMPT,
    "review": CODE_REVIEW_PROMPT,
}