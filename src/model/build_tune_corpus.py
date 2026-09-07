#!/usr/bin/env python3
"""
Сборка фокусного датасета для обучения мини-GPT.
Цель: дать модели достаточно НЕбольшой, но высококачественный набор
реальных Python-паттернов, чтобы она могла запомнить их с приемлемым
числом шагов обучения и генерировать структурно-валидный код.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT, 'src', 'model')
DATA = os.path.join(ROOT, 'data', 'tune')
os.makedirs(DATA, exist_ok=True)

# Включаем собственный код проекта (реальные паттерны)
def gather_project_code():
    chunks = []
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in
                   ('.git', '__pycache__', 'node_modules', '.venv', 'venv', 'data', 'tune')]
        for fn in files:
            if fn.endswith('.py'):
                p = os.path.join(root, fn)
                try:
                    c = open(p, encoding='utf-8').read()
                    if len(c) > 100:
                        chunks.append(c)
                except Exception:
                    pass
    return chunks

# Ключевые идиомы, которые обязательно должны быть запомнены
IDIOMS = [
    """def solve_task(task):
    \"\"\"Решает задачу, описанную в task.\"\"\"
    result = []
    for item in task:
        if item:
            result.append(item)
    return result
""",
    """class Processor:
    def __init__(self, data=None):
        self.data = data or []

    def add(self, item):
        self.data.append(item)

    def run(self):
        return [self.transform(x) for x in self.data]

    def transform(self, x):
        return x
""",
    """def main():
    import sys
    args = sys.argv[1:]
    if not args:
        print("Нужны аргументы")
        return
    for arg in args:
        print(f"arg: {arg}")

if __name__ == "__main__":
    main()
""",
    """def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return abs(a * b) // gcd(a, b) if a and b else 0
""",
    """import math

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True
""",
]


def build_tune_corpus():
    parts = []
    # Проектный код (реальные стили)
    proj = gather_project_code()
    parts.extend(proj)
    # Идиомы, продублированные несколько раз для надёжного запоминания
    for _ in range(6):
        parts.extend(IDIOMS)

    # Последовательно: отсортировать по длине, объединить
    text = "\n\n# ---\n\n".join(parts)
    out = os.path.join(DATA, 'tune_corpus.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Tune corpus: {len(text)} chars, {len(parts)} blocks -> {out}")
    return out


if __name__ == '__main__':
    build_tune_corpus()
