#!/usr/bin/env python3
"""
Нейрогенератор кода — использует локальную языковую модель Mini-GPT (C)
как "мозг" агента для генерации кода.

Архитектура агента:
  1. Нейрогенератор (MiniGPT, чистый C, через ctypes) — основной генератор.
  2. Шаблонный генератор — запасной и для "ремонта" некорректного кода.
  3. Оценщик (compile/анализ) — проверка валидности, цикл исправления.

Всё работает локально, без облачных/внешних API.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.model.minigpt_binding import MiniGPT, compile_lib

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model')
DEFAULT_MODEL_PATH = os.path.join(_MODEL_DIR, 'model.bin')


class NeuralGenerator:
    """
    Генерация кода нейросетью.
    Промпт строится из задачи, язык подставляется, на выходе — код.
    """

    def __init__(self, model_path=None, temperature=0.7, max_tokens=600):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = None
        self._loaded = False

    def is_available(self) -> bool:
        """Проверяет наличие скомпилированной модели."""
        if self._loaded:
            return True
        return os.path.exists(self.model_path)

    def load(self):
        if not self._loaded:
            if not os.path.exists(os.path.join(_MODEL_DIR, 'minigpt_lib.dll')):
                ok = compile_lib(quiet=True)
                if not ok:
                    return False
            self.model = MiniGPT(model_path=self.model_path, quiet=True)
            self._loaded = True
        return True

    # ── Построение промпта ─────────────────────────────────────
    def build_prompt(self, task: str, language: str, context: str = "") -> str:
        # Для максимального "вспоминания" из нейробиблиотеки лучше давать
        # компактную сигнатуру функции (модель обучалась достраивать от signature).
        sig = self._signature_from_task(task)
        if sig:
            return sig
        prompt_lines = [
            f"# Реши задачу: {task}",
            f"# Язык: {language}",
        ]
        if context:
            prompt_lines.append(f"# Похожее решение:\n{context[:500]}")
        prompt_lines.append("```")
        prompt_lines.append(f"# {language} solution:")
        return "\n".join(prompt_lines) + "\n"

    def _signature_from_task(self, task: str):
        """Извлекает сигнатуру функции для максимально точного достраивания
        нейромоделью (обучалась на завершении от сигнатуры)."""
        try:
            from src.generator.code_generator import (
                CodeGenerator, _extract_name, _extract_params,
            )
            gen = CodeGenerator()
            # Для нейро-вспоминания предпочитаем канонические имена из
            # библиотеки (_guess_name), а не сырую трансплитерацию задачи.
            name = gen._guess_name(task)
            if not name:
                name = _extract_name(task)
            params = _extract_params(task)
            if not params:
                params = gen._infer_params(task)
            if name:
                param_names = ", ".join(p[0] for p in params) if params else ""
                return f"def {name}({param_names}):\n"
        except Exception:
            pass
        return None

    def _guess_name(self, task: str) -> str:
        from src.generator.code_generator import CodeGenerator
        return CodeGenerator()._guess_name(task)

    def generate(self, task: str, language: str = "python",
                 context: str = "") -> tuple:
        """
        Возвращает (code, success).
        success=True — модель доступна и сгенерирован непустой код.
        """
        if not self.load():
            return "", False

        prompt = self.build_prompt(task, language, context)
        sig = self._signature_from_task(task)
        # Если известна сигнатура — минимальный промпт для точного достраивания.
        start = sig if sig else prompt
        raw = self.model.generate(start, max_tokens=self.max_tokens,
                                  temperature=self.temperature)

        # Извлекаем код (после маркера решения и до закрывающего ``` )
        code = self._extract_code(raw, start)
        return code, bool(code.strip())

    def _extract_code(self, raw: str, prompt: str) -> str:
        # Убираем промпт из начала
        code = raw[len(prompt):] if raw.startswith(prompt) else raw

        # Обрезаем до маркера закрытия
        for marker in ['```', '# End', '# end']:
            idx = code.find(marker)
            if idx != -1:
                code = code[:idx]
                break

        # Обрезаем по последней пустой строке + лишние трейлинг-комментарии
        lines = code.split('\n')
        # Убираем пустые строки снизу
        while lines and not lines[-1].strip():
            lines.pop()
        return '\n'.join(lines)

    def close(self):
        if self.model:
            self.model.close()
            self.model = None
            self._loaded = False
