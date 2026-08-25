"""
Мультиязычный агент для решения задач программирования.
Генерирует код на любом языке, проверяет и самообучается.
"""
import re
from typing import Dict, Any, List, Optional

from src.generator.code_generator import CodeGenerator
from src.evaluator.code_evaluator import CodeEvaluator
from src.languages.registry import LanguageRegistry, LanguageInfo
from src.memory.vector_store import LocalVectorStore


class CodingAgent:
    def __init__(
        self,
        vector_store: LocalVectorStore,
        max_attempts: int = 3,
    ):
        self.vector_store = vector_store
        self.generator = CodeGenerator()
        self.evaluator = CodeEvaluator()
        self.max_attempts = max_attempts

    def solve(self, task: str, language: Optional[str] = None) -> Dict[str, Any]:
        lang_name = self._resolve_language(task, language)
        lang_info = LanguageRegistry.get(lang_name)

        if not lang_info:
            return {
                'success': False,
                'code': f"// Unsupported language: {lang_name}\n// Supported: {', '.join(LanguageRegistry.supported())}",
                'language': lang_name,
                'attempts': 0,
                'feedback': f"Language '{lang_name}' not supported",
                'history': [],
            }

        print(f"  [lang] {lang_info.name} ({lang_info.ext})")

        context_docs = self.vector_store.search(f"{task} {lang_name}", n_results=3)
        context = "\n\n".join(d['text'] for d in context_docs) if context_docs else ""

        attempts = []
        best_code = ""
        best_success = False
        feedback = ""

        for attempt in range(self.max_attempts):
            plan = self._generate_plan(task, lang_info, context, attempts)
            code = self._generate_code(task, lang_info, context, plan, attempts)
            success, feedback = self._evaluate(code, lang_info)

            attempts.append({
                'code': code,
                'feedback': feedback,
                'success': success,
            })

            if success:
                best_code = code
                best_success = True
                break
            elif len(code) > len(best_code):
                best_code = code

        if not best_code and attempts:
            best_code = attempts[-1].get('code', '')

        self._learn(task, best_code, lang_name, best_success, feedback)

        return {
            'success': best_success,
            'code': best_code,
            'language': lang_name,
            'language_info': {
                'name': lang_info.name,
                'ext': lang_info.ext,
            },
            'attempts': len(attempts),
            'feedback': feedback,
            'history': attempts,
        }

    def _resolve_language(self, task: str, explicit: Optional[str]) -> str:
        if explicit:
            info = LanguageRegistry.get(explicit)
            if info:
                return info.name.lower()

        detected = LanguageRegistry.detect_language(task)
        if detected:
            return detected

        task_lower = task.lower()
        for lang_name in LanguageRegistry.supported():
            info = LanguageRegistry.get(lang_name)
            if info:
                for kw in info.keywords[:5]:
                    if kw.lower() in task_lower:
                        return lang_name
        return 'python'

    def _generate_plan(self, task: str, lang: LanguageInfo,
                       context: str, prev_attempts: List[Dict]) -> List[str]:
        steps = []
        steps.append(f"Определить имя и параметры функции для языка {lang.name}")
        steps.append("Реализовать основную логику")
        steps.append("Добавить обработку ошибок")
        if prev_attempts:
            last = prev_attempts[-1]
            if not last.get('success') and last.get('feedback'):
                steps.append(f"Исправить: {last['feedback'][:80]}")
        return steps

    def _generate_code(self, task: str, lang: LanguageInfo,
                       context: str, plan: List[str],
                       prev_attempts: List[Dict]) -> str:
        error_hint = ""
        if prev_attempts and not prev_attempts[-1].get('success'):
            error_hint = prev_attempts[-1].get('feedback', '')

        enhanced_task = task
        if error_hint:
            enhanced_task += f" (Предыдущая ошибка: {error_hint})"

        return self.generator.generate(enhanced_task, lang.name, context)

    def _evaluate(self, code: str, lang: LanguageInfo) -> tuple:
        if not code or not code.strip():
            return False, "Generated code is empty"

        result = self.evaluator.evaluate(code, lang.name)
        if result['success']:
            return True, "Code passed all checks"
        else:
            return False, "; ".join(result['errors'])

    def _learn(self, task: str, code: str, lang: str,
               success: bool, feedback: str):
        if not code:
            return

        if success:
            entry = f"Task: {task}\nLanguage: {lang}\nSolution:\n{code}"
            self.vector_store.add_documents(
                texts=[entry],
                metadatas=[{'task': task, 'language': lang, 'success': True}],
            )
            self.generator.register_solution(task, code, lang)
        else:
            entry = f"Task: {task}\nLanguage: {lang}\nError: {feedback}\nFailed code:\n{code}"
            self.vector_store.add_documents(
                texts=[entry],
                metadatas=[{'task': task, 'language': lang, 'success': False, 'error': feedback[:50]}],
            )
