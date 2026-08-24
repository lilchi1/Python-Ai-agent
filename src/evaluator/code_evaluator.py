"""
Модуль оценки и проверки кода.
"""
import subprocess
import tempfile
import os
import ast
from typing import Tuple, Optional


class CodeEvaluator:
    @staticmethod
    def check_syntax(code: str) -> Tuple[bool, Optional[str]]:
        try:
            compile(code, '<string>', 'exec')
            return True, None
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"

    @staticmethod
    def check_style(code: str) -> Tuple[bool, Optional[str]]:
        warnings = []

        for i, line in enumerate(code.split("\n"), 1):
            if len(line) > 100:
                warnings.append(f"Строка {i} слишком длинная ({len(line)} > 100)")

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Global):
                    warnings.append(f"Использование global на строке {node.lineno}")
        except:
            pass

        return len(warnings) == 0, "\n".join(warnings) if warnings else None

    @staticmethod
    def run_tests(code: str, timeout: int = 5) -> Tuple[bool, str]:
        if "assert" not in code and "test_" not in code:
            return True, "Тесты не найдены"

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            result = subprocess.run(
                ["python", temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            os.unlink(temp_file)

            if result.returncode == 0:
                return True, "Все тесты пройдены"
            else:
                return False, f"Ошибка тестов:\n{result.stderr or result.stdout}"

        except subprocess.TimeoutExpired:
            return False, "Превышено время выполнения"
        except Exception as e:
            return False, f"Ошибка выполнения: {e}"

    @staticmethod
    def check_imports(code: str) -> Tuple[bool, Optional[str]]:
        forbidden = ["os.system", "subprocess", "eval", "exec", "__import__"]
        for forbidden_item in forbidden:
            if forbidden_item in code:
                return False, f"Обнаружен запрещенный импорт: {forbidden_item}"
        return True, None

    @classmethod
    def evaluate(cls, code: str) -> dict:
        errors = []
        warnings = []

        is_valid, error = cls.check_syntax(code)
        if not is_valid:
            errors.append(error)

        is_clean, warning = cls.check_style(code)
        if warning:
            warnings.append(warning)

        is_safe, error = cls.check_imports(code)
        if not is_safe:
            errors.append(error)

        tests_passed, output = cls.run_tests(code)
        if not tests_passed:
            errors.append(output)

        return {
            "success": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "tests_passed": tests_passed
        }