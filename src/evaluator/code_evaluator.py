"""
Мультиязычный оценщик кода.
Проверяет синтаксис, стиль, безопасность и наличие заглушек.
"""
import ast
import re
import subprocess
import tempfile
import os
from typing import Optional, Tuple


class CodeEvaluator:

    @staticmethod
    def check_syntax_python(code: str) -> Tuple[bool, Optional[str]]:
        try:
            compile(code, '<string>', 'exec')
            return True, None
        except SyntaxError as e:
            return False, f"SyntaxError: line {e.lineno}: {e.msg}"

    @staticmethod
    def check_syntax_generic(code: str, lang: str) -> Tuple[bool, Optional[str]]:
        lang_lower = lang.lower()

        if lang_lower == 'python':
            return CodeEvaluator.check_syntax_python(code)

        if lang_lower in ('javascript', 'typescript'):
            return CodeEvaluator._check_braces(code, lang)

        if lang_lower in ('c', 'c++', 'java', 'c#', 'go', 'rust', 'kotlin', 'swift', 'php'):
            return CodeEvaluator._check_braces(code, lang)

        if lang_lower == 'lua':
            do_count = len(re.findall(r'\bdo\b', code))
            end_count = len(re.findall(r'\bend\b', code))
            if do_count > end_count:
                return False, f"Unmatched 'do': {do_count} do vs {end_count} end"
            return True, None

        if lang_lower == 'haskell':
            return True, None

        if lang_lower in ('bash', 'sh'):
            fi_count = len(re.findall(r'\bfi\b', code))
            do_count = len(re.findall(r'\bdo\b', code))
            done_count = len(re.findall(r'\bdone\b', code))
            if fi_count % 2 != 0:
                return False, f"Odd number of 'fi' keywords: {fi_count}"
            if do_count != done_count:
                return False, f"Mismatched do/done: {do_count} do, {done_count} done"
            return True, None

        return True, None

    @staticmethod
    def _check_braces(code: str, lang: str) -> Tuple[bool, Optional[str]]:
        stack = []
        in_string = False
        string_char = None
        in_line_comment = False
        in_block_comment = False
        prev = ''

        for i, ch in enumerate(code):
            if in_line_comment:
                if ch == '\n':
                    in_line_comment = False
                continue
            if in_block_comment:
                if prev == '*' and ch == '/':
                    in_block_comment = False
                prev = ch
                continue
            if in_string:
                if ch == string_char and prev != '\\':
                    in_string = False
                prev = ch
                continue

            if ch in ('"', "'") and (i == 0 or code[i-1] != '\\'):
                in_string = True
                string_char = ch
                prev = ch
                continue
            if ch == '/' and i + 1 < len(code):
                if code[i+1] == '/':
                    in_line_comment = True
                    prev = ch
                    continue
                if code[i+1] == '*':
                    in_block_comment = True
                    prev = ch
                    continue
            if ch == '#' and lang.lower() in ('python', 'ruby', 'bash', 'php', 'lua'):
                in_line_comment = True
                prev = ch
                continue

            if ch == '{':
                stack.append('{')
            elif ch == '}':
                if not stack:
                    return False, f"Unmatched '}}' at position {i}"
                stack.pop()
            elif ch == '(':
                stack.append('(')
            elif ch == ')':
                if not stack:
                    return False, f"Unmatched ')' at position {i}"
                stack.pop()
            elif ch == '[':
                stack.append('[')
            elif ch == ']':
                if not stack:
                    return False, f"Unmatched ']' at position {i}"
                stack.pop()
            prev = ch

        if stack:
            return False, f"Unclosed brackets: {len(stack)} unclosed ({stack[-1]})"
        return True, None

    @staticmethod
    def check_style(code: str) -> Tuple[bool, Optional[str]]:
        warnings = []
        for i, line in enumerate(code.split("\n"), 1):
            if len(line) > 120:
                warnings.append(f"Line {i} too long ({len(line)} > 120)")
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Global):
                    warnings.append(f"Global statement at line {node.lineno}")
        except SyntaxError:
            pass
        return len(warnings) == 0, "\n".join(warnings) if warnings else None

    @staticmethod
    def check_safety(code: str, lang: str) -> Tuple[bool, Optional[str]]:
        forbidden = ['os.system', 'subprocess', 'eval(', 'exec(',
                     '__import__', 'system(', 'popen(', 'shell=True',
                     'rm -rf', 'rm -r /', 'format(', 'chmod 777']
        for pattern in forbidden:
            if pattern in code:
                return False, f"Potentially unsafe: '{pattern}'"
        return True, None

    @staticmethod
    def check_stubs(code: str, lang: str) -> Tuple[bool, Optional[str]]:
        lang_lower = lang.lower()
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped == 'pass':
                return False, f"Stub 'pass' found at line {i}"
            if stripped == '...':
                return False, f"Stub '...' found at line {i}"
            if stripped.startswith('// TODO') or stripped.startswith('# TODO'):
                continue
            if stripped == 'return None' or stripped == 'return null':
                prev_lines = [lines[j].strip() for j in range(max(0, i-3), i-1)]
                has_logic = any(l and l != line.strip() and not l.startswith('#') and not l.startswith('//')
                               for l in prev_lines)
                if not has_logic:
                    return False, f"Possibly incomplete at line {i}"
        return True, None

    @staticmethod
    def check_structure(code: str, lang: str) -> Tuple[bool, Optional[str]]:
        lang_lower = lang.lower()
        if lang_lower == 'python':
            if 'def ' in code and 'return' not in code and 'print' not in code:
                return False, "Function defined but no return/print statement"
        elif lang_lower in ('c', 'c++', 'java', 'c#', 'go', 'rust', 'kotlin', 'swift', 'php'):
            if ('int main' in code or 'func main' in code or 'fn main' in code) and 'return' not in code:
                return False, "Main function missing return statement"
        return True, None

    @classmethod
    def evaluate(cls, code: str, lang: str = 'python') -> dict:
        errors = []
        warnings = []

        is_valid, error = cls.check_syntax_generic(code, lang)
        if not is_valid:
            errors.append(error)

        is_clean, warning = cls.check_style(code)
        if warning:
            warnings.append(warning)

        is_safe, error = cls.check_safety(code, lang)
        if not is_safe:
            errors.append(error)

        is_complete, error = cls.check_stubs(code, lang)
        if not is_complete:
            errors.append(error)

        is_structured, error = cls.check_structure(code, lang)
        if not is_structured:
            errors.append(error)

        return {
            'success': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
        }
