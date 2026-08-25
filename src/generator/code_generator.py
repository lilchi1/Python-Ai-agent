"""
Шаблонный генератор кода — создаёт код на любом языке без внешних LLM.
Использует паттерны извлечения имён, параметров и типов из текста задачи.
"""
import re
from typing import Dict, List, Optional, Tuple

from src.languages.registry import LanguageInfo, LanguageRegistry


# ── Извлечение информации из задачи ──────────────────────────────

CYRILLIC_MAP = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e',
    'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
    'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
}

NAME_TRANSLATIONS = {
    'факториал': 'factorial', 'простое': 'is_prime', 'простых': 'is_prime',
    'простого': 'is_prime', 'простая': 'is_prime',
    'сортировки': 'sort_list', 'сортировка': 'sort_list', 'сортировать': 'sort_list',
    'максимум': 'find_max', 'минимум': 'find_min',
    'сумм': 'sum_list', 'сумма': 'sum_list', 'суммирование': 'sum_list',
    'среднее': 'average', 'средний': 'average',
    'реверс': 'reverse_string', 'обратная': 'reverse_string',
    'палиндром': 'is_palindrome',
    'фибоначчи': 'fibonacci',
    'нод': 'gcd', 'нок': 'lcm',
    'последовательность': 'sequence',
    'бинарный': 'binary_search', 'поиск': 'search',
}


def _normalize_name(name: str) -> str:
    name_lower = name.lower()
    if name_lower in NAME_TRANSLATIONS:
        return NAME_TRANSLATIONS[name_lower]
    result = []
    for ch in name_lower:
        mapped = CYRILLIC_MAP.get(ch, ch)
        if mapped or ch.isalnum() or ch == '_':
            result.append(mapped if mapped else '_')
    normalized = ''.join(result).strip('_')
    if not normalized or not normalized[0].isalpha():
        normalized = 'func_' + normalized
    return normalized


def _extract_name(task: str) -> Optional[str]:
    patterns = [
        r'(?:функци[юяе]|function|func|метод|method)\s+[`"\']?(\w+)[`"\']?',
        r'(?:написать|создать|реализовать|напиши|создай|реализуй|write|create|implement)\s+'
        r'(?:функци[юяе]|function|func|класс|class)\s+[`"\']?(\w+)[`"\']?',
        r'[`"\'](\w+)[`"\']\s*(?:функци|function)',
        r'(?:def|function|func|fn)\s+(\w+)',
    ]
    skip = {
        'которая', 'который', 'которое', 'которые', 'которых', 'которому',
        'that', 'the', 'a', 'an', 'i', 'it',
        'на', 'для', 'из', 'по', 'от', 'с', 'к', 'о', 'у',
        'proverki', 'proverka', 'proverit', 'proveryaet',
        'проверки', 'проверка', 'проверить', 'проверяющую',
        'вычисления', 'вычисление', 'вычислить',
        'сортировки', 'сортировка', 'сортировать',
        'поиска', 'поиск', 'найти', 'найти',
        'reverse', 'reversing', 'reversed',
        'случайного', 'случайных', 'случайный',
        'обработки', 'обработка', 'обработать',
        'генерации', 'генерация', 'генерировать',
        'конвертации', 'конвертация', 'конвертировать',
        'подсчета', 'подсчет', 'подсчитать',
        'суммирования', 'суммирование', 'суммировать',
        'перевода', 'перевод', 'перевести',
        'шифрования', 'шифрование', 'шифровать',
        'парсинга', 'парсинг', 'парсить',
        'валидации', 'валидация', 'валидировать',
        'форматирования', 'форматирование', 'форматировать',
        'оптимизации', 'оптимизация', 'оптимизировать',
        'проверки', 'проверка', 'проверить', 'проверяющую',
        'простого', 'простых', 'простая', 'простое',
        'числа', 'число', 'чисел',
        'reverse', 'reversing', 'reversed',
        'случайного', 'случайных', 'случайный',
    }
    for p in patterns:
        m = re.search(p, task, re.IGNORECASE)
        if m:
            name = m.group(1)
            if name.lower() not in skip and len(name) > 1:
                return _normalize_name(name)
    return None


def _extract_params(task: str) -> List[Tuple[str, str]]:
    params = []
    patterns = [
        r'(?:принима[её]т|takes?|accepts?|аргумент|argument|параметр|parameter)\s+'
        r'[`"\']?(\w+)[`"\']?\s*(?:\((\w+)\)|—\s*(\w+))?',
        r'[`"\'](\w+)[`"\']\s*[:\-\u2014]\s*(\w+)',
        r'(?:список|list|массив|array|множество|set)\s+[`"\']?(\w+)[`"\']?',
        r'(?:строк[уае]|string)\s+[`"\']?(\w+)[`"\']?',
        r'(?:числ[оеа]|number|int|integer|float)\s+[`"\']?(\w+)[`"\']?',
    ]
    seen = set()
    skip = {
        'которая', 'который', 'которое', 'которые', 'которых', 'которому',
        'that', 'the', 'a', 'an', 'i', 'it',
        'на', 'для', 'из', 'по', 'от', 'с', 'к', 'о', 'у', 'не', 'и', 'или',
        'javascript', 'python', 'java', 'typescript', 'golang', 'ruby',
        'без', 'при', 'над', 'под', 'между', 'через', 'после', 'перед',
        'функция', 'функцию', 'function', 'класс', 'class', 'метод', 'method',
    }
    for p in patterns:
        for m in re.finditer(p, task, re.IGNORECASE):
            groups = m.groups()
            name = groups[0] if groups else None
            ptype = ''
            if len(groups) > 1 and groups[1]:
                ptype = groups[1]
            elif len(groups) > 2 and groups[2]:
                ptype = groups[2]
            if name and name.lower() not in seen and name.lower() not in skip:
                seen.add(name.lower())
                params.append((name, ptype or 'any'))
    return params


def _detect_return_type(task: str, lang_name: str) -> str:
    t = task.lower()
    bool_kws = ['bool', 'является ли', 'проверяет', 'true/false', 'true or false']
    int_kws = ['int', 'число', 'количество', 'чисел', 'индекс', 'степень', 'факториал']
    str_kws = ['string', 'строк', 'текст']
    list_kws = ['list', 'список', 'массив', 'sorted', 'отсортиров']
    float_kws = ['float', 'дробн', 'процент']
    dict_kws = ['dict', 'словарь', 'mapping', 'hash']

    if any(k in t for k in bool_kws):
        return _type_for('bool', lang_name)
    if any(k in t for k in int_kws):
        return _type_for('int', lang_name)
    if any(k in t for k in float_kws):
        return _type_for('float', lang_name)
    if any(k in t for k in str_kws):
        return _type_for('str', lang_name)
    if any(k in t for k in list_kws):
        return _type_for('list', lang_name)
    if any(k in t for k in dict_kws):
        return _type_for('dict', lang_name)
    return _type_for('int', lang_name)


def _type_for(base: str, lang_name: str) -> str:
    types = {
        'python': {'bool': 'bool', 'int': 'int', 'float': 'float', 'str': 'str',
                    'list': 'list', 'dict': 'dict'},
        'javascript': {'bool': 'boolean', 'int': 'number', 'float': 'number',
                       'str': 'string', 'list': 'Array', 'dict': 'object'},
        'typescript': {'bool': 'boolean', 'int': 'number', 'float': 'number',
                       'str': 'string', 'list': 'Array<any>', 'dict': 'Record<string, any>'},
        'java': {'bool': 'boolean', 'int': 'int', 'float': 'double',
                 'str': 'String', 'list': 'List<Object>', 'dict': 'Map<String, Object>'},
        'c': {'bool': 'int', 'int': 'int', 'float': 'double', 'str': 'char*',
              'list': 'int*', 'dict': 'void*'},
        'c++': {'bool': 'bool', 'int': 'int', 'float': 'double',
                'str': 'std::string', 'list': 'std::vector<int>', 'dict': 'std::map<std::string, int>'},
        'go': {'bool': 'bool', 'int': 'int', 'float': 'float64',
               'str': 'string', 'list': '[]int', 'dict': 'map[string]int'},
        'rust': {'bool': 'bool', 'int': 'i32', 'float': 'f64',
                 'str': 'String', 'list': 'Vec<i32>', 'dict': 'HashMap<String, i32>'},
        'php': {'bool': 'bool', 'int': 'int', 'float': 'float',
                'str': 'string', 'list': 'array', 'dict': 'array'},
        'ruby': {'bool': 'Boolean', 'int': 'Integer', 'float': 'Float',
                 'str': 'String', 'list': 'Array', 'dict': 'Hash'},
        'kotlin': {'bool': 'Boolean', 'int': 'Int', 'float': 'Double',
                   'str': 'String', 'list': 'List<Any>', 'dict': 'Map<String, Any>'},
        'c#': {'bool': 'bool', 'int': 'int', 'float': 'double',
               'str': 'string', 'list': 'List<object>', 'dict': 'Dictionary<string, object>'},
        'swift': {'bool': 'Bool', 'int': 'Int', 'float': 'Double',
                  'str': 'String', 'list': '[Any]', 'dict': '[String: Any]'},
        'lua': {'bool': 'boolean', 'int': 'number', 'float': 'number',
                'str': 'string', 'list': 'table', 'dict': 'table'},
        'haskell': {'bool': 'Bool', 'int': 'Int', 'float': 'Double',
                    'str': 'String', 'list': '[Int]', 'dict': 'Map String Int'},
        'bash': {'bool': '0/1', 'int': '0', 'float': '0.0',
                 'str': '""', 'list': '()', 'dict': '()'},
    }
    lang_types = types.get(lang_name.lower(), types.get('python', {}))
    return lang_types.get(base, 'any')


# ── Генерация кода ────────────────────────────────────────────────

class CodeGenerator:
    def __init__(self):
        self.known_solutions: Dict[str, str] = {}

    def register_solution(self, task_keywords: str, code: str, lang: str):
        key = f"{lang}:{task_keywords.lower()}"
        self.known_solutions[key] = code

    def generate(self, task: str, language: str, context: str = "") -> str:
        lang_info = LanguageRegistry.get(language)
        if not lang_info:
            return f"// Язык '{language}' не поддерживается.\n// Поддерживаемые: {', '.join(LanguageRegistry.supported())}"

        name = _extract_name(task) or self._guess_name(task)
        params = _extract_params(task)
        if not params:
            params = self._infer_params(task)
        ret_type = _detect_return_type(task, lang_info.name.lower())
        body = self._generate_body(task, lang_info, name, params, ret_type)

        param_str = self._format_params(params, lang_info)
        template = lang_info.templates.get('function', '')
        if not template:
            template = '{body}\n'

        try:
            code = template.format(
                name=name,
                params=param_str,
                return_type=ret_type,
                body=body,
                type_sig=f"{ret_type}",
            )
        except KeyError:
            code = f"{name}({param_str}) {{\n{body}\n}}\n"

        boilerplate = lang_info.boilerplate or ""
        code = boilerplate + code

        if boilerplate and lang_info.name.lower() in ('java', 'c#', 'kotlin'):
            code = code.rstrip('\n') + '\n}\n'

        if context:
            context_code = self._extract_code_from_context(context, lang_info)
            if context_code:
                comment = lang_info.comment
                code = f"{comment} Based on similar solution:\n{comment} ---\n{context_code}{comment} ---\n\n{code}"

        return code.rstrip('\n') + '\n'

    def _guess_name(self, task: str) -> str:
        t = task.lower()
        name_patterns = [
            (r'факториал', 'factorial'), (r'прост', 'is_prime'),
            (r'сортир', 'sort_list'), (r'максимум', 'find_max'),
            (r'минимум', 'find_min'), (r'сумм', 'sum_list'),
            (r'средн', 'average'), (r'реверс', 'reverse_string'),
            (r'палиндром', 'is_palindrome'), (r'фибоначчи', 'fibonacci'),
            (r'гуль', 'gcd'), (r'нод', 'gcd'), (r'нок', 'lcm'),
            (r'последовательность', 'sequence'), (r'числ', 'number'),
            (r'строк', 'string'), (r'массив', 'array'),
            (r'список', 'list'), (r'словарь', 'dictionary'),
            (r'граф', 'graph'), (r'дерев', 'tree'),
            (r'стек', 'stack'), (r'очередь', 'queue'),
            (r'поиск', 'search'), (r'bst', 'binary_search'),
            (r'бинарн', 'binary_search'), (r'линейн', 'linear_search'),
            (r'FizzBuzz', 'fizz_buzz'), (r'fizzbuzz', 'fizz_buzz'),
            (r'клас', 'MyClass'), (r'class', 'MyClass'),
        ]
        for pattern, name in name_patterns:
            if re.search(pattern, t):
                return name
        return 'solution'

    def _infer_params(self, task: str) -> List[Tuple[str, str]]:
        t = task.lower()
        if any(k in t for k in ['факториал', 'factorial']):
            return [('n', 'int')]
        if any(k in t for k in ['прост', 'prime']):
            return [('n', 'int')]
        if any(k in t for k in ['сортир', 'sort']):
            return [('arr', 'list')]
        if any(k in t for k in ['максимум', 'max', 'минимум', 'min']):
            return [('arr', 'list')]
        if any(k in t for k in ['сумм', 'sum']):
            return [('arr', 'list')]
        if any(k in t for k in ['средн', 'average', 'mean']):
            return [('arr', 'list')]
        if any(k in t for k in ['реверс', 'reverse']):
            return [('s', 'str')]
        if any(k in t for k in ['палиндром', 'palindrome']):
            return [('s', 'str')]
        if any(k in t for k in ['фибоначчи', 'fibonacci']):
            return [('n', 'int')]
        if any(k in t for k in ['гуль', 'gcd', 'нод']):
            return [('a', 'int'), ('b', 'int')]
        if any(k in t for k in ['нок', 'lcm']):
            return [('a', 'int'), ('b', 'int')]
        if any(k in t for k in ['buzz', 'fizz']):
            return [('n', 'int')]
        if any(k in t for k in ['бинарн', 'binary_search']):
            return [('arr', 'list'), ('target', 'int')]
        return []

    def _generate_body(self, task: str, lang: LanguageInfo, name: str,
                       params: List[Tuple[str, str]], ret_type: str) -> str:
        t = task.lower()
        lang_name = lang.name.lower()

        if any(k in t for k in ['факториал', 'factorial']):
            return self._factorial(lang_name, name)
        if any(k in t for k in ['прост', 'prime']):
            return self._is_prime(lang_name, name)
        if any(k in t for k in ['сортир', 'sort']):
            return self._sort_list(lang_name, name)
        if any(k in t for k in ['максимум', 'max']):
            return self._find_max(lang_name, name)
        if any(k in t for k in ['минимум', 'min']):
            return self._find_min(lang_name, name)
        if any(k in t for k in ['сумм', 'sum']):
            return self._sum_list(lang_name, name)
        if any(k in t for k in ['средн', 'average', 'mean']):
            return self._average(lang_name, name)
        if any(k in t for k in ['реверс', 'reverse']):
            return self._reverse(lang_name, name)
        if any(k in t for k in ['палиндром', 'palindrome']):
            return self._palindrome(lang_name, name)
        if any(k in t for k in ['фибоначчи', 'fibonacci']):
            return self._fibonacci(lang_name, name)
        if any(k in t for k in ['гуль', 'gcd', 'нод']):
            return self._gcd(lang_name, name)
        if any(k in t for k in ['нок', 'lcm']):
            return self._lcm(lang_name, name)
        if any(k in t for k in ['buzz', 'fizz']):
            return self._fizzbuzz(lang_name, name)
        if any(k in t for k in ['бинарн', 'binary_search']):
            return self._binary_search(lang_name, name)
        return self._generic_task(task, lang_name, name, params, ret_type)

    # ── Task-specific generators ──────────────────────────────────

    def _factorial(self, lang: str, name: str) -> str:
        impls = {
            'python': f"    if n <= 1:\n        return 1\n    return n * {name}(n - 1)",
            'javascript': f"    if (n <= 1) return 1;\n    return n * {name}(n - 1);",
            'typescript': f"    if (n <= 1) return 1;\n    return n * {name}(n - 1);",
            'java': f"    if (n <= 1) return 1;\n    return n * {name}(n - 1);",
            'c': f"    if (n <= 1) return 1;\n    return n * {name}(n - 1);",
            'c++': f"    if (n <= 1) return 1;\n    return n * {name}(n - 1);",
            'go': f"    if n <= 1 {{ return 1 }}\n    return n * {name}(n - 1)",
            'rust': f"    if n <= 1 {{ 1 }} else {{ n * {name}(n - 1) }}",
            'php': f"    if ($n <= 1) return 1;\n    return $n * {name}($n - 1);",
            'ruby': f"    return 1 if n <= 1\n    n * {name}(n - 1)",
            'kotlin': f"    return if (n <= 1) 1 else n * {name}(n - 1)",
            'c#': f"    if (n <= 1) return 1;\n    return n * {name}(n - 1);",
            'swift': f"    if n <= 1 {{ return 1 }}\n    return n * {name}(n: n - 1)",
            'lua': f"    if n <= 1 then return 1 end\n    return n * {name}(n - 1)",
            'haskell': f"{name} n\n    | n <= 1    = 1\n    | otherwise = n * {name} (n - 1)",
            'bash': f'    if [ "$n" -le 1 ]; then echo 1; return; fi\n    echo $(( n * $({name} $((n - 1))) ))',
        }
        return impls.get(lang, impls['python'])

    def _is_prime(self, lang: str, name: str) -> str:
        impls = {
            'python': (
                f"    if n < 2:\n        return False\n"
                f"    for i in range(2, int(n**0.5) + 1):\n"
                f"        if n % i == 0:\n            return False\n    return True"
            ),
            'javascript': (
                f"    if (n < 2) return false;\n"
                f"    for (let i = 2; i <= Math.sqrt(n); i++) {{\n"
                f"        if (n % i === 0) return false;\n    }}\n    return true;"
            ),
            'typescript': (
                f"    if (n < 2) return false;\n"
                f"    for (let i = 2; i <= Math.sqrt(n); i++) {{\n"
                f"        if (n % i === 0) return false;\n    }}\n    return true;"
            ),
            'java': (
                f"    if (n < 2) return false;\n"
                f"    for (int i = 2; i <= Math.sqrt(n); i++) {{\n"
                f"        if (n % i == 0) return false;\n    }}\n    return true;"
            ),
            'c': (
                f"    if (n < 2) return 0;\n"
                f"    for (int i = 2; i * i <= n; i++) {{\n"
                f"        if (n % i == 0) return 0;\n    }}\n    return 1;"
            ),
            'c++': (
                f"    if (n < 2) return false;\n"
                f"    for (int i = 2; i * i <= n; i++) {{\n"
                f"        if (n % i == 0) return false;\n    }}\n    return true;"
            ),
            'go': (
                f"    if n < 2 {{ return false }}\n"
                f"    for i := 2; i*i <= n; i++ {{\n"
                f"        if n%i == 0 {{ return false }}\n    }}\n    return true"
            ),
            'rust': (
                f"    if n < 2 {{ return false; }}\n"
                f"    for i in 2..=((n as f64).sqrt() as i32) {{\n"
                f"        if n % i == 0 {{ return false; }}\n    }}\n    true"
            ),
            'php': (
                f"    if ($n < 2) return false;\n"
                f"    for ($i = 2; $i * $i <= $n; $i++) {{\n"
                f"        if ($n % $i == 0) return false;\n    }}\n    return true;"
            ),
            'ruby': (
                f"    return false if n < 2\n"
                f"    (2..Math.sqrt(n).to_i).each {{ |i| return false if n % i == 0 }}\n"
                f"    true"
            ),
            'kotlin': (
                f"    if (n < 2) return false\n"
                f"    for (i in 2..Math.sqrt(n.toDouble()).toInt()) {{\n"
                f"        if (n % i == 0) return false\n    }}\n    return true"
            ),
            'c#': (
                f"    if (n < 2) return false;\n"
                f"    for (int i = 2; i * i <= n; i++) {{\n"
                f"        if (n % i == 0) return false;\n    }}\n    return true;"
            ),
            'swift': (
                f"    if n < 2 {{ return false }}\n"
                f"    for i in 2...Int(Double(n).squareRoot()) {{\n"
                f"        if n % i == 0 {{ return false }}\n    }}\n    return true"
            ),
            'lua': (
                f"    if n < 2 then return false end\n"
                f"    local i = 2\n"
                f"    while i * i <= n do\n"
                f"        if n % i == 0 then return false end\n"
                f"        i = i + 1\n    end\n    return true"
            ),
            'haskell': (
                f"{name} n\n"
                f"    | n < 2     = False\n"
                f"    | otherwise = all (\\i -> n `mod` i /= 0) [2..floor (sqrt (fromIntegral n))]"
            ),
            'bash': (
                '    if [ "$n" -lt 2 ]; then echo "false"; return; fi\n'
                '    local i=2\n'
                '    while [ $((i * i)) -le "$n" ]; do\n'
                '        if [ $((n % i)) -eq 0 ]; then echo "false"; return; fi\n'
                '        i=$((i + 1))\n    done\n    echo "true"'
            ),
        }
        return impls.get(lang, impls['python'])

    def _sort_list(self, lang: str, name: str) -> str:
        impls = {
            'python': f"    return sorted(arr)",
            'javascript': f"    return [...arr].sort((a, b) => a - b);",
            'typescript': f"    return [...arr].sort((a, b) => a - b);",
            'java': f"    List<Integer> sorted = new ArrayList<>(arr);\n    Collections.sort(sorted);\n    return sorted;",
            'c': f"    // Пузырьковая сортировка\n    for (int i = 0; i < n - 1; i++)\n        for (int j = 0; j < n - i - 1; j++)\n            if (arr[j] > arr[j+1]) {{ int t = arr[j]; arr[j] = arr[j+1]; arr[j+1] = t; }}\n    return arr;",
            'c++': f"    std::sort(arr.begin(), arr.end());\n    return arr;",
            'go': f"    sort.Ints(arr)\n    return arr",
            'rust': f"    arr.sort();\n    arr",
            'php': f"    sort($arr);\n    return $arr;",
            'ruby': f"    arr.sort",
            'kotlin': f"    return arr.sorted()",
            'c#': f"    arr.Sort();\n    return arr;",
            'swift': f"    return arr.sorted()",
            'lua': f"    table.sort(arr)\n    return arr",
            'haskell': f"{name} = Prelude.sort",
            'bash': f'    echo "${{arr[@]}}" | tr " " "\\n" | sort -n | tr "\\n" " "',
        }
        return impls.get(lang, impls['python'])

    def _find_max(self, lang: str, name: str) -> str:
        impls = {
            'python': f"    if not arr:\n        return None\n    return max(arr)",
            'javascript': f"    if (!arr.length) return undefined;\n    return Math.max(...arr);",
            'typescript': f"    if (!arr.length) return undefined;\n    return Math.max(...arr);",
            'java': f"    if (arr.isEmpty()) return null;\n    return Collections.max(arr);",
            'c': f"    if (n <= 0) return 0;\n    int mx = arr[0];\n    for (int i = 1; i < n; i++)\n        if (arr[i] > mx) mx = arr[i];\n    return mx;",
            'c++': f"    if (arr.empty()) return 0;\n    return *std::max_element(arr.begin(), arr.end());",
            'go': f"    if len(arr) == 0 {{ return 0 }}\n    mx := arr[0]\n    for _, v := range arr[1:] {{ if v > mx {{ mx = v }} }}\n    return mx",
            'rust': f"    arr.iter().cloned().max().unwrap_or(0)",
            'php': f"    if (empty($arr)) return null;\n    return max($arr);",
            'ruby': f"    arr.max",
            'kotlin': f"    return arr.maxOrNull()",
            'c#': f"    return arr.Max();",
            'swift': f"    return arr.max()",
            'lua': f"    local mx = arr[1]\n    for _, v in ipairs(arr) do if v > mx then mx = v end end\n    return mx",
            'haskell': f"{name} = maximum",
            'bash': f'    echo "${{arr[@]}}" | tr " " "\\n" | sort -n | tail -1',
        }
        return impls.get(lang, impls['python'])

    def _find_min(self, lang: str, name: str) -> str:
        impls = {
            'python': f"    if not arr:\n        return None\n    return min(arr)",
            'javascript': f"    if (!arr.length) return undefined;\n    return Math.min(...arr);",
            'typescript': f"    if (!arr.length) return undefined;\n    return Math.min(...arr);",
            'java': f"    if (arr.isEmpty()) return null;\n    return Collections.min(arr);",
            'c': f"    if (n <= 0) return 0;\n    int mn = arr[0];\n    for (int i = 1; i < n; i++)\n        if (arr[i] < mn) mn = arr[i];\n    return mn;",
            'c++': f"    if (arr.empty()) return 0;\n    return *std::min_element(arr.begin(), arr.end());",
            'go': f"    if len(arr) == 0 {{ return 0 }}\n    mn := arr[0]\n    for _, v := range arr[1:] {{ if v < mn {{ mn = v }} }}\n    return mn",
            'rust': f"    arr.iter().cloned().min().unwrap_or(0)",
            'php': f"    if (empty($arr)) return null;\n    return min($arr);",
            'ruby': f"    arr.min",
            'kotlin': f"    return arr.minOrNull()",
            'c#': f"    return arr.Min();",
            'swift': f"    return arr.min()",
            'lua': f"    local mn = arr[1]\n    for _, v in ipairs(arr) do if v < mn then mn = v end end\n    return mn",
            'haskell': f"{name} = minimum",
            'bash': f'    echo "${{arr[@]}}" | tr " " "\\n" | sort -n | head -1',
        }
        return impls.get(lang, impls['python'])

    def _sum_list(self, lang: str, name: str) -> str:
        impls = {
            'python': f"    return sum(arr)",
            'javascript': f"    return arr.reduce((a, b) => a + b, 0);",
            'typescript': f"    return arr.reduce((a, b) => a + b, 0);",
            'java': f"    return arr.stream().mapToInt(Integer::intValue).sum();",
            'c': f"    int s = 0;\n    for (int i = 0; i < n; i++) s += arr[i];\n    return s;",
            'c++': f"    int s = 0;\n    for (auto x : arr) s += x;\n    return s;",
            'go': f"    s := 0\n    for _, v := range arr {{ s += v }}\n    return s",
            'rust': f"    arr.iter().sum()",
            'php': f"    return array_sum($arr);",
            'ruby': f"    arr.sum",
            'kotlin': f"    return arr.sum()",
            'c#': f"    return arr.Sum();",
            'swift': f"    return arr.reduce(0, +)",
            'lua': f"    local s = 0\n    for _, v in ipairs(arr) do s = s + v end\n    return s",
            'haskell': f"{name} = sum",
            'bash': f'    echo "${{arr[@]}}" | tr " " "\\n" | paste -sd+ | bc',
        }
        return impls.get(lang, impls['python'])

    def _average(self, lang: str, name: str) -> str:
        impls = {
            'python': f"    return sum(arr) / len(arr) if arr else 0",
            'javascript': f"    return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;",
            'typescript': f"    return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;",
            'java': f"    return arr.stream().mapToDouble(Integer::doubleValue).average().orElse(0.0);",
            'c': f"    if (n <= 0) return 0.0;\n    double s = 0;\n    for (int i = 0; i < n; i++) s += arr[i];\n    return s / n;",
            'c++': f"    if (arr.empty()) return 0.0;\n    double s = 0;\n    for (auto x : arr) s += x;\n    return s / arr.size();",
            'go': f"    if len(arr) == 0 {{ return 0.0 }}\n    s := 0\n    for _, v := range arr {{ s += v }}\n    return float64(s) / float64(len(arr))",
            'rust': f"    if arr.is_empty() {{ 0.0 }} else {{ arr.iter().sum::<i32>() as f64 / arr.len() as f64 }}",
            'php': f"    return empty($arr) ? 0.0 : array_sum($arr) / count($arr);",
            'ruby': f"    arr.empty? ? 0.0 : arr.sum.to_f / arr.size",
            'kotlin': f"    return if (arr.isEmpty()) 0.0 else arr.average()",
            'c#': f"    return arr.Any() ? arr.Average() : 0.0;",
            'swift': f"    return arr.isEmpty ? 0.0 : Double(arr.reduce(0, +)) / Double(arr.count)",
            'lua': f"    if #arr == 0 then return 0 end\n    local s = 0\n    for _, v in ipairs(arr) do s = s + v end\n    return s / #arr",
            'haskell': f"{name} xs = if null xs then 0 else fromIntegral (sum xs) / fromIntegral (length xs)",
            'bash': f'    echo "${{arr[@]}}" | tr " " "\\n" | awk \'{{s+=$1}} END {{print s/NR}}\'',
        }
        return impls.get(lang, impls['python'])

    def _reverse(self, lang: str, name: str) -> str:
        impls = {
            'python': f"    return s[::-1]",
            'javascript': f"    return s.split('').reverse().join('');",
            'typescript': f"    return s.split('').reverse().join('');",
            'java': f"    return new StringBuilder(s).reverse().toString();",
            'c': f"    int len = strlen(s);\n    for (int i = 0; i < len / 2; i++) {{ char t = s[i]; s[i] = s[len-1-i]; s[len-1-i] = t; }}\n    return s;",
            'c++': f"    std::reverse(s.begin(), s.end());\n    return s;",
            'go': f"    runes := []rune(s)\n    for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {{\n        runes[i], runes[j] = runes[j], runes[i]\n    }}\n    return string(runes)",
            'rust': f"    s.chars().rev().collect()",
            'php': f"    return strrev($s);",
            'ruby': f"    s.reverse",
            'kotlin': f"    return s.reversed()",
            'c#': f"    char[] arr = s.ToCharArray();\n    Array.Reverse(arr);\n    return new string(arr);",
            'swift': f"    return String(s.reversed())",
            'lua': f"    return string.reverse(s)",
            'haskell': f"{name} = reverse",
            'bash': f'    echo -n "$s" | rev',
        }
        return impls.get(lang, impls['python'])

    def _palindrome(self, lang: str, name: str) -> str:
        impls = {
            'python': f"    s = s.lower().replace(' ', '')\n    return s == s[::-1]",
            'javascript': f"    const clean = s.toLowerCase().replace(/\\s/g, '');\n    return clean === clean.split('').reverse().join('');",
            'typescript': f"    const clean = s.toLowerCase().replace(/\\s/g, '');\n    return clean === clean.split('').reverse().join('');",
            'java': f"    String clean = s.toLowerCase().replaceAll(\\s, \"\");\n    return clean.equals(new StringBuilder(clean).reverse().toString());",
            'c': f"    int len = strlen(s);\n    for (int i = 0; i < len / 2; i++)\n        if (s[i] != s[len-1-i]) return 0;\n    return 1;",
            'c++': f"    std::string clean = s;\n    std::transform(clean.begin(), clean.end(), clean.begin(), ::tolower);\n    clean.erase(std::remove(clean.begin(), clean.end(), ' '), clean.end());\n    return clean == std::string(clean.rbegin(), clean.rend());",
            'go': f"    s = strings.ToLower(s)\n    s = strings.ReplaceAll(s, \" \", \"\")\n    runes := []rune(s)\n    for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {{\n        if runes[i] != runes[j] {{ return false }}\n    }}\n    return true",
            'rust': f"    let clean: String = s.chars().filter(|c| !c.is_whitespace()).collect();\n    let lower = clean.to_lowercase();\n    lower == lower.chars().rev().collect::<String>()",
            'php': f"    $clean = strtolower(preg_replace('/\\s/', '', $s));\n    return $clean === strrev($clean);",
            'ruby': f"    clean = s.downcase.gsub(/\\s/, '')\n    clean == clean.reverse",
            'kotlin': f"    val clean = s.lowercase().replace(Regex(\"\\\\s\"), \"\")\n    return clean == clean.reversed()",
            'c#': f"    string clean = s.ToLower().Replace(\" \", \"\");\n    return clean == new string(clean.Reverse().ToArray());",
            'swift': f"    let clean = s.lowercased().filter {{ !$0.isWhitespace }}\n    return String(clean) == String(clean.reversed())",
            'lua': f"    local clean = s:lower():gsub('%s', '')\n    local rev = clean:reverse()\n    return clean == rev",
            'haskell': f"{name} s = let clean = map toLower (filter (/= ' ') s)\n         in clean == reverse clean",
            'bash': f'    local clean=$(echo -n "$s" | tr "[:upper:]" "[:lower:]" | tr -d " ")\n    echo "$clean" | rev | grep -q "$clean" && echo "true" || echo "false"',
        }
        return impls.get(lang, impls['python'])

    def _fibonacci(self, lang: str, name: str) -> str:
        impls = {
            'python': f"    if n <= 0:\n        return 0\n    if n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b",
            'javascript': f"    if (n <= 0) return 0;\n    if (n === 1) return 1;\n    let a = 0, b = 1;\n    for (let i = 2; i <= n; i++) {{ [a, b] = [b, a + b]; }}\n    return b;",
            'typescript': f"    if (n <= 0) return 0;\n    if (n === 1) return 1;\n    let a = 0, b = 1;\n    for (let i = 2; i <= n; i++) {{ [a, b] = [b, a + b]; }}\n    return b;",
            'java': f"    if (n <= 0) return 0;\n    if (n == 1) return 1;\n    int a = 0, b = 1;\n    for (int i = 2; i <= n; i++) {{ int t = a + b; a = b; b = t; }}\n    return b;",
            'c': f"    if (n <= 0) return 0;\n    if (n == 1) return 1;\n    int a = 0, b = 1;\n    for (int i = 2; i <= n; i++) {{ int t = a + b; a = b; b = t; }}\n    return b;",
            'c++': f"    if (n <= 0) return 0;\n    if (n == 1) return 1;\n    int a = 0, b = 1;\n    for (int i = 2; i <= n; i++) {{ int t = a + b; a = b; b = t; }}\n    return b;",
            'go': f"    if n <= 0 {{ return 0 }}\n    if n == 1 {{ return 1 }}\n    a, b := 0, 1\n    for i := 2; i <= n; i++ {{ a, b = b, a+b }}\n    return b",
            'rust': f"    if n <= 0 {{ return 0; }}\n    if n == 1 {{ return 1; }}\n    let (mut a, mut b) = (0, 1);\n    for _ in 2..=n {{ let t = a + b; a = b; b = t; }}\n    b",
            'php': f"    if ($n <= 0) return 0;\n    if ($n == 1) return 1;\n    $a = 0; $b = 1;\n    for ($i = 2; $i <= $n; $i++) {{ $t = $a + $b; $a = $b; $b = $t; }}\n    return $b;",
            'ruby': f"    return 0 if n <= 0\n    return 1 if n == 1\n    a, b = 0, 1\n    (2..n).each {{ a, b = b, a + b }}\n    b",
            'kotlin': f"    if (n <= 0) return 0\n    if (n == 1) return 1\n    var a = 0; var b = 1\n    for (i in 2..n) {{ val t = a + b; a = b; b = t }}\n    return b",
            'c#': f"    if (n <= 0) return 0;\n    if (n == 1) return 1;\n    int a = 0, b = 1;\n    for (int i = 2; i <= n; i++) {{ int t = a + b; a = b; b = t; }}\n    return b;",
            'swift': f"    if n <= 0 {{ return 0 }}\n    if n == 1 {{ return 1 }}\n    var a = 0, b = 1\n    for _ in 2...n {{ (a, b) = (b, a + b) }}\n    return b",
            'lua': f"    if n <= 0 then return 0 end\n    if n == 1 then return 1 end\n    local a, b = 0, 1\n    for i = 2, n do a, b = b, a + b end\n    return b",
            'haskell': f"{name} n\n    | n <= 0    = 0\n    | n == 1    = 1\n    | otherwise = fib (n-1) + fib (n-2)\n  where fib 0 = 0; fib 1 = 1; fib k = fib (k-1) + fib (k-2)",
            'bash': f'    if [ "$n" -le 0 ]; then echo 0; return; fi\n    if [ "$n" -eq 1 ]; then echo 1; return; fi\n    local a=0 b=1\n    for ((i=2; i<=n; i++)); do local t=$((a+b)); a=$b; b=$t; done\n    echo $b',
        }
        return impls.get(lang, impls['python'])

    def _gcd(self, lang: str, name: str) -> str:
        impls = {
            'python': f"    while b:\n        a, b = b, a % b\n    return a",
            'javascript': f"    while (b) {{ [a, b] = [b, a % b]; }}\n    return a;",
            'typescript': f"    while (b) {{ [a, b] = [b, a % b]; }}\n    return a;",
            'java': f"    while (b != 0) {{ int t = a % b; a = b; b = t; }}\n    return a;",
            'c': f"    while (b) {{ int t = a % b; a = b; b = t; }}\n    return a;",
            'c++': f"    while (b) {{ int t = a % b; a = b; b = t; }}\n    return a;",
            'go': f"    for b != 0 {{ a, b = b, a%b }}\n    return a",
            'rust': f"    while b != 0 {{ let t = a % b; a = b; b = t; }}\n    a",
            'php': f"    while ($b) {{ $t = $a % $b; $a = $b; $b = $t; }}\n    return $a;",
            'ruby': f"    a, b = b, a % b until b.zero?\n    a",
            'kotlin': f"    while (b != 0) {{ val t = a % b; a = b; b = t }}\n    return a",
            'c#': f"    while (b != 0) {{ int t = a % b; a = b; b = t; }}\n    return a;",
            'swift': f"    while b != 0 {{ (a, b) = (b, a % b) }}\n    return a",
            'lua': f"    while b ~= 0 do a, b = b, a % b end\n    return a",
            'haskell': f"{name} a 0 = a\n{name} a b = {name} b (a `mod` b)",
            'bash': f'    local t\n    while [ "$b" -ne 0 ]; do t=$((a % b)); a=$b; b=$t; done\n    echo $a',
        }
        return impls.get(lang, impls['python'])

    def _lcm(self, lang: str, name: str) -> str:
        gcd_name = 'gcd'
        impls = {
            'python': f"    return abs(a * b) // {gcd_name}(a, b) if a and b else 0",
            'javascript': f"    return a && b ? Math.abs(a * b) / {gcd_name}(a, b) : 0;",
            'typescript': f"    return a && b ? Math.abs(a * b) / {gcd_name}(a, b) : 0;",
            'java': f"    return a != 0 && b != 0 ? Math.abs(a / {gcd_name}(a, b) * b) : 0;",
            'c': f"    return a && b ? a / {gcd_name}(a, b) * b : 0;",
            'c++': f"    return a && b ? a / {gcd_name}(a, b) * b : 0;",
            'go': f"    if a == 0 || b == 0 {{ return 0 }}\n    return a / {gcd_name}(a, b) * b",
            'rust': f"    if a == 0 || b == 0 {{ 0 }} else {{ a / {gcd_name}(a, b) * b }}",
            'php': f"    return $a && $b ? abs($a / {gcd_name}($a, $b) * $b) : 0;",
            'ruby': f"    a.abs * b.abs / {gcd_name}(a, b)",
            'kotlin': f"    return if (a == 0 || b == 0) 0 else Math.abs(a / {gcd_name}(a, b) * b)",
            'c#': f"    return a != 0 && b != 0 ? Math.Abs(a / {gcd_name}(a, b) * b) : 0;",
            'swift': f"    return a != 0 && b != 0 ? abs(a / {gcd_name}(a: a, b: b) * b) : 0",
            'lua': f"    if a == 0 or b == 0 then return 0 end\n    return math.abs(a * b / {gcd_name}(a, b))",
            'haskell': f"{name} a b = abs (a * b) `div` {gcd_name} a b",
            'bash': f'    if [ "$a" -eq 0 ] || [ "$b" -eq 0 ]; then echo 0; return; fi\n    local g=$({gcd_name} $a $b)\n    echo $(( a / g * b ))',
        }
        return impls.get(lang, impls['python'])

    def _fizzbuzz(self, lang: str, name: str) -> str:
        impls = {
            'python': (
                f"    for i in range(1, n + 1):\n"
                f"        if i % 15 == 0: yield 'FizzBuzz'\n"
                f"        elif i % 3 == 0: yield 'Fizz'\n"
                f"        elif i % 5 == 0: yield 'Buzz'\n"
                f"        else: yield str(i)"
            ),
            'javascript': (
                f"    const result = [];\n"
                f"    for (let i = 1; i <= n; i++) {{\n"
                f"        if (i % 15 === 0) result.push('FizzBuzz');\n"
                f"        else if (i % 3 === 0) result.push('Fizz');\n"
                f"        else if (i % 5 === 0) result.push('Buzz');\n"
                f"        else result.push(String(i));\n    }}\n    return result;"
            ),
            'typescript': (
                f"    const result: string[] = [];\n"
                f"    for (let i = 1; i <= n; i++) {{\n"
                f"        if (i % 15 === 0) result.push('FizzBuzz');\n"
                f"        else if (i % 3 === 0) result.push('Fizz');\n"
                f"        else if (i % 5 === 0) result.push('Buzz');\n"
                f"        else result.push(String(i));\n    }}\n    return result;"
            ),
            'java': (
                f"    List<String> result = new ArrayList<>();\n"
                f"    for (int i = 1; i <= n; i++) {{\n"
                f"        if (i % 15 == 0) result.add(\"FizzBuzz\");\n"
                f"        else if (i % 3 == 0) result.add(\"Fizz\");\n"
                f"        else if (i % 5 == 0) result.add(\"Buzz\");\n"
                f"        else result.add(String.valueOf(i));\n    }}\n    return result;"
            ),
            'c': (
                f"    for (int i = 1; i <= n; i++) {{\n"
                f"        if (i % 15 == 0) printf(\"FizzBuzz\\n\");\n"
                f"        else if (i % 3 == 0) printf(\"Fizz\\n\");\n"
                f"        else if (i % 5 == 0) printf(\"Buzz\\n\");\n"
                f"        else printf(\"%d\\n\", i);\n    }}"
            ),
            'c++': (
                f"    for (int i = 1; i <= n; i++) {{\n"
                f"        if (i % 15 == 0) std::cout << \"FizzBuzz\" << std::endl;\n"
                f"        else if (i % 3 == 0) std::cout << \"Fizz\" << std::endl;\n"
                f"        else if (i % 5 == 0) std::cout << \"Buzz\" << std::endl;\n"
                f"        else std::cout << i << std::endl;\n    }}"
            ),
            'go': (
                f"    for i := 1; i <= n; i++ {{\n"
                f"        switch {{\n"
                f"        case i%15 == 0: fmt.Println(\"FizzBuzz\")\n"
                f"        case i%3 == 0: fmt.Println(\"Fizz\")\n"
                f"        case i%5 == 0: fmt.Println(\"Buzz\")\n"
                f"        default: fmt.Println(i)\n    }}}}"
            ),
            'rust': (
                "    for i in 1..=n {\n"
                "        match i % 15 {\n"
                "            0 => println!(\"FizzBuzz\"),\n"
                "            _ => match i % 3 {\n"
                "                0 => println!(\"Fizz\"),\n"
                "                _ => match i % 5 {\n"
                "                    0 => println!(\"Buzz\"),\n"
                "                    _ => println!(\"{}\", i),\n"
                "                }\n"
                "            }\n"
                "        }\n"
                "    }"
            ),
            'php': (
                f"    for ($i = 1; $i <= $n; $i++) {{\n"
                f"        if ($i % 15 == 0) echo \"FizzBuzz\\n\";\n"
                f"        else if ($i % 3 == 0) echo \"Fizz\\n\";\n"
                f"        else if ($i % 5 == 0) echo \"Buzz\\n\";\n"
                f"        else echo $i . \"\\n\";\n    }}"
            ),
            'ruby': (
                f"    (1..n).each do |i|\n"
                f"        if i % 15 == 0 then puts \"FizzBuzz\"\n"
                f"        elsif i % 3 == 0 then puts \"Fizz\"\n"
                f"        elsif i % 5 == 0 then puts \"Buzz\"\n"
                f"        else puts i\n    end end"
            ),
            'kotlin': (
                f"    for (i in 1..n) {{\n"
                f"        when {{\n"
                f"            i % 15 == 0 -> println(\"FizzBuzz\")\n"
                f"            i % 3 == 0 -> println(\"Fizz\")\n"
                f"            i % 5 == 0 -> println(\"Buzz\")\n"
                f"            else -> println(i)\n    }}}}"
            ),
            'c#': (
                f"    for (int i = 1; i <= n; i++) {{\n"
                f"        if (i % 15 == 0) Console.WriteLine(\"FizzBuzz\");\n"
                f"        else if (i % 3 == 0) Console.WriteLine(\"Fizz\");\n"
                f"        else if (i % 5 == 0) Console.WriteLine(\"Buzz\");\n"
                f"        else Console.WriteLine(i);\n    }}"
            ),
            'swift': (
                "    for i in 1...n {\n"
                "        switch i % 15 {\n"
                "        case 0: print(\"FizzBuzz\")\n"
                "        default:\n"
                "            switch i % 3 {\n"
                "            case 0: print(\"Fizz\")\n"
                "            default:\n"
                "                switch i % 5 {\n"
                "                case 0: print(\"Buzz\")\n"
                "                default: print(i)\n"
                "    }}}}"
            ),
            'lua': (
                f"    for i = 1, n do\n"
                f"        if i % 15 == 0 then print(\"FizzBuzz\")\n"
                f"        elseif i % 3 == 0 then print(\"Fizz\")\n"
                f"        elseif i % 5 == 0 then print(\"Buzz\")\n"
                f"        else print(i)\n    end end"
            ),
            'haskell': (
                f"map fizzBuzz [1..n]\n"
                f"  where fizzBuzz i\n"
                f"          | i `mod` 15 == 0 = \"FizzBuzz\"\n"
                f"          | i `mod` 3 == 0  = \"Fizz\"\n"
                f"          | i `mod` 5 == 0  = \"Buzz\"\n"
                f"          | otherwise        = show i"
            ),
            'bash': (
                '    for ((i=1; i<=n; i++)); do\n'
                '        if [ $((i % 15)) -eq 0 ]; then echo "FizzBuzz"\n'
                '        elif [ $((i % 3)) -eq 0 ]; then echo "Fizz"\n'
                '        elif [ $((i % 5)) -eq 0 ]; then echo "Buzz"\n'
                '        else echo $i\n    done'
            ),
        }
        return impls.get(lang, impls['python'])

    def _binary_search(self, lang: str, name: str) -> str:
        impls = {
            'python': (
                f"    lo, hi = 0, len(arr) - 1\n"
                f"    while lo <= hi:\n"
                f"        mid = (lo + hi) // 2\n"
                f"        if arr[mid] == target: return mid\n"
                f"        elif arr[mid] < target: lo = mid + 1\n"
                f"        else: hi = mid - 1\n"
                f"    return -1"
            ),
            'javascript': (
                f"    let lo = 0, hi = arr.length - 1;\n"
                f"    while (lo <= hi) {{\n"
                f"        const mid = (lo + hi) >> 1;\n"
                f"        if (arr[mid] === target) return mid;\n"
                f"        else if (arr[mid] < target) lo = mid + 1;\n"
                f"        else hi = mid - 1;\n    }}\n    return -1;"
            ),
            'c': (
                f"    int lo = 0, hi = n - 1;\n"
                f"    while (lo <= hi) {{\n"
                f"        int mid = lo + (hi - lo) / 2;\n"
                f"        if (arr[mid] == target) return mid;\n"
                f"        else if (arr[mid] < target) lo = mid + 1;\n"
                f"        else hi = mid - 1;\n    }}\n    return -1;"
            ),
            'go': (
                f"    lo, hi := 0, len(arr)-1\n"
                f"    for lo <= hi {{\n"
                f"        mid := lo + (hi-lo)/2\n"
                f"        if arr[mid] == target {{ return mid }}\n"
                f"        if arr[mid] < target {{ lo = mid + 1 }} else {{ hi = mid - 1 }}\n"
                f"    }}\n    return -1"
            ),
        }
        return impls.get(lang, impls.get('python', '    // binary search'))

    def _generic_task(self, task: str, lang: str, name: str,
                      params: List[Tuple[str, str]], ret_type: str) -> str:
        lang_info = LanguageRegistry.get(lang)
        if not lang_info:
            return f"    // Не удалось сгенерировать код для: {task}"

        comment = lang_info.comment
        keywords_lower = [k.lower() for k in lang_info.keywords]

        lines = [f"{comment} Задача: {task}", f"{comment} Язык: {lang_info.name}", ""]

        if lang in ('python', 'ruby', 'lua', 'bash', 'haskell'):
            lines.append(f"{comment} TODO: Реализовать логику функции")
            lines.append(f"{comment} Подсказки по синтаксису ({lang_info.name}):")
            for kw in lang_info.keywords[:10]:
                lines.append(f"{comment}   - {kw}")
        else:
            lines.append(f"{comment} TODO: Implement logic")
            lines.append(f"{comment} Hints ({lang_info.name}):")
            for kw in lang_info.keywords[:10]:
                lines.append(f"{comment}   - {kw}")

        if params:
            lines.append("")
            lines.append(f"{comment} Parameters:")
            for pname, ptype in params:
                lines.append(f"{comment}   - {pname}: {ptype}")

        if 'int' in ret_type.lower() or 'number' in ret_type.lower() or 'i32' in ret_type:
            lines.append(f"    return 0  // TODO: implement")
        elif 'bool' in ret_type.lower() or 'boolean' in ret_type.lower():
            lines.append(f"    return true  // TODO: implement")
        elif 'string' in ret_type.lower():
            lines.append(f'    return ""  // TODO: implement')
        elif 'list' in ret_type.lower() or 'array' in ret_type.lower() or 'vec' in ret_type.lower():
            lines.append(f"    return []  // TODO: implement")
        elif 'dict' in ret_type.lower() or 'map' in ret_type.lower() or 'hash' in ret_type.lower():
            lines.append(f"    return {{}}  // TODO: implement")
        elif 'void' in ret_type.lower() or lang in ('c', 'bash'):
            lines.append(f"    // TODO: implement")
        else:
            lines.append(f"    return None  // TODO: implement")

        return "\n".join(lines)

    def _format_params(self, params: List[Tuple[str, str]], lang: LanguageInfo) -> str:
        if not params:
            return ""
        lang_name = lang.name.lower()
        parts = []
        for pname, ptype in params:
            if lang_name == 'python':
                parts.append(pname)
            elif lang_name == 'go':
                mapped = _type_for(ptype, lang_name) if ptype != 'any' else 'int'
                parts.append(f"{pname} {mapped}")
            elif lang_name == 'rust':
                mapped = _type_for(ptype, lang_name) if ptype != 'any' else 'i32'
                parts.append(f"{pname}: {mapped}")
            elif lang_name in ('c', 'c++', 'java', 'c#', 'kotlin', 'swift'):
                mapped = _type_for(ptype, lang_name) if ptype != 'any' else 'int'
                parts.append(f"{mapped} {pname}")
            elif lang_name in ('javascript', 'typescript'):
                if ptype != 'any':
                    mapped = _type_for(ptype, lang_name)
                    parts.append(f"{pname}: {mapped}" if lang_name == 'typescript' else pname)
                else:
                    parts.append(pname)
            elif lang_name == 'php':
                parts.append(f"${pname}")
            elif lang_name == 'ruby':
                parts.append(pname)
            elif lang_name == 'lua':
                parts.append(pname)
            elif lang_name == 'haskell':
                mapped = _type_for(ptype, lang_name) if ptype != 'any' else 'Int'
                parts.append(f"{pname} :: {mapped}")
            else:
                parts.append(f"{ptype} {pname}")
        return ", ".join(parts)

    @staticmethod
    def _extract_code_from_context(context: str, lang_info) -> str:
        if not context:
            return ""
        lang_lower = lang_info.name.lower()
        sections = context.split("\n\n")
        for section in sections:
            lang_match = re.search(r'Language:\s*(\w+)', section, re.IGNORECASE)
            if lang_match:
                ctx_lang = lang_match.group(1).lower()
                if ctx_lang != lang_lower:
                    continue
            sol_match = re.search(r'Solution:\n(.*)', section, re.DOTALL)
            if sol_match:
                code = sol_match.group(1).strip()
                lines = code.split("\n")
                clean_lines = []
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith("Task:") or stripped.startswith("Language:"):
                        continue
                    clean_lines.append(line)
                return "\n".join(clean_lines).strip() + "\n"
        return ""
