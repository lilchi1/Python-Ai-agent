"""
Реестр языков программирования — шаблоны, ключевые слова, синтаксис.
Поддерживает: Python, JavaScript, TypeScript, C, C++, Java, Go, Rust, PHP, Ruby, Swift, Kotlin, C#, Lua, Haskell.
"""
from typing import Dict, List, Optional


class LanguageInfo:
    __slots__ = ('name', 'ext', 'comment', 'keywords', 'templates', 'boilerplate')

    def __init__(self, name: str, ext: str, comment: str,
                 keywords: List[str], templates: Dict[str, str],
                 boilerplate: str = ""):
        self.name = name
        self.ext = ext
        self.comment = comment
        self.keywords = keywords
        self.templates = templates
        self.boilerplate = boilerplate


LANGUAGES: Dict[str, LanguageInfo] = {}


def _reg(name: str, ext: str, comment: str, keywords: List[str],
         templates: Dict[str, str], boilerplate: str = ""):
    LANGUAGES[name.lower()] = LanguageInfo(name, ext, comment, keywords, templates, boilerplate)


# ── Python ───────────────────────────────────────────────────────
_reg("Python", ".py", "#",
     ["def", "class", "import", "from", "return", "if", "else", "for", "while",
      "try", "except", "finally", "with", "as", "lambda", "yield", "raise",
      "True", "False", "None", "self", "async", "await", "in", "not", "and", "or"],
     {
         "function": (
             "def {name}({params}):\n"
             "{body}\n"
         ),
         "class": (
             "class {name}:\n"
             "    def __init__(self{params}):\n"
             "        {body}\n"
         ),
         "main": (
             "if __name__ == \"__main__\":\n"
             "    {body}\n"
         ),
     },
     "#!/usr/bin/env python3\n\n"
),

# ── JavaScript ───────────────────────────────────────────────────
_reg("JavaScript", ".js", "//",
     ["function", "const", "let", "var", "return", "if", "else", "for", "while",
      "class", "extends", "new", "this", "async", "await", "import", "export",
      "default", "try", "catch", "finally", "throw", "typeof", "instanceof",
      "switch", "case", "break", "continue", "null", "undefined", "true", "false"],
     {
         "function": (
             "/** TODO: description */\n"
             "function {name}({params}) {{\n"
             "{body}\n"
             "}}\n"
         ),
         "class": (
             "class {name} {{\n"
             "    constructor({params}) {{\n"
             "        {body}\n"
             "    }}\n"
             "}}\n"
         ),
         "main": "{body}\n",
     },
     "\"use strict\";\n\n"
),

# ── TypeScript ───────────────────────────────────────────────────
_reg("TypeScript", ".ts", "//",
     ["function", "const", "let", "return", "if", "else", "for", "while",
      "class", "extends", "new", "this", "async", "await", "import", "export",
      "interface", "type", "enum", "namespace", "public", "private", "protected",
      "readonly", "abstract", "implements", "keyof", "unknown", "any", "void",
      "never", "string", "number", "boolean", "null", "undefined"],
     {
         "function": (
             "/** TODO: description */\n"
             "function {name}({params}): {return_type} {{\n"
             "{body}\n"
             "}}\n"
         ),
         "class": (
             "class {name} {{\n"
             "    constructor({params}) {{\n"
             "        {body}\n"
             "    }}\n"
             "}}\n"
         ),
         "main": "{body}\n",
     },
),

# ── C ─────────────────────────────────────────────────────────────
_reg("C", ".c", "//",
     ["int", "char", "float", "double", "void", "long", "short", "unsigned",
      "struct", "enum", "typedef", "sizeof", "return", "if", "else", "for",
      "while", "do", "switch", "case", "break", "continue", "goto",
      "const", "static", "extern", "volatile", "register", "NULL", "true", "false"],
     {
         "function": (
             "{return_type} {name}({params}) {{\n"
             "{body}\n"
             "}}\n"
         ),
         "struct": (
             "typedef struct {{\n"
             "    {body}\n"
             "}} {name};\n"
         ),
         "main": (
             "int main(int argc, char *argv[]) {{\n"
             "    {body}\n"
             "    return 0;\n"
             "}}\n"
         ),
     },
     "#include <stdio.h>\n#include <stdlib.h>\n\n"
),

# ── C++ ──────────────────────────────────────────────────────────
_reg("C++", ".cpp", "//",
     ["int", "char", "float", "double", "void", "long", "auto", "const",
      "class", "struct", "template", "namespace", "using", "new", "delete",
      "return", "if", "else", "for", "while", "do", "switch", "case",
      "break", "continue", "try", "catch", "throw", "virtual", "override",
      "public", "private", "protected", "static", "nullptr", "true", "false",
      "std", "string", "vector", "map", "set", "cout", "cin", "endl"],
     {
         "function": (
             "{return_type} {name}({params}) {{\n"
             "{body}\n"
             "}}\n"
         ),
         "class": (
             "class {name} {{\n"
             "public:\n"
             "    {name}({params}) {{\n"
             "        {body}\n"
             "    }}\n"
             "}};\n"
         ),
         "main": (
             "int main() {{\n"
             "    {body}\n"
             "    return 0;\n"
             "}}\n"
         ),
     },
     "#include <iostream>\n#include <string>\n#include <vector>\n\n"
),

# ── Java ──────────────────────────────────────────────────────────
_reg("Java", ".java", "//",
     ["public", "private", "protected", "static", "final", "abstract", "class",
      "interface", "extends", "implements", "new", "this", "super", "return",
      "if", "else", "for", "while", "do", "switch", "case", "break", "continue",
      "try", "catch", "finally", "throw", "throws", "void", "int", "long",
      "double", "float", "boolean", "char", "String", "null", "true", "false"],
     {
         "function": (
             "public {return_type} {name}({params}) {{\n"
             "{body}\n"
             "}}\n"
         ),
         "class": (
             "public class {name} {{\n"
             "    public {name}({params}) {{\n"
             "        {body}\n"
             "    }}\n"
             "}}\n"
         ),
         "main": (
             "public static void main(String[] args) {{\n"
             "    {body}\n"
             "}}\n"
         ),
     },
     "public class Main {\n"
),

# ── Go ───────────────────────────────────────────────────────────
_reg("Go", ".go", "//",
     ["func", "package", "import", "return", "if", "else", "for", "range",
      "switch", "case", "default", "break", "continue", "go", "chan", "select",
      "struct", "interface", "map", "var", "const", "type", "defer", "nil",
      "true", "false", "fmt", "int", "string", "bool", "float64", "error"],
     {
         "function": (
             "func {name}({params}) {return_type} {{\n"
             "{body}\n"
             "}}\n"
         ),
         "struct": (
             "type {name} struct {{\n"
             "    {body}\n"
             "}}\n"
         ),
         "main": (
             "func main() {{\n"
             "    {body}\n"
             "}}\n"
         ),
     },
     "package main\n\nimport \"fmt\"\n\n"
),

# ── Rust ──────────────────────────────────────────────────────────
_reg("Rust", ".rs", "//",
     ["fn", "let", "mut", "pub", "struct", "enum", "impl", "trait", "use",
      "mod", "return", "if", "else", "for", "while", "loop", "match", "break",
      "continue", "self", "Self", "super", "crate", "async", "await",
      "move", "ref", "dyn", "where", "type", "const", "static", "unsafe",
      "true", "false", "Some", "None", "Ok", "Err", "String", "Vec", "Box"],
     {
         "function": (
             "fn {name}({params}) -> {return_type} {{\n"
             "{body}\n"
             "}}\n"
         ),
         "struct": (
             "struct {name} {{\n"
             "    {body}\n"
             "}}\n"
         ),
         "main": (
             "fn main() {{\n"
             "    {body}\n"
             "}}\n"
         ),
     },
),

# ── PHP ──────────────────────────────────────────────────────────
_reg("PHP", ".php", "//",
     ["function", "class", "extends", "implements", "new", "this", "self",
      "return", "if", "else", "elseif", "for", "foreach", "while", "do",
      "switch", "case", "break", "continue", "try", "catch", "finally",
      "throw", "public", "private", "protected", "static", "abstract",
      "interface", "trait", "namespace", "use", "echo", "print", "var",
      "const", "array", "null", "true", "false", "None"],
     {
         "function": (
             "<?php\n\n/** TODO: description */\n"
             "function {name}({params}) {{\n"
             "{body}\n"
             "}}\n"
         ),
         "class": (
             "<?php\n\nclass {name} {{\n"
             "    public function __construct({params}) {{\n"
             "        {body}\n"
             "    }}\n"
             "}}\n"
         ),
         "main": "<?php\n\n{body}\n",
     },
     "<?php\n"
),

# ── Ruby ──────────────────────────────────────────────────────────
_reg("Ruby", ".rb", "#",
     ["def", "class", "module", "end", "if", "elsif", "else", "unless",
      "while", "until", "for", "do", "break", "next", "return", "yield",
      "lambda", "proc", "begin", "rescue", "ensure", "raise", "require",
      "include", "extend", "attr_reader", "attr_writer", "attr_accessor",
      "self", "super", "nil", "true", "false", "puts", "print", "gets"],
     {
         "function": (
             "# TODO: description\n"
             "def {name}({params})\n"
             "{body}\n"
             "end\n"
         ),
         "class": (
             "class {name}\n"
             "    def initialize({params})\n"
             "        {body}\n"
             "    end\n"
             "end\n"
         ),
         "main": "{body}\n",
     },
),

# ── Swift ─────────────────────────────────────────────────────────
_reg("Swift", ".swift", "//",
     ["func", "class", "struct", "enum", "protocol", "extension", "import",
      "return", "if", "else", "for", "while", "repeat", "switch", "case",
      "default", "break", "continue", "guard", "defer", "try", "catch",
      "throw", "throws", "async", "await", "let", "var", "inout",
      "public", "private", "internal", "static", "mutating",
      "true", "false", "nil", "print", "String", "Int", "Double", "Bool", "Array"],
     {
         "function": (
             "/// TODO: description\n"
             "func {name}({params}) -> {return_type} {{\n"
             "{body}\n"
             "}}\n"
         ),
         "class": (
             "class {name} {{\n"
             "    init({params}) {{\n"
             "        {body}\n"
             "    }}\n"
             "}}\n"
         ),
         "main": "{body}\n",
     },
),

# ── Kotlin ────────────────────────────────────────────────────────
_reg("Kotlin", ".kt", "//",
     ["fun", "class", "interface", "object", "data", "sealed", "enum",
      "val", "var", "return", "if", "else", "when", "for", "while", "do",
      "break", "continue", "try", "catch", "finally", "throw", "is", "as",
      "in", "by", "lazy", "companion", "private", "public", "protected",
      "internal", "override", "open", "abstract", "annotation",
      "true", "false", "null", "println", "String", "Int", "Double", "Boolean", "List", "Map"],
     {
         "function": (
             "// TODO: description\n"
             "fun {name}({params}): {return_type} {{\n"
             "{body}\n"
             "}}\n"
         ),
         "class": (
             "class {name}({params}) {{\n"
             "    {body}\n"
             "}}\n"
         ),
         "main": (
             "fun main() {{\n"
             "    {body}\n"
             "}}\n"
         ),
     },
),

# ── C# ────────────────────────────────────────────────────────────
_reg("C#", ".cs", "//",
     ["class", "struct", "interface", "enum", "namespace", "using", "public",
      "private", "protected", "internal", "static", "virtual", "override",
      "abstract", "sealed", "new", "this", "base", "return", "if", "else",
      "for", "foreach", "while", "do", "switch", "case", "break", "continue",
      "try", "catch", "finally", "throw", "async", "await", "var", "val",
      "int", "string", "bool", "double", "float", "decimal", "void",
      "null", "true", "false", "Console", "List", "Dictionary"],
     {
         "function": (
             "/// TODO: description\n"
             "public {return_type} {name}({params}) {{\n"
             "{body}\n"
             "}}\n"
         ),
         "class": (
             "public class {name} {{\n"
             "    public {name}({params}) {{\n"
             "        {body}\n"
             "    }}\n"
             "}}\n"
         ),
         "main": (
             "static void Main(string[] args) {{\n"
             "    {body}\n"
             "}}\n"
         ),
     },
),

# ── Lua ──────────────────────────────────────────────────────────
_reg("Lua", ".lua", "--",
     ["function", "end", "local", "return", "if", "then", "else", "elseif",
      "for", "while", "do", "repeat", "until", "break", "continue",
      "in", "pairs", "ipairs", "next", "select", "pcall", "xpcall",
      "error", "assert", "type", "tostring", "tonumber", "print",
      "require", "module", "nil", "true", "false", "self"],
     {
         "function": (
             "-- TODO: description\n"
             "function {name}({params})\n"
             "{body}\n"
             "end\n"
         ),
         "class": (
             "{name} = {{}}\n"
             "function {name}:new({params})\n"
             "    local obj = {{}}\n"
             "    setmetatable(obj, {{__index = {name}}})\n"
             "    {body}\n"
             "    return obj\n"
             "end\n"
         ),
         "main": "{body}\n",
     },
),

# ── Haskell ──────────────────────────────────────────────────────
_reg("Haskell", ".hs", "--",
     ["where", "let", "in", "if", "then", "else", "case", "of", "data",
      "type", "newtype", "class", "instance", "deriving", "import",
      "module", "do", "return", "IO", "Maybe", "Either", "True", "False",
      "String", "Int", "Float", "Double", "Bool", "Char", "Int", "Integer"],
     {
         "function": (
             "-- TODO: description\n"
             "{name} :: {type_sig}\n"
             "{name} {params} =\n"
             "{body}\n"
         ),
         "data_type": (
             "data {name} =\n"
             "    {body}\n"
             "    deriving (Show, Eq)\n"
         ),
         "main": (
             "main :: IO ()\n"
             "main = do\n"
             "    {body}\n"
         ),
     },
     "-- Language: Haskell\n\n"
),

# ── SQL ──────────────────────────────────────────────────────────
_reg("SQL", ".sql", "--",
     ["SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES", "UPDATE",
      "SET", "DELETE", "CREATE", "TABLE", "ALTER", "DROP", "INDEX",
      "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "ON", "AND", "OR",
      "NOT", "NULL", "IS", "IN", "BETWEEN", "LIKE", "ORDER", "BY",
      "GROUP", "HAVING", "LIMIT", "OFFSET", "AS", "DISTINCT", "COUNT",
      "SUM", "AVG", "MIN", "MAX", "UNION", "ALL", "EXISTS", "CASE",
      "WHEN", "THEN", "ELSE", "END", "PRIMARY", "KEY", "FOREIGN",
      "REFERENCES", "CONSTRAINT", "DEFAULT", "AUTO_INCREMENT", "VARCHAR",
      "INTEGER", "TEXT", "BOOLEAN", "DATE", "TIMESTAMP", "FLOAT", "DOUBLE"],
     {
         "select": (
             "SELECT {columns}\nFROM {table}\nWHERE {condition};\n"
         ),
         "insert": (
             "INSERT INTO {table} ({columns})\nVALUES ({values});\n"
         ),
         "create_table": (
             "CREATE TABLE {name} (\n{columns}\n);\n"
         ),
     },
),

# ── Shell/Bash ───────────────────────────────────────────────────
_reg("Bash", ".sh", "#",
     ["echo", "if", "then", "else", "elif", "fi", "for", "while", "do",
      "done", "case", "esac", "function", "return", "exit", "local",
      "export", "source", "alias", "unalias", "set", "unset", "shift",
      "read", "test", "[", "[[", "expr", "let", "declare", "readonly",
      "true", "false", "in", "select"],
     {
         "function": (
             "#!/bin/bash\n\n"
             "# TODO: description\n"
             "{name}() {{\n"
             "{body}\n"
             "}}\n"
         ),
         "main": (
             "#!/bin/bash\n\n"
             "{body}\n"
         ),
     },
     "#!/bin/bash\nset -euo pipefail\n\n"
),

# ── YAML ─────────────────────────────────────────────────────────
_reg("YAML", ".yaml", "#",
     ["key", "value", "true", "false", "null", "yes", "no"],
     {
         "document": "{body}\n",
     },
),

# ── JSON ─────────────────────────────────────────────────────────
_reg("JSON", ".json", "",
     ["true", "false", "null"],
     {
         "object": "{{\n{body}\n}}\n",
     },
),

# ── Markdown ─────────────────────────────────────────────────────
_reg("Markdown", ".md", "",
     ["#", "##", "###", "**", "*", "`", "```", "-", "[", "]", "(", ")"],
     {
         "document": "# {title}\n\n{body}\n",
     },
),

# ── HTML ─────────────────────────────────────────────────────────
_reg("HTML", ".html", "",
     ["html", "head", "body", "div", "span", "p", "a", "h1", "h2", "h3",
      "ul", "ol", "li", "table", "tr", "td", "th", "form", "input",
      "button", "script", "style", "link", "meta", "title", "img"],
     {
         "page": (
             "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
             "    <meta charset=\"UTF-8\">\n"
             "    <title>{title}</title>\n"
             "</head>\n<body>\n{body}\n</body>\n</html>\n"
         ),
     },
),

# ── CSS ──────────────────────────────────────────────────────────
_reg("CSS", ".css", "",
     ["color", "background", "margin", "padding", "border", "display",
      "position", "width", "height", "font", "text", "flex", "grid",
      "transition", "animation", "transform", "media", "hover"],
     {
         "rule": (
             "{selector} {{\n{properties}\n}}\n"
         ),
     },
),

# ── Regex helpers ────────────────────────────────────────────────
ALIAS_MAP = {
    "py": "python", "python3": "python", "py3": "python",
    "js": "javascript", "node": "javascript", "nodejs": "javascript",
    "ts": "typescript",
    "c++": "c++", "cpp": "c++", "cxx": "c++",
    "c#": "c#", "cs": "c#", "csharp": "c#",
    "rb": "ruby",
    "kt": "kotlin",
    "golang": "go",
    "sh": "bash", "shell": "bash", "zsh": "bash",
    "yml": "yaml",
    "md": "markdown",
    "htm": "html",
    "hs": "haskell",
    "sql": "sql",
    "css": "css",
    "lua": "lua",
    "swift": "swift",
    "rs": "rust",
}

TASK_KEYWORD_MAP = {
    "python": ["python", "пайтон", "питон"],
    "javascript": ["javascript", "js", "джаваскрипт", "джс", ".node"],
    "typescript": ["typescript", "ts", "тайпскрипт"],
    "c": ["на си", " на c ", "c lang", "ansi c"],
    "c++": ["c++", "плюсы", "cpp"],
    "java": ["java", "джава"],
    "go": ["go", "голанг", "го ", "golang"],
    "rust": ["rust", "раст"],
    "php": ["php", "пхп"],
    "ruby": ["ruby", "руби"],
    "swift": ["swift", "свифт"],
    "kotlin": ["kotlin", "котлин", "котлиниум"],
    "c#": ["c#", "csharp", "шарп"],
    "lua": ["lua", "луа"],
    "haskell": ["haskell", "хаскель"],
    "sql": ["sql", "запрос", "таблиц"],
    "bash": ["bash", "shell", "шелл", "скрипт"],
    "yaml": ["yaml", "yml", "конфиг"],
    "json": ["json", "джейсон"],
    "html": ["html", "страниц", "сайт", "веб-страниц"],
    "css": ["css", "стил", "дизайн"],
    "markdown": ["markdown", "md", "разметк"],
}


class LanguageRegistry:
    @staticmethod
    def get_all() -> Dict[str, LanguageInfo]:
        return dict(LANGUAGES)

    @staticmethod
    def get(name: str) -> Optional[LanguageInfo]:
        key = name.lower().strip()
        key = ALIAS_MAP.get(key, key)
        return LANGUAGES.get(key)

    @staticmethod
    def supported() -> List[str]:
        return sorted(LANGUAGES.keys())

    @staticmethod
    def detect_language(task: str) -> Optional[str]:
        task_lower = task.lower()
        best = None
        best_score = 0
        for lang, keywords in TASK_KEYWORD_MAP.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            if score > best_score:
                best_score = score
                best = lang
        return best

    @staticmethod
    def ext_to_lang(ext: str) -> Optional[str]:
        ext = ext.lower().strip().lstrip('.')
        for lang, info in LANGUAGES.items():
            if info.ext.lstrip('.') == ext:
                return lang
        return None
