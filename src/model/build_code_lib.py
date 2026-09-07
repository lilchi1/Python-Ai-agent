#!/usr/bin/env python3
"""
Строит компактную "библиотеку кода" — небольшой набор канонических
Python-сниппетов, повторённых многократно, чтобы небольшая модель
(Mini-GPT, ~496K params) смогла их ЗАПОМНИТЬ и воспроизводить.

Идея: для маленькой локальной модели проще выучить ограниченный,
но качественный набор паттернов, чем свободный текст. Так модель
становится настоящей "нейропамятью" агента для кода.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'data', 'lib')
os.makedirs(DATA, exist_ok=True)

LIBRARY = {
    "def factorial(n):":"def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n",
    "def fib(n):":"def fib(n):\n    if n <= 0:\n        return 0\n    if n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n",
    "def is_prime(n):":"def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n",
    "def gcd(a, b):":"def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
    "def lcm(a, b):":"def lcm(a, b):\n    return abs(a * b) // gcd(a, b) if a and b else 0\n",
    "def reverse_string(s):":"def reverse_string(s):\n    return s[::-1]\n",
    "def is_palindrome(s):":"def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]\n",
    "def sort_list(arr):":"def sort_list(arr):\n    return sorted(arr)\n",
    "def find_max(arr):":"def find_max(arr):\n    if not arr:\n        return None\n    return max(arr)\n",
    "def find_min(arr):":"def find_min(arr):\n    if not arr:\n        return None\n    return min(arr)\n",
    "def sum_list(arr):":"def sum_list(arr):\n    if not arr:\n        return 0\n    return sum(arr)\n",
    "def average(arr):":"def average(arr):\n    if not arr:\n        return 0\n    return sum(arr) / len(arr)\n",
    "def binary_search(arr, target):":"def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
    "def fizzbuzz(n):":"def fizzbuzz(n):\n    result = []\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            result.append('FizzBuzz')\n        elif i % 3 == 0:\n            result.append('Fizz')\n        elif i % 5 == 0:\n            result.append('Buzz')\n        else:\n            result.append(str(i))\n    return result\n",
    "def bubble_sort(arr):":"def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr\n",
    "class Stack:":"class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n        self.items.append(item)\n\n    def pop(self):\n        if self.is_empty():\n            return None\n        return self.items.pop()\n\n    def is_empty(self):\n        return len(self.items) == 0\n\n    def size(self):\n        return len(self.items)\n",
    "def count_vowels(s):":"def count_vowels(s):\n    count = 0\n    for ch in s.lower():\n        if ch in 'aeiou':\n            count += 1\n    return count\n",
    "def remove_duplicates(arr):":"def remove_duplicates(arr):\n    seen = set()\n    result = []\n    for x in arr:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result\n",
    "def caesar_cipher(text, shift):":"def caesar_cipher(text, shift):\n    result = ''\n    for ch in text:\n        if ch.isalpha():\n            base = ord('A') if ch.isupper() else ord('a')\n            result += chr((ord(ch) - base + shift) % 26 + base)\n        else:\n            result += ch\n    return result\n",
}

def build(repeats=12):
    parts = []
    for _ in range(repeats):
        for name, code in LIBRARY.items():
            parts.append(code)
    text = "\n".join(parts)
    out = os.path.join(DATA, 'code_lib.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Code library: {len(LIBRARY)} snippets x {repeats} reps = {len(text)} chars -> {out}")
    return out


def build_pairs(repeats=18):
    """Собирает ПАРНЫЙ датасет (signature -> body), повторяя его многократно,
    чтобы модель сильно выучила связь 'сигнатура -> тело функции'.

    Структура каждой пары:
        подсказка\n Тело функции
    Модель учится продолжать текст сразу после сигнатуры."""
    parts = []
    for _ in range(repeats):
        for name, code in LIBRARY.items():
            # сигнатура + тело (без повторного имени на сигнатуре)
            sig = name + "\n"
            parts.append(code)
    text = "\n".join(parts)
    out = os.path.join(DATA, 'code_pairs.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Code pairs: {len(LIBRARY)} x {repeats} = {len(text)} chars -> {out}")
    return out


if __name__ == '__main__':
    build()
    build_pairs()
