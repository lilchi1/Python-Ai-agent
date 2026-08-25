#!/usr/bin/env python3
"""
Саморазвивающийся мультиязычный агент для генерации кода.
Без внешних зависимостей — только стандартная библиотека Python.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.memory.vector_store import LocalVectorStore
from src.agent.coding_agent import CodingAgent
from src.orchestrator import Orchestrator
from src.languages.registry import LanguageRegistry
from src.learner.self_educator import SelfEducator

DIVIDER = "=" * 60


def print_banner():
    print(DIVIDER)
    print("  SELF-DEV CODE AGENT")
    print("  Python 3.x | No external dependencies")
    print(DIVIDER)
    print("  Commands:")
    print("    /lang              - list supported languages")
    print("    /stats             - show statistics")
    print("    /history           - show recent tasks")
    print("    /search <q>        - search archive")
    print("    /init              - seed knowledge base")
    print("    /learn [depth]     - start self-learning from internet")
    print("    /learn-stop        - stop learning (saves progress)")
    print("    /learn-clear       - reset learning progress")
    print("    /help              - this help")
    print("    /quit              - exit")
    print(DIVIDER)


def init_knowledge(vector_store: LocalVectorStore):
    examples = [
        "Task: Factorial of a number\nLanguage: python\nSolution:\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
        "Task: Check if number is prime\nLanguage: python\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
        "Task: Sort a list of numbers\nLanguage: python\ndef sort_list(arr):\n    return sorted(arr)",
        "Task: Find maximum element\nLanguage: python\ndef find_max(arr):\n    if not arr:\n        return None\n    return max(arr)",
        "Task: Reverse a string\nLanguage: python\ndef reverse_string(s):\n    return s[::-1]",
        "Task: Fibonacci number\nLanguage: python\ndef fibonacci(n):\n    if n <= 0: return 0\n    if n == 1: return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b",
        "Task: Sum of list elements\nLanguage: python\ndef sum_list(arr):\n    return sum(arr)",
        "Task: Check palindrome\nLanguage: python\ndef is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]",
        "Task: Binary search\nLanguage: javascript\nfunction binarySearch(arr, target) {\n    let lo = 0, hi = arr.length - 1;\n    while (lo <= hi) {\n        const mid = (lo + hi) >> 1;\n        if (arr[mid] === target) return mid;\n        else if (arr[mid] < target) lo = mid + 1;\n        else hi = mid - 1;\n    }\n    return -1;\n}",
        "Task: Factorial in C\nLanguage: c\nint factorial(int n) {\n    if (n <= 1) return 1;\n    return n * factorial(n - 1);\n}",
        "Task: Is prime in Go\nLanguage: go\nfunc isPrime(n int) bool {\n    if n < 2 { return false }\n    for i := 2; i*i <= n; i++ {\n        if n%i == 0 { return false }\n    }\n    return true\n}",
        "Task: Factorial in Rust\nLanguage: rust\nfn factorial(n: i32) -> i32 {\n    if n <= 1 { 1 } else { n * factorial(n - 1) }\n}",
        "Task: Reverse string in Java\nLanguage: java\npublic static String reverseString(String s) {\n    return new StringBuilder(s).reverse().toString();\n}",
        "Task: FizzBuzz\nLanguage: python\ndef fizzbuzz(n):\n    for i in range(1, n + 1):\n        if i % 15 == 0: yield 'FizzBuzz'\n        elif i % 3 == 0: yield 'Fizz'\n        elif i % 5 == 0: yield 'Buzz'\n        else: yield str(i)",
        "Task: GCD in Kotlin\nLanguage: kotlin\nfun gcd(a: Int, b: Int): Int {\n    var x = a; var y = b\n    while (y != 0) { val t = x % y; x = y; y = t }\n    return x\n}",
        "Task: Sum list in Haskell\nLanguage: haskell\nsumList :: [Int] -> Int\nsumList = sum",
        "Task: Binary search in C++\nLanguage: c++\nint binarySearch(std::vector<int> arr, int target) {\n    int lo = 0, hi = arr.size() - 1;\n    while (lo <= hi) {\n        int mid = lo + (hi - lo) / 2;\n        if (arr[mid] == target) return mid;\n        else if (arr[mid] < target) lo = mid + 1;\n        else hi = mid - 1;\n    }\n    return -1;\n}",
        "Task: Find max in Ruby\nLanguage: ruby\ndef find_max(arr)\n    arr.max\nend",
        "Task: Fibonacci in Lua\nLanguage: lua\nfunction fibonacci(n)\n    if n <= 0 then return 0 end\n    if n == 1 then return 1 end\n    local a, b = 0, 1\n    for i = 2, n do a, b = b, a + b end\n    return b\nend",
    ]
    vector_store.add_documents(examples)
    print(f"  Added {len(examples)} examples to knowledge base")


def main():
    print_banner()

    vector_store = LocalVectorStore()
    agent = CodingAgent(vector_store, max_attempts=3)
    orchestrator = Orchestrator(agent)

    print("  Ready. Type a task or /help for commands.\n")

    try:
        while True:
            try:
                raw = input("task> ").strip()
            except EOFError:
                break

            if not raw:
                continue

            if raw.startswith('/'):
                cmd = raw.lower().split()
                name = cmd[0]

                if name == '/quit' or name == '/exit' or name == '/q':
                    print("  Bye!")
                    break

                elif name == '/help':
                    print_banner()

                elif name == '/lang':
                    langs = LanguageRegistry.supported()
                    print(f"\n  Supported languages ({len(langs)}):")
                    for l in langs:
                        info = LanguageRegistry.get(l)
                        print(f"    {info.name:12s}  {info.ext}")

                elif name == '/stats':
                    stats = orchestrator.get_stats()
                    print(f"\n  Tasks: {stats['total_tasks']}  |  Success: {stats['successful']}  |  Rate: {stats['success_rate']*100:.0f}%")
                    if stats['languages_used']:
                        print("  Languages used:")
                        for lang, count in sorted(stats['languages_used'].items(), key=lambda x: -x[1]):
                            print(f"    {lang}: {count}")

                elif name == '/history':
                    n = int(cmd[1]) if len(cmd) > 1 else 10
                    hist = orchestrator.get_history(n)
                    if not hist:
                        print("  No history yet.")
                    else:
                        for h in hist:
                            r = h['result']
                            ok = "OK" if r.get('success') else "FAIL"
                            lang = r.get('language', '?')
                            print(f"  [{ok}] ({lang}) {h['task'][:60]}")

                elif name == '/search':
                    query = " ".join(cmd[1:])
                    results = orchestrator.search_archive(query)
                    if not results:
                        print("  Nothing found.")
                    else:
                        for h in results[-5:]:
                            r = h['result']
                            print(f"  [{r.get('language','?')}] {h['task'][:60]}")

                elif name == '/init':
                    init_knowledge(vector_store)

                elif name == '/learn':
                    depth = int(cmd[1]) if len(cmd) > 1 else None
                    educator = SelfEducator(vector_store)
                    educator.start(depth=depth)

                elif name == '/learn-stop':
                    print("  Остановка не поддерживается из CLI — используйте Ctrl+C во время /learn")
                    print("  Прогресс автоматически сохраняется при прерывании")

                elif name == '/learn-clear':
                    educator = SelfEducator(vector_store)
                    educator.clear_progress()

                else:
                    print(f"  Unknown command: {name}. Type /help")

            else:
                result = orchestrator.execute(raw)
                print()
                if result['success']:
                    print(DIVIDER)
                    print(f"  RESULT ({result['language'].upper()})")
                    print(DIVIDER)
                    print(result['code'])
                    print(DIVIDER)
                else:
                    print(f"  FAILED: {result['feedback']}")
                    if result.get('code'):
                        print(f"  Code ({result['attempts']} attempts):")
                        print(result['code'])
                print()

    except KeyboardInterrupt:
        print("\n  Bye!")

    stats = orchestrator.get_stats()
    print(f"\n  Session: {stats['total_tasks']} tasks, {stats['successful']} successful ({stats['success_rate']*100:.0f}%)")


if __name__ == "__main__":
    main()
