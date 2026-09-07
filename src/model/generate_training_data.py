#!/usr/bin/env python3
"""
Генератор данных для обучения модели Mini-GPT.
Создаёт тысячи примеров кода на Python для тренировки языковой модели.
"""
import os
import random
import json

# ══════════════════════════════════════════════════════════════
#  ШАБЛОНЫ КОДА
# ══════════════════════════════════════════════════════════════

FUNCTION_TEMPLATES = [
    # Factorial
    """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
""",

    # Factorial iterative
    """def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
""",

    # Fibonacci
    """def fibonacci(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
""",

    # Fibonacci list
    """def fibonacci_list(n):
    if n <= 0:
        return []
    if n == 1:
        return [0]
    seq = [0, 1]
    for i in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq
""",

    # Is prime
    """def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
""",

    # Sum
    """def sum_list(arr):
    if not arr:
        return 0
    return sum(arr)
""",

    # Max
    """def find_max(arr):
    if not arr:
        return None
    result = arr[0]
    for x in arr[1:]:
        if x > result:
            result = x
    return result
""",

    # Min
    """def find_min(arr):
    if not arr:
        return None
    result = arr[0]
    for x in arr[1:]:
        if x < result:
            result = x
    return result
""",

    # Reverse string
    """def reverse_string(s):
    return s[::-1]
""",

    # Reverse string loop
    """def reverse_string(s):
    result = ""
    for ch in s:
        result = ch + result
    return result
""",

    # Palindrome
    """def is_palindrome(s):
    s = s.lower().replace(' ', '')
    return s == s[::-1]
""",

    # Sort
    """def sort_list(arr):
    return sorted(arr)
""",

    # Bubble sort
    """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
""",

    # Binary search
    """def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
""",

    # GCD
    """def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
""",

    # LCM
    """def lcm(a, b):
    return abs(a * b) // gcd(a, b) if a and b else 0
""",

    # FizzBuzz
    """def fizzbuzz(n):
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result
""",

    # Count vowels
    """def count_vowels(s):
    count = 0
    for ch in s.lower():
        if ch in 'aeiou':
            count += 1
    return count
""",

    # Count consonants
    """def count_consonants(s):
    count = 0
    for ch in s.lower():
        if ch.isalpha() and ch not in 'aeiou':
            count += 1
    return count
""",

    # Is even
    """def is_even(n):
    return n % 2 == 0
""",

    # Absolute value
    """def abs_val(n):
    return n if n >= 0 else -n
""",

    # Power
    """def power(base, exp):
    if exp == 0:
        return 1
    if exp < 0:
        return 1.0 / power(base, -exp)
    result = 1
    for _ in range(exp):
        result *= base
    return result
""",

    # Square root
    """def sqrt_approx(n, precision=0.00001):
    if n < 0:
        return None
    if n == 0:
        return 0
    guess = n / 2.0
    while abs(guess * guess - n) > precision:
        guess = (guess + n / guess) / 2.0
    return guess
""",

    # Average
    """def average(arr):
    if not arr:
        return 0
    return sum(arr) / len(arr)
""",

    # Median
    """def median(arr):
    if not arr:
        return 0
    sorted_arr = sorted(arr)
    n = len(sorted_arr)
    if n % 2 == 1:
        return sorted_arr[n // 2]
    return (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2
""",

    # Mode
    """def mode(arr):
    if not arr:
        return None
    freq = {}
    for x in arr:
        freq[x] = freq.get(x, 0) + 1
    return max(freq, key=freq.get)
""",

    # Flatten list
    """def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
""",

    # Remove duplicates
    """def remove_duplicates(arr):
    seen = set()
    result = []
    for x in arr:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result
""",

    # Is anagram
    """def is_anagram(s1, s2):
    return sorted(s1.lower().replace(' ', '')) == sorted(s2.lower().replace(' ', ''))
""",

    # Caesar cipher
    """def caesar_cipher(text, shift):
    result = ""
    for ch in text:
        if ch.isalpha():
            base = ord('A') if ch.isupper() else ord('a')
            result += chr((ord(ch) - base + shift) % 26 + base)
        else:
            result += ch
    return result
""",

    # Matrix transpose
    """def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
""",

    # Matrix multiply
    """def matrix_multiply(a, b):
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    if cols_a != rows_b:
        raise ValueError("Cannot multiply")
    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result
""",

    # Hanoi towers
    """def hanoi(n, source, target, auxiliary):
    if n == 1:
        print(f"Move disk 1 from {source} to {target}")
        return
    hanoi(n - 1, source, auxiliary, target)
    print(f"Move disk {n} from {source} to {target}")
    hanoi(n - 1, auxiliary, target, source)
""",

    # Quick sort
    """def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
""",

    # Merge sort
    """def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
""",

    # Stack
    """class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        return None

    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        return None

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)
""",

    # Queue
    """class Queue:
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        return None

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)
""",

    # Linked list
    """class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def find(self, data):
        current = self.head
        while current:
            if current.data == data:
                return True
            current = current.next
        return False
""",

    # Binary tree
    """class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def insert(self, value):
        if value < self.value:
            if self.left is None:
                self.left = TreeNode(value)
            else:
                self.left.insert(value)
        else:
            if self.right is None:
                self.right = TreeNode(value)
            else:
                self.right.insert(value)

    def inorder(self):
        result = []
        if self.left:
            result.extend(self.left.inorder())
        result.append(self.value)
        if self.right:
            result.extend(self.right.inorder())
        return result
""",

    # Hash table
    """class HashTable:
    def __init__(self, size=100):
        self.size = size
        self.table = [[] for _ in range(size)]

    def _hash(self, key):
        return hash(key) % self.size

    def set(self, key, value):
        idx = self._hash(key)
        for pair in self.table[idx]:
            if pair[0] == key:
                pair[1] = value
                return
        self.table[idx].append([key, value])

    def get(self, key):
        idx = self._hash(key)
        for pair in self.table[idx]:
            if pair[0] == key:
                return pair[1]
        return None
""",

    # Graph BFS
    """from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    order = []

    while queue:
        vertex = queue.popleft()
        order.append(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order
""",

    # Graph DFS
    """def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    order = [start]
    for neighbor in graph[start]:
        if neighbor not in visited:
            order.extend(dfs(graph, neighbor, visited))
    return order
""",

    # Dijkstra
    """import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    queue = [(0, start)]

    while queue:
        dist, node = heapq.heappop(queue)
        if dist > distances[node]:
            continue
        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(queue, (new_dist, neighbor))
    return distances
""",

    # Knapsack
    """def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i-1][w]
            if weights[i-1] <= w:
                val = dp[i-1][w - weights[i-1]] + values[i-1]
                if val > dp[i][w]:
                    dp[i][w] = val
    return dp[n][capacity]
""",

    # LRU Cache
    """from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return -1

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
""",
]

# ══════════════════════════════════════════════════════════════
#  ДОПОЛНИТЕЛЬНЫЕ ПАТТЕРНЫ
# ══════════════════════════════════════════════════════════════

VARIABLE_NAMES = ['x', 'y', 'n', 'i', 'j', 'k', 'val', 'idx', 'count', 'result', 'data', 'item', 'num', 'arr', 'lst']
OPERATORS = ['+', '-', '*', '//', '%', '**']
COMPARISONS = ['<', '>', '<=', '>=', '==', '!=']
LOOPS_FOR = [
    "for i in range({n}):",
    "for i in range({n}, 0, -1):",
    "for i in range(0, {n}, 2):",
    "for j in range({n}):",
    "for x in range({n}):",
    "for _ in range({n}):",
]
LOOPS_WHILE = [
    "while {cond}:",
    "while n > 0:",
    "while i < {n}:",
    "while True:",
]
IF_ELSE = [
    "if {cond}:\n    {body}",
    "if {cond}:\n    {body}\nelse:\n    {body2}",
    "if {cond}:\n    {body}\nelif {cond2}:\n    {body}\nelse:\n    {body2}",
]
RETURN_STATEMENTS = [
    "return {val}",
    "return {val} + {val2}",
    "return {val} * {val2}",
    "return True",
    "return False",
    "return None",
    "return []",
    "return {{}}",
    "return \"\"",
    "return 0",
    "return 1",
    "return -1",
]
FUNCTION_DEF = [
    "def {name}({params}):\n    {body}",
    "def {name}(self, {params}):\n    {body}",
]

# ══════════════════════════════════════════════════════════════
#  ГЕНЕРАТОРЫ
# ══════════════════════════════════════════════════════════════

def gen_random_function():
    name = random.choice(['calculate', 'process', 'compute', 'evaluate', 'transform', 'analyze', 'optimize', 'validate', 'transform', 'perform'])
    params = random.choice(['x', 'n', 'arr', 'data', 'items', 'values', 'a, b', 'x, y'])
    body_lines = []

    body_lines.append(f"    result = 0")
    body_lines.append(f"    for i in range(n):")
    op = random.choice(OPERATORS)
    body_lines.append(f"        result = result {op} i")
    body_lines.append(f"    return result")

    body = "\n".join(body_lines)
    return f"def {name}({params}):\n{body}\n"


def gen_random_class():
    name = random.choice(['Processor', 'Manager', 'Handler', 'Builder', 'Factory', 'Cache', 'Store', 'Queue', 'Pool', 'Wrapper'])
    attrs = random.choice(['data', 'items', 'cache', 'buffer', 'queue', 'stack', 'registry', 'handlers', 'callbacks'])

    code = f"""class {name}:
    def __init__(self):
        self.{attrs} = []

    def add(self, item):
        self.{attrs}.append(item)

    def get_all(self):
        return list(self.{attrs})

    def clear(self):
        self.{attrs} = []

    def size(self):
        return len(self.{attrs})

"""
    return code


def gen_random_loop():
    var = random.choice(VARIABLE_NAMES)
    n = random.choice([10, 20, 50, 100])
    loop = random.choice(LOOPS_FOR).format(n=n)
    op = random.choice(OPERATORS)

    body = f"    result = 0\n    {loop}\n        result = result {op} i\n    return result"

    name = random.choice(['compute', 'sum_range', 'accumulate', 'calculate'])
    return f"def {name}(n={n}):\n{body}\n"


def gen_random_list_ops():
    op = random.choice(['reverse', 'sort', 'filter', 'map', 'unique'])
    if op == 'reverse':
        return f"""def reverse_list(arr):
    result = []
    for i in range(len(arr) - 1, -1, -1):
        result.append(arr[i])
    return result
"""
    elif op == 'sort':
        return f"""def custom_sort(arr):
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
    return arr
"""
    elif op == 'filter':
        return f"""def filter_positive(arr):
    return [x for x in arr if x > 0]
"""
    elif op == 'map':
        return f"""def double_all(arr):
    return [x * 2 for x in arr]
"""
    else:
        return f"""def unique_elements(arr):
    seen = set()
    result = []
    for x in arr:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result
"""


def gen_random_dict_ops():
    return random.choice([
        """def merge_dicts(a, b):
    result = dict(a)
    result.update(b)
    return result
""",
        """def invert_dict(d):
    return {v: k for k, v in d.items()}
""",
        """def count_keys(lst):
    counts = {}
    for item in lst:
        counts[item] = counts.get(item, 0) + 1
    return counts
""",
    ])


def gen_random_string_ops():
    return random.choice([
        """def count_chars(s):
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    return counts
""",
        """def remove_spaces(s):
    return s.replace(' ', '')
""",
        """def capitalize_words(s):
    return ' '.join(word.capitalize() for word in s.split())
""",
        """def is_pangram(s):
    alphabet = set('abcdefghijklmnopqrstuvwxyz')
    return alphabet <= set(s.lower())
""",
    ])


def generate_dataset(output_path, num_samples=5000):
    """Generate training dataset."""
    all_code = []

    print(f"  Generating {num_samples} code samples...")

    # Fixed templates
    for _ in range(len(FUNCTION_TEMPLATES)):
        all_code.append(random.choice(FUNCTION_TEMPLATES))

    # Random templates
    generators = [
        gen_random_function,
        gen_random_class,
        gen_random_loop,
        gen_random_list_ops,
        gen_random_dict_ops,
        gen_random_string_ops,
    ]

    while len(all_code) < num_samples:
        gen = random.choice(generators)
        all_code.append(gen())

    # Mix with imports and docstrings
    prefixes = [
        "#!/usr/bin/env python3\n",
        "\"\"\"Module with various utility functions.\"\"\"\n",
        "import os\nimport sys\nimport math\n",
        "from collections import defaultdict, Counter\n",
        "# Utility functions for data processing\n",
        "",
    ]

    # Create training text by concatenating with delimiters
    sep = "\n\n" + "=" * 60 + "\n\n"
    training_text = sep.join(all_code)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(training_text)

    print(f"  Written {len(all_code)} samples to {output_path}")
    print(f"  Total size: {len(training_text)} characters")
    return training_text


if __name__ == "__main__":
    random.seed(42)
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'training')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'code_corpus.txt')
    text = generate_dataset(output_path, num_samples=3000)

    # Also generate individual files for directory loading
    files_dir = os.path.join(output_dir, 'code_files')
    os.makedirs(files_dir, exist_ok=True)

    for i, template in enumerate(FUNCTION_TEMPLATES):
        fname = os.path.join(files_dir, f"sample_{i:04d}.py")
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(template)

    print(f"\n  Generated {len(FUNCTION_TEMPLATES)} individual code files in {files_dir}")
    print(f"  Training corpus: {output_path}")
    print(f"\n  To compile and train:")
    print(f"    gcc -O2 -o minigpt.exe src/model/mini_gpt.c src/model/main.c -lm")
    print(f"    minigpt.exe train data/training/code_files")
