import sys, os
sys.path.insert(0, '.')
from src.model.minigpt_binding import MiniGPT

m = MiniGPT(model_path='src/model/model.bin', quiet=True)
prompts = ["def factorial(n):", "def is_prime(n):", "def fibonacci(n):",
           "def reverse_string(s):", "def gcd(a, b):", "class Stack:",
           "def is_palindrome(s):", "def sum_list(arr):"]
for p in prompts:
    print('=' * 50)
    print('PROMPT:', p)
    print(m.generate(p, max_tokens=160, temperature=0.5))
    print()
m.close()
