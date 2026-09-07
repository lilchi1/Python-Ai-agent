# Task: Write a function that calculates factorial
# Language: python
# Based on similar solution:
# ---
# Based on similar solution:
# ---
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
# ---
# ---

#!/usr/bin/env python3

def solution(n):
    if n <= 1:
        return 1
    return n * solution(n - 1)
