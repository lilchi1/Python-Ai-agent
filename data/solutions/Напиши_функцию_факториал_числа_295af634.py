# Task: Напиши функцию факториал числа
# Language: python
# Based on similar solution:
# ---
#!/usr/bin/env python3
# ---

#!/usr/bin/env python3

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
