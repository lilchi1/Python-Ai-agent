# Task: Напиши функцию проверки простого числа
# Language: python
# Based on similar solution:
# ---
#!/usr/bin/env python3
# ---

#!/usr/bin/env python3

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
