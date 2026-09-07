def power(base, exp):
    if exp == 0:
        return 1
    if exp < 0:
        return 1.0 / power(base, -exp)
    result = 1
    for _ in range(exp):
        result *= base
    return result
