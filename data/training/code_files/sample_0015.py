def lcm(a, b):
    return abs(a * b) // gcd(a, b) if a and b else 0
