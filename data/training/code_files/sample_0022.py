def sqrt_approx(n, precision=0.00001):
    if n < 0:
        return None
    if n == 0:
        return 0
    guess = n / 2.0
    while abs(guess * guess - n) > precision:
        guess = (guess + n / guess) / 2.0
    return guess
