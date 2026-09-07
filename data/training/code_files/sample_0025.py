def mode(arr):
    if not arr:
        return None
    freq = {}
    for x in arr:
        freq[x] = freq.get(x, 0) + 1
    return max(freq, key=freq.get)
