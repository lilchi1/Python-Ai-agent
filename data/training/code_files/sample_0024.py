def median(arr):
    if not arr:
        return 0
    sorted_arr = sorted(arr)
    n = len(sorted_arr)
    if n % 2 == 1:
        return sorted_arr[n // 2]
    return (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2
