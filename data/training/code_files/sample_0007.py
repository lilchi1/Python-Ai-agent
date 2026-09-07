def find_min(arr):
    if not arr:
        return None
    result = arr[0]
    for x in arr[1:]:
        if x < result:
            result = x
    return result
