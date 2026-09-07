def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i-1][w]
            if weights[i-1] <= w:
                val = dp[i-1][w - weights[i-1]] + values[i-1]
                if val > dp[i][w]:
                    dp[i][w] = val
    return dp[n][capacity]
