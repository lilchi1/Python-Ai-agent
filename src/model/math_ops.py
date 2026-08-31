"""
Матричные операции на чистом Python.
Все вычисления — списки списков, без numpy/torch.
"""
import math
import random


def matmul(A, B):
    """Умножение матриц: (m x k) * (k x n) -> (m x n)"""
    m = len(A)
    k = len(A[0])
    n = len(B[0])
    result = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k):
                s += A[i][p] * B[p][j]
            result[i][j] = s
    return result


def matmul_transA(A, B):
    """A^T * B: (k x m) * (k x n) -> (m x n) через A^T"""
    k = len(A)
    m = len(A[0])
    n = len(B[0])
    result = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k):
                s += A[p][i] * B[p][j]
            result[i][j] = s
    return result


def matmul_transB(A, B):
    """A * B^T: (m x k) * (n x k) -> (m x n)"""
    m = len(A)
    k = len(A[0])
    n = len(B)
    result = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k):
                s += A[i][p] * B[j][p]
            result[i][j] = s
    return result


def transpose(M):
    """Транспонирование матрицы"""
    rows = len(M)
    cols = len(M[0])
    return [[M[i][j] for i in range(rows)] for j in range(cols)]


def add_matrices(A, B):
    """Поэлементное сложение"""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def scale_matrix(M, s):
    """Умножение матрицы на скаляр"""
    return [[M[i][j] * s for j in range(len(M[0]))] for i in range(len(M))]


def softmax(logits):
    """Softmax по последней оси для списка векторов.
    Вход: [[z1, z2, ...], ...] — батч векторов.
    """
    result = []
    for row in logits:
        max_val = max(row)
        exps = [math.exp(x - max_val) for x in row]
        s = sum(exps)
        result.append([e / s for e in exps])
    return result


def softmax_3d(logits):
    """Softmax для 3D тензора [batch][seq][vocab]"""
    result = []
    for batch in logits:
        result.append(softmax(batch))
    return result


def layer_norm_forward(x, gamma, beta, eps=1e-5):
    """Layer Normalization: x, gamma, beta — lists of dim size.
    x: [batch][seq][dim]
    """
    batch_size = len(x)
    seq_len = len(x[0])
    dim = len(x[0][0])
    result = []

    for b in range(batch_size):
        result_b = []
        for s in range(seq_len):
            vals = x[b][s]
            mean = sum(vals) / dim
            var = sum((v - mean) ** 2 for v in vals) / dim
            norm = [(v - mean) / math.sqrt(var + eps) for v in vals]
            out = [norm[i] * gamma[i] + beta[i] for i in range(dim)]
            result_b.append(out)
        result.append(result_b)

    return result


def cross_entropy_loss(logits, targets):
    """Cross-entropy loss для батча.
    logits: [batch][vocab] — после линейного слоя
    targets: [batch] — индексы целевых токенов
    """
    total_loss = 0.0
    batch_size = len(logits)
    vocab_size = len(logits[0])

    for b in range(batch_size):
        logit = logits[b]
        max_l = max(logit)
        exps = [math.exp(x - max_l) for x in logit]
        s = sum(exps)
        probs = [e / s for e in exps]
        target = targets[b]
        total_loss -= math.log(max(probs[target], 1e-10))

    return total_loss / batch_size


def cross_entropy_grad(logits, targets):
    """Градиент cross-entropy + softmax."""
    batch_size = len(logits)
    vocab_size = len(logits[0])
    grad = [[0.0] * vocab_size for _ in range(batch_size)]

    for b in range(batch_size):
        logit = logits[b]
        max_l = max(logit)
        exps = [math.exp(x - max_l) for x in logit]
        s = sum(exps)
        probs = [e / s for e in exps]
        target = targets[b]
        for j in range(vocab_size):
            grad[b][j] = probs[j] - (1.0 if j == target else 0.0)

    return grad


def relu(x):
    """ReLU для матрицы"""
    return [[max(0, v) for v in row] for row in x]


def relu_grad(x):
    """Градиент ReLU"""
    return [[1.0 if v > 0 else 0.0 for v in row] for row in x]


def add_bias(x, bias):
    """Добавить bias к каждому вектору в батче.
    x: [batch][seq][dim] или [batch][dim]
    bias: [dim]
    """
    if isinstance(x[0][0], list):
        # 3D: [batch][seq][dim]
        return [
            [
                [x[b][s][i] + bias[i] for i in range(len(bias))]
                for s in range(len(x[b]))
            ]
            for b in range(len(x))
        ]
    else:
        # 2D: [batch][dim]
        return [
            [x[b][i] + bias[i] for i in range(len(bias))]
            for b in range(len(x))
        ]


def zeros(rows, cols):
    """Нулевая матрица"""
    return [[0.0] * cols for _ in range(rows)]


def rand_normal(rows, cols, scale=0.02):
    """Матрица с нормальным распределением (Box-Muller)"""
    result = []
    for i in range(rows):
        row = []
        for j in range(cols):
            u1 = max(random.random(), 1e-10)
            u2 = random.random()
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            row.append(z * scale)
        result.append(row)
    return result


def embedding_lookup(table, indices):
    """Lookup embedding по индексам.
    table: [vocab_size][dim]
    indices: [batch][seq]
    Возвращает: [batch][seq][dim]
    """
    dim = len(table[0])
    batch_size = len(indices)
    seq_len = len(indices[0])
    result = []
    for b in range(batch_size):
        result_b = []
        for s in range(seq_len):
            idx = indices[b][s]
            result_b.append(list(table[idx]))
        result.append(result_b)
    return result


def gather_grad(embedding_grad, indices, vocab_size, dim):
    """Собрать градиент для embedding.
    embedding_grad: [batch][seq][dim]
    indices: [batch][seq]
    Возвращает: [vocab_size][dim]
    """
    grad_table = [[0.0] * dim for _ in range(vocab_size)]
    batch_size = len(indices)
    seq_len = len(indices[0])
    for b in range(batch_size):
        for s in range(seq_len):
            idx = indices[b][s]
            for d in range(dim):
                grad_table[idx][d] += embedding_grad[b][s][d]
    return grad_table


def clip_grad_norm(grads, max_norm=1.0):
    """Gradient clipping по норме."""
    total_sq = 0.0
    for g in grads:
        if isinstance(g[0], list):
            for row in g:
                for v in row:
                    total_sq += v * v
        else:
            for v in g:
                total_sq += v * v
    norm = math.sqrt(total_sq)
    if norm > max_norm:
        scale = max_norm / norm
        return [scale_matrix(g, scale) if isinstance(g[0], list) else [v * scale for v in g] for g in grads]
    return grads
