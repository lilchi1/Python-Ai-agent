#!/usr/bin/env python3
"""
Компактная кодовая токенизация для Mini-GPT.

Полный байтовый алфавит (256) даёт модели слишком высокую энтропию следующего
токена, чтобы ~полумиллионная модель могла запомнить структуру кода. Решение —
ОГРАНИЧЕННЫЙ кодо-ориентированный алфавит: только ASCII-синтаксис (пробел,
\n, буквы, цифры, операторы), а остальные байты свертываются в один UNK-токен.

Это резко снижает энтропию и позволяет небольшой модели реально учить код.
"""
import os

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ALPHA_FILE = os.path.join(_MODEL_DIR, 'code_alphabet.txt')


def _alphabet_list():
    """Список байт алфавита: \n + все ASCII-печатные (0x20..0x7e)."""
    return sorted(set([10] + list(range(32, 127))))


def _ensure_alphabet_file():
    if not os.path.exists(ALPHA_FILE):
        with open(ALPHA_FILE, 'w') as f:
            for b in _alphabet_list():
                f.write('%d\n' % b)


class CodeTokenizer:
    UNK_VALUE = 0  # байт-заглушка при декодировании UNK-токена

    def __init__(self):
        _ensure_alphabet_file()
        alpha = []
        with open(ALPHA_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    alpha.append(int(line))
        alpha = sorted(set(alpha))
        self.idx_to_byte = dict(enumerate(alpha))
        self.byte_to_idx = {b: i for i, b in enumerate(alpha)}
        # UNK-токен размещаем сразу за алфавитом
        self.unk_idx = len(alpha)
        self.idx_to_byte[self.unk_idx] = self.UNK_VALUE
        self.vocab_size = len(alpha) + 1  # = 97

    def encode(self, data: bytes) -> list:
        idx = self.byte_to_idx
        unk = self.unk_idx
        return [idx.get(b, unk) for b in data]

    def decode(self, tokens) -> bytes:
        inv = self.idx_to_byte
        out = bytearray()
        for t in tokens:
            out.append(inv.get(t, self.UNK_VALUE))
        return bytes(out)


_TOKENIZER_INSTANCE = None


def get_tokenizer():
    global _TOKENIZER_INSTANCE
    if _TOKENIZER_INSTANCE is None:
        _TOKENIZER_INSTANCE = CodeTokenizer()
    return _TOKENIZER_INSTANCE


def encode_bytes(data: bytes) -> list:
    return get_tokenizer().encode(data)


def decode_tokens(tokens) -> bytes:
    return get_tokenizer().decode(tokens)


if __name__ == '__main__':
    t = get_tokenizer()
    print('vocab_size:', t.vocab_size)
    s = 'def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n'
    b = s.encode('utf-8')
    dec = t.decode(t.encode(b))
    print('roundtrip ok:', dec.decode('utf-8') == s)
