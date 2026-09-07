#!/usr/bin/env python3
"""
Python-обёртка для мини-GPT языковой модели на C (ctypes).
Позволяет использовать модель из Python без внешних зависимостей.
"""
import ctypes
import os
import subprocess

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_MODEL_DIR))


def _find_compiler():
    for name in ('gcc', 'clang', 'cc'):
        try:
            r = subprocess.run([name, '--version'], capture_output=True)
            if r.returncode == 0:
                return name
        except Exception:
            continue
    return None


def compile_lib(lib_path=None, quiet=False):
    """Компилирует C-библиотеку модели. Возвращает путь или None."""
    if lib_path is None:
        lib_path = os.path.join(_MODEL_DIR, 'minigpt_lib.dll')

    if os.path.exists(lib_path):
        return lib_path

    compiler = _find_compiler()
    if not compiler:
        return None

    src = os.path.join(_MODEL_DIR, 'mini_gpt.c')
    host = os.path.join(_MODEL_DIR, 'minigpt_host.c')
    cmd = [compiler, '-O2', '-shared', '-o', lib_path, src, host, '-lm']
    if not quiet:
        print(f"  Компиляция языковой модели: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        if not quiet:
            print("  Ошибка компиляции:")
            print(r.stderr)
        return None
    if not quiet:
        print(f"  Модель скомпилирована: {lib_path}")
    return lib_path if os.path.exists(lib_path) else None


class MiniGPT:
    """
    Языковая модель на чистом C, вызываемая через ctypes.
    Побайтовая токенизация (VOCAB_SIZE=256) — работает с любым текстом.

    Модель живёт в куче C (host_init), передаётся как void*.
    """

    def __init__(self, model_path=None, lib_path=None, quiet=False):
        self.lib = None
        self._model = None

        if lib_path is None:
            lib_path = os.path.join(_MODEL_DIR, 'minigpt_lib.dll')

        compile_lib(lib_path, quiet=quiet)

        if not os.path.exists(lib_path):
            raise RuntimeError(
                f"Библиотека {lib_path} не найдена. Установите gcc "
                f"(MSYS2/MinGW) и запустите снова.")

        self.lib = ctypes.CDLL(lib_path)
        self._setup_signatures()
        self._model = self.lib.host_init()

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    # ── Сигнатуры ───────────────────────────────────────────────
    def _setup_signatures(self):
        L = self.lib

        L.host_init.argtypes = []
        L.host_init.restype = ctypes.c_void_p

        L.host_free.argtypes = [ctypes.c_void_p]

        L.host_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.host_load.restype = ctypes.c_int
        L.host_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.host_save.restype = ctypes.c_int
        L.host_nparams.argtypes = [ctypes.c_void_p]
        L.host_nparams.restype = ctypes.c_int

        L.host_generate.argtypes = [ctypes.c_void_p,
                                    ctypes.POINTER(ctypes.c_int),
                                    ctypes.c_int, ctypes.c_int,
                                    ctypes.c_float,
                                    ctypes.POINTER(ctypes.c_int)]
        L.host_generate.restype = ctypes.c_int

        L.host_forward.argtypes = [ctypes.c_void_p,
                                   ctypes.POINTER(ctypes.c_int),
                                   ctypes.c_int, ctypes.c_int]
        L.host_forward.restype = ctypes.c_float

        L.host_train_step.argtypes = [ctypes.c_void_p,
                                      ctypes.POINTER(ctypes.c_int),
                                      ctypes.c_int]
        L.host_train_step.restype = ctypes.c_float

        L.host_train_data.argtypes = [ctypes.c_void_p,
                                      ctypes.POINTER(ctypes.c_int),
                                      ctypes.c_int, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_int]

    # ── API ──────────────────────────────────────────────────────
    def load(self, path):
        return self.lib.host_load(self._model, os.fspath(path).encode())

    def save(self, path):
        return self.lib.host_save(self._model, os.fspath(path).encode())

    def nparams(self):
        return self.lib.host_nparams(self._model)

    def generate(self, prompt, max_tokens=500, temperature=0.8):
        # Кодируем через компактный кодовой токенизатор (индексы алфавита).
        from src.model.code_tokenizer import get_tokenizer
        tok = get_tokenizer()
        text = prompt.encode('utf-8', errors='replace')
        tokens = tok.encode(text)
        n_tok = len(tokens)

        arr = (ctypes.c_int * max(1, n_tok))()
        for i, t in enumerate(tokens):
            arr[i] = t

        out = (ctypes.c_int * max_tokens)()

        n = self.lib.host_generate(self._model, arr, n_tok, max_tokens,
                                   ctypes.c_float(temperature), out)

        result = text + tok.decode(out[:n])
        return result.decode('utf-8', errors='replace')

    def close(self):
        if self.lib and self._model:
            self.lib.host_free(self._model)
            self._model = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def get_model(model_path=None, quiet=True):
    """Возвращает экземпляр MiniGPT."""
    if model_path and not os.path.exists(model_path):
        model_path = None
    return MiniGPT(model_path=model_path, quiet=quiet)


if __name__ == "__main__":
    m = MiniGPT()
    print(f"Параметры: {m.nparams()}")
    print(f"Генерация:\n{m.generate('def factorial(', max_tokens=200)}")
    m.close()
