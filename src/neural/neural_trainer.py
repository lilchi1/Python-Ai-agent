#!/usr/bin/env python3
"""
Нейронный тренер Mini-GPT — обёртка над C-моделью с несколькими способами обучения.

Режимы обучения:
  1. corpus_learn    — обучение на корпусе кода (из data/training и заданных файлов).
  2. self_learn      — обучение на собственном коде проекта (агент учится на себе).
  3. knowledge_report— обучение на извлечённых знаниях (из векторного хранилища / web).
  4. feedback_learn  — обучение на ранее сгенерированных и подтверждённо валидных решениях
                      (усиление успешных решений агента).
  5. eval            — оценка перплексии/потерь.
"""
import ctypes
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model.minigpt_binding import MiniGPT, compile_lib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
MODEL_DIR = os.path.join(ROOT, 'src', 'model')
DATA_TRAIN = os.path.join(ROOT, 'data', 'training')
DATA_KNOWLEDGE = os.path.join(ROOT, 'data', 'knowledge')
DEFAULT_MODEL = os.path.join(MODEL_DIR, 'model.bin')

CODE_EXTS = ('.py', '.c', '.js', '.ts', '.go', '.rs', '.h', '.cpp', '.java')


def discover_code_files(root, extensions=CODE_EXTS):
    files = []
    if not os.path.isdir(root):
        return files
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in
                       ('.git', '__pycache__', 'node_modules', '.venv', 'venv', 'data')]
        for fn in filenames:
            if fn.endswith(extensions):
                files.append(os.path.join(dirpath, fn))
    return files


def read_bytes(path):
    try:
        with open(path, 'rb') as f:
            return f.read()
    except Exception:
        return b''


class NeuralTrainer:
    """Обучает языковую модель Mini-GPT на байтовом потоке данных."""

    def __init__(self, model_path=None, lib_path=None, quiet=True):
        self.model_path = model_path or DEFAULT_MODEL
        if lib_path is None:
            lib_path = os.path.join(MODEL_DIR, 'minigpt_lib.dll')
        self.quiet = quiet
        self.lib_path = lib_path
        if not os.path.exists(lib_path):
            compile_lib(lib_path, quiet=quiet)
        self._m = None
        self._lib = None

    # ── Низкоуровневый доступ ─────────────────────────────────
    def _load_model(self):
        if self._m is not None:
            return True
        if not os.path.exists(self.lib_path):
            return False
        if self._lib is None:
            import ctypes as C
            self._lib = C.CDLL(self.lib_path)
            self._lib.host_init.restype = C.c_void_p
            self._lib.host_init.argtypes = []
            self._lib.host_free.argtypes = [C.c_void_p]
            self._lib.host_nparams.argtypes = [C.c_void_p]
            self._lib.host_nparams.restype = C.c_int
            self._lib.host_load.argtypes = [C.c_void_p, C.c_char_p]
            self._lib.host_load.restype = C.c_int
            self._lib.host_save.argtypes = [C.c_void_p, C.c_char_p]
            self._lib.host_save.restype = C.c_int
            self._lib.host_train_step.argtypes = [C.c_void_p, C.POINTER(C.c_int), C.c_int]
            self._lib.host_train_step.restype = C.c_float
            self._lib.host_forward.argtypes = [C.c_void_p, C.POINTER(C.c_int), C.c_int, C.c_int]
            self._lib.host_forward.restype = C.c_float
            # host_train_data: void* h, int* data, int len, int seq, int epochs, int print_every
            self._lib.host_train_data.argtypes = [C.c_void_p, C.POINTER(C.c_int),
                                                  C.c_int, C.c_int, C.c_int, C.c_int]
            self._lib.host_generate.argtypes = [C.c_void_p, C.POINTER(C.c_int),
                                                C.c_int, C.c_int, C.c_float, C.POINTER(C.c_int)]
            self._lib.host_generate.restype = C.c_int
        self._m = self._lib.host_init()
        if os.path.exists(self.model_path):
            self._lib.host_load(self._m, os.fspath(self.model_path).encode())
        return True

    def close(self):
        if self._m is not None and self._lib is not None:
            self._lib.host_free(self._m)
            self._m = None

    def nparams(self):
        if not self._load_model():
            return 0
        return self._lib.host_nparams(self._m)

    def save(self, path=None):
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not self._load_model():
            return
        self._lib.host_save(self._m, os.fspath(path).encode())

    # ── Обучение шаг за шагом (полный контроль) ───────────────
    def train_steps(self, data_bytes, seq_len=64, n_steps=1000, lr=None,
                    print_every=100, start_offset=None, shuffle=True,
                    model_path=None):
        """Обучает на байтовом потоке фиксированное число шагов."""
        import ctypes as C
        if len(data_bytes) < seq_len + 1:
            raise ValueError("Данных слишком мало для обучения")
        if not self._load_model():
            raise RuntimeError("Модель не загружена")
        seq_len = min(seq_len, 256, len(data_bytes) - 1)
        from src.model.code_tokenizer import get_tokenizer
        data = get_tokenizer().encode(data_bytes)
        arr = (C.c_int * len(data))(*data)
        max_start = len(data) - seq_len - 1

        max_off = max_start + 1
        offsets = list(range(0, max_off, seq_len))
        if shuffle:
            random.shuffle(offsets)

        t0 = time.time()
        total_loss = 0.0
        steps_done = 0
        for i in range(n_steps):
            off = offsets[i % len(offsets)]
            ptr = C.cast(C.byref(arr, off * C.sizeof(C.c_int)), C.POINTER(C.c_int))
            loss = self._lib.host_train_step(self._m, ptr, seq_len)
            total_loss += loss
            steps_done += 1
            if print_every > 0 and (i + 1) % print_every == 0:
                avg = total_loss / (i + 1)
                print(f"  step {i+1}/{n_steps} | loss {loss:.4f} | avg {avg:.4f} | "
                      f"{(time.time()-t0)/(i+1)*1000:.0f}ms/step")
                if model_path and (i + 1) % (print_every * 20) == 0:
                    self.save(model_path + '.chkpt')
        print(f"  Обучение: {n_steps} шагов за {time.time()-t0:.1f}с, "
              f"средняя потеря {total_loss/n_steps:.4f}")
        if model_path:
            self.save(model_path)
        return total_loss / n_steps

    def train_corpus(self, corpus_paths=(), seq_len=64, n_steps=1500, epochs_hint=0,
                     print_every=100, model_path=None, extra_bytes=b''):
        """Обучает на нескольких файлах корпуса."""
        chunks = []
        for p in corpus_paths:
            if os.path.isdir(p):
                for fp in discover_code_files(p):
                    chunks.append(read_bytes(fp))
            elif os.path.isfile(p):
                chunks.append(read_bytes(p))
        if extra_bytes:
            chunks.append(extra_bytes)
        data = b'\n\n'.join([c for c in chunks if c]) or b'\n'
        return self.train_steps(data, seq_len=seq_len, n_steps=n_steps,
                                print_every=print_every, model_path=model_path)

    # ── Режимы обучения (разные способы) ──────────────────────
    def learn_corpus(self, n_steps=2000, seq_len=64, corpus=None, model_path=None):
        """Режим 1: обучение на корпусе кода."""
        print("=" * 60)
        print("  РЕЖИМ 1: ОБУЧЕНИЕ НА КОРПУСЕ КОДА")
        print("=" * 60)
        if corpus is None:
            lib = os.path.join(os.path.dirname(DATA_TRAIN), 'lib', 'code_lib.txt')
            tune = os.path.join(os.path.dirname(DATA_TRAIN), 'tune', 'tune_corpus.txt')
            if os.path.exists(lib):
                corpus = lib
            elif os.path.exists(tune):
                corpus = tune
            else:
                corpus = os.path.join(DATA_TRAIN, 'code_corpus.txt')
        paths = [corpus] if isinstance(corpus, str) else list(corpus)
        return self.train_corpus(paths, seq_len=seq_len, n_steps=n_steps,
                                 model_path=model_path)

    def learn_pairs(self, n_steps=3000, seq_len=64, model_path=None,
                    pairs_file='data/lib/code_pairs.txt'):
        """Режим обучения на ПАРАХ (signature -> body), собранных
        build_code_lib.build_pairs(). Это supervised-достраивание кода:
        модель учится продолжать строку после сигнатуры функции, поэтому
        при генерации от 'def factorial(n):' воспроизводит её тело."""
        print("=" * 60)
        print("  РЕЖИМ ПАР: ОБУЧЕНИЕ НА ПАРАХ (SIGNATURE -> BODY)")
        print("=" * 60)
        pairs_file = os.path.join(ROOT, pairs_file)
        if not os.path.exists(pairs_file):
            print(f"  Файла пар нет ({pairs_file}) — собираем через build_code_lib")
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                'bcl', os.path.join(MODEL_DIR, 'build_code_lib.py'))
            bcl = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bcl)
            bcl.build_pairs()
        data = read_bytes(pairs_file)
        if not data:
            print("  Пар нет — пропускаем")
            return None
        # Для парной задачи предпочитаем короткие окна, выровненные так, чтобы
        # модель чаще видела начало сигнатуры и как можно больше тела.
        return self.train_steps(data, seq_len=seq_len, n_steps=n_steps,
                                model_path=model_path)

    def learn_self(self, n_steps=1500, seq_len=64, model_path=None):
        """Режим 2: самообучение на собственном коде проекта."""
        print("=" * 60)
        print("  РЕЖИМ 2: САМООБУЧЕНИЕ НА КОДЕ ПРОЕКТА")
        print("=" * 60)
        files = discover_code_files(ROOT)
        if not files:
            print("  Нет файлов кода для самообучения")
            return None
        chunks = [read_bytes(f) for f in files]
        data = b'\n\n'.join([c for c in chunks if c])
        print(f"  Кода проекта: {len(files)} файлов, {len(data)} байт")
        return self.train_steps(data, seq_len=seq_len, n_steps=n_steps,
                                model_path=model_path)

    def learn_knowledge(self, n_steps=1500, seq_len=64, model_path=None):
        """Режим 3: обучение на извлечённых знаниях (векторное хранилище)."""
        print("=" * 60)
        print("  РЕЖИМ 3: ОБУЧЕНИЕ НА ЗНАНИЯХ (STORAGE)")
        print("=" * 60)
        text_parts = []
        know_path = os.path.join(DATA_KNOWLEDGE, 'code_knowledge.json')
        if os.path.exists(know_path):
            try:
                import json
                with open(know_path, 'r', encoding='utf-8') as f:
                    kdata = json.load(f)
                docs = kdata.get('documents', [])
                for d in docs:
                    t = d.get('text', '')
                    if len(t) > 30:
                        text_parts.append(t)
            except Exception as e:
                print(f"  [!] {e}")
        if not text_parts:
            print("  Знаний нет — пропускаем (сначала запустите /learn)")
            return None
        data = b'\n\n'.join(t.encode('utf-8', errors='ignore') for t in text_parts)
        print(f"  Фрагментов знаний: {len(text_parts)}, {len(data)} байт")
        return self.train_steps(data, seq_len=seq_len, n_steps=n_steps,
                                model_path=model_path)

    def learn_feedback(self, solutions_dir='data/solutions', n_steps=1200, seq_len=64,
                       model_path=None):
        """Режим 4: обучение на подтверждённо валидных решениях агента."""
        print("=" * 60)
        print("  РЕЖИМ 4: ОБУЧЕНИЕ НА УСПЕШНЫХ РЕШЕНИЯХ")
        print("=" * 60)
        sol_dir = os.path.join(ROOT, solutions_dir)
        files = []
        if os.path.isdir(sol_dir):
            for dp, dn, fn in os.walk(sol_dir):
                for f in fn:
                    if f.endswith('.py'):
                        files.append(os.path.join(dp, f))
        if not files:
            print(f"  Решений пока нет в {sol_dir} — пропускаем")
            return None
        chunks = [read_bytes(f) for f in files]
        data = b'\n\n'.join([c for c in chunks if c])
        print(f"  Решений: {len(files)}, {len(data)} байт")
        return self.train_steps(data, seq_len=seq_len, n_steps=n_steps,
                                model_path=model_path)

    # ── Оценка ────────────────────────────────────────────────
    def evaluate(self, path, seq_len=64):
        """Оценка перплексии на файле."""
        import ctypes as C
        if not self._load_model():
            return None
        data = read_bytes(path)
        if len(data) < seq_len + 1:
            print(f"  [{os.path.basename(path)}] мало данных для оценки")
            return None
        seq_len = min(seq_len, len(data) - 1)
        from src.model.code_tokenizer import get_tokenizer
        enc = get_tokenizer().encode(data)
        total, cnt = 0.0, 0
        for start in range(0, len(enc) - seq_len, seq_len // 2):
            seg = (C.c_int * seq_len)(*enc[start:start + seq_len])
            loss = self._lib.host_forward(self._m, seg, seq_len, seq_len - 1)
            total += loss
            cnt += 1
        avg = total / cnt if cnt else None
        if avg is not None:
            print(f"  [{os.path.basename(path)}] loss {avg:.4f} | "
                  f"perplexity {2**avg:.1f} ({cnt} окон)")
        return avg


def multi_mode_trainer(model_path=None):
    """Запускает обучение по всем доступным режимам."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=1500, help='Шагов на каждый режим')
    ap.add_argument('--seq', type=int, default=64)
    ap.add_argument('--mode', default='all',
                    choices=['all', 'corpus', 'pairs', 'self', 'knowledge', 'feedback', 'eval'])
    ap.add_argument('--model', default=None)
    ap.add_argument('--eval-file', default=None)
    ap.add_argument('--save', default=None)
    args = ap.parse_args()

    mp = args.model or DEFAULT_MODEL
    tr = NeuralTrainer(model_path=mp)
    if not tr._load_model():
        print("  [!] Не удалось загрузить модель")
        sys.exit(1)
    print(f"  Параметры модели: {tr.nparams()}")
    print(f"  Модель: {mp}")

    if args.mode == 'eval' or (args.mode == 'all' and args.eval_file):
        if args.eval_file:
            tr.evaluate(args.eval_file, seq_len=args.seq)
    elif args.mode == 'corpus':
        tr.learn_corpus(n_steps=args.steps, seq_len=args.seq)
    elif args.mode == 'pairs':
        tr.learn_pairs(n_steps=args.steps, seq_len=args.seq)
    elif args.mode == 'self':
        tr.learn_self(n_steps=args.steps, seq_len=args.seq)
    elif args.mode == 'knowledge':
        tr.learn_knowledge(n_steps=args.steps, seq_len=args.seq)
    elif args.mode == 'feedback':
        tr.learn_feedback(n_steps=args.steps, seq_len=args.seq)
    else:
        tr.learn_pairs(n_steps=args.steps, seq_len=args.seq, model_path=mp)
        tr.learn_corpus(n_steps=max(1200, args.steps // 2), seq_len=args.seq,
                        model_path=mp)
        tr.learn_self(n_steps=max(800, args.steps // 2), seq_len=args.seq,
                      model_path=mp)
        tr.learn_knowledge(n_steps=max(500, args.steps // 3), seq_len=args.seq,
                           model_path=mp)

    tr.save(args.save or mp)
    tr.close()
    print("\n  Обучение завершено. Модель сохранена:",
          args.save or mp)


if __name__ == "__main__":
    multi_mode_trainer()
