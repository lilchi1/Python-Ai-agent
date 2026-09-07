#!/usr/bin/env python3
"""
Обучение и самообучение языковой модели Mini-GPT.

Способы обучения (разные режимы):
  python train.py --mode corpus    — обучение на корпусе кода
  python train.py --mode self      — самообучение на собственном коде проекта
  python train.py --mode knowledge — обучение на извлечённых знаниях (web/хранилище)
  python train.py --mode feedback  — обучение на успешных решениях агента
  python train.py --mode all       — все доступные режимы подряд
  python train.py --mode eval      — оценка перплексии

Параметры:
  --steps N  — число шагов обучения на режим
  --seq N    — длина последовательности (итого ~N токенов контекста)
  --model P  — путь к файлу модели (по умолчанию src/model/model.bin)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.neural.neural_trainer import NeuralTrainer, DEFAULT_MODEL


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Обучение Mini-GPT')
    ap.add_argument('--steps', type=int, default=1500)
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
        print("  [!] Не удалось загрузить библиотеку модели")
        sys.exit(1)
    print(f"  Параметры: {tr.nparams()} | Модель: {mp}")
    print(f"  Шагов/режим: {args.steps} | seq: {args.seq}")

    if args.eval_file:
        tr.evaluate(args.eval_file, seq_len=args.seq)
    elif args.mode == 'corpus':
        tr.learn_corpus(n_steps=args.steps, seq_len=args.seq, model_path=args.save or mp)
    elif args.mode == 'pairs':
        tr.learn_pairs(n_steps=args.steps, seq_len=args.seq, model_path=args.save or mp)
    elif args.mode == 'self':
        tr.learn_self(n_steps=args.steps, seq_len=args.seq, model_path=args.save or mp)
    elif args.mode == 'knowledge':
        tr.learn_knowledge(n_steps=args.steps, seq_len=args.seq, model_path=args.save or mp)
    elif args.mode == 'feedback':
        tr.learn_feedback(n_steps=args.steps, seq_len=args.seq, model_path=args.save or mp)
    else:
        tr.learn_pairs(n_steps=args.steps, seq_len=args.seq, model_path=args.save or mp)
        tr.learn_corpus(n_steps=max(500, args.steps // 2), seq_len=args.seq,
                        model_path=args.save or mp)
        tr.learn_self(n_steps=max(400, args.steps // 2), seq_len=args.seq,
                      model_path=args.save or mp)
        tr.learn_knowledge(n_steps=max(300, args.steps // 3), seq_len=args.seq,
                           model_path=args.save or mp)

    tr.save(args.save or mp)
    tr.close()
    print("\n  Готово. Модель сохранена:", args.save or mp)


if __name__ == "__main__":
    main()
