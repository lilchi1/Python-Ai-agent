import sys, os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
from src.neural.neural_trainer import NeuralTrainer, DEFAULT_MODEL

lib = os.path.join('data', 'lib', 'code_lib.txt')
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
seq = int(sys.argv[2]) if len(sys.argv) > 2 else 48

tr = NeuralTrainer(model_path=DEFAULT_MODEL)
tr._load_model()
print('Starting library training: %d steps, seq %d' % (steps, seq))
tr.train_corpus([lib], seq_len=seq, n_steps=steps, model_path=DEFAULT_MODEL)
tr.close()
print('Library training DONE, saved')
