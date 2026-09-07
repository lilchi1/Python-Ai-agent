import sys, os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
from src.model.minigpt_binding import MiniGPT
from src.model.code_tokenizer import get_tokenizer

t = get_tokenizer()
print('vocab_size:', t.vocab_size)

m = MiniGPT()  # fresh random model, 128 vocab
print('params:', m.nparams())
print('--- generation without save ---')
print(m.generate('def factorial(n):\n', max_tokens=40))
m.close()
