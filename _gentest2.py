import sys, os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
from src.model.minigpt_binding import MiniGPT

m = MiniGPT(model_path='src/model/model.bin', quiet=True)
print('params:', m.nparams())

prompts = [
    "def factorial(n):",
    "def is_prime(n):",
    "class Stack:",
    "def main():",
    "import os\n\n",
]
for p in prompts:
    print('=' * 40)
    print('PROMPT:', repr(p))
    out = m.generate(p, max_tokens=150, temperature=0.6)
    print('OUTPUT:')
    print(out)
    print()
m.close()
