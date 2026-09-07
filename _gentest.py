import sys, os, ctypes
sys.path.insert(0, '.')

from src.model.minigpt_binding import MiniGPT

m = MiniGPT(model_path='src/model/model.bin', quiet=True)
print('params:', m.nparams())

prompts = ["def factorial(", "def fibonacci(", "class Stack:", "import os\n\ndef main():"]
for p in prompts:
    print('=' * 50)
    print('PROMPT:', repr(p))
    try:
        out = m.generate(p, max_tokens=200, temperature=0.7)
        print('OUTPUT:')
        print(out)
    except Exception as e:
        print('ERR:', e)
    print()
m.close()
print('DONE')
