import sys, os
log = open(os.path.join(os.environ['TEMP'], 'gentest2.txt'), 'w', encoding='utf-8')
def p(x):
    log.write(str(x) + '\n'); log.flush()
p('start')
sys.path.insert(0, '.')
from src.neural.neural_generator import NeuralGenerator, DEFAULT_MODEL_PATH
p('DEFAULT_MODEL_PATH: ' + str(DEFAULT_MODEL_PATH))
p('exists: ' + str(os.path.exists(DEFAULT_MODEL_PATH)))
ng = NeuralGenerator()
p('is_available: ' + str(ng.is_available()))
for prompt in ['def factorial(', 'def hello():']:
    code, ok = ng.generate('факториал числа', 'python')
    p('--- ok=' + str(ok))
    p(repr(code[:300]))
ng.close()
p('done')
log.close()
print('FINISHED, log in temp')
