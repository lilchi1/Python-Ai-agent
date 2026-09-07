import sys, os, traceback
sys.path.insert(0, '.')
log = open(os.path.join(os.environ['TEMP'], 'sigt_debug.txt'), 'w', encoding='utf-8')
p = lambda *a: log.write(' '.join(str(x) for x in a) + '\n') or log.flush()

from src.neural.neural_generator import NeuralGenerator
from src.generator.code_generator import CodeGenerator, _extract_name, _extract_params
_cg = CodeGenerator()

ng = NeuralGenerator()
tasks = ['Напиши функцию факториал числа', 'проверить простое ли число',
         'функция fibonacci', 'сумма списка', 'класс стек',
         'функция реверса строки']
for t in tasks:
    p('=' * 30)
    p('TASK:', repr(t))
    try:
        p('  _extract_name ->', _extract_name(t))
    except Exception as e:
        p('  _extract_name ERR', repr(e))
    try:
        p('  _guess_name ->', ng._guess_name(t))
    except Exception as e:
        p('  _guess_name ERR', repr(e))
        traceback.print_exc(file=log)
    try:
        p('  _extract_params ->', _extract_params(t))
        p('  _infer_params ->', _cg._infer_params(t))
    except Exception as e:
        p('  params ERR', repr(e))
    p('  signature ->', repr(ng._signature_from_task(t)))

log.close()
print('DONE')
