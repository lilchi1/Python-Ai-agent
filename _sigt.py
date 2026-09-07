import sys
sys.path.insert(0, '.')
from src.neural.neural_generator import NeuralGenerator
ng = NeuralGenerator()
for t in ['Напиши функцию факториал числа', 'проверить простое ли число',
          'функция fibonacci', 'сумма списка', 'класс стек',
          'функция реверса строки']:
    print(repr(t), '->', repr(ng._signature_from_task(t)))
