import sys, os
log = open(os.path.join(os.environ['TEMP'], 'savetest.txt'), 'w', encoding='utf-8')
def p(x):
    log.write(str(x) + '\n'); log.flush()

p('start')
sys.path.insert(0, '.')
from src.neural.neural_trainer import NeuralTrainer
p('imported')
tr = NeuralTrainer(model_path=None)
p('created')
ok = tr._load_model()
p('loaded: ' + str(ok))
if ok:
    p('params ' + str(tr.nparams()))
    tr.save('src/model/_savetest.bin')
    p('saved')
tr.close()
p('done')
log.close()
print('FINISHED, log in temp')
