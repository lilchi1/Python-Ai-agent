import sys, os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
log = open(os.path.join(os.environ['TEMP'], 'agent_test.txt'), 'w', encoding='utf-8')
def p(x):
    log.write(str(x) + '\n'); log.flush()

p('start')
sys.path.insert(0, '.')
from src.memory.vector_store import LocalVectorStore
from src.agent.coding_agent import CodingAgent
from src.orchestrator import Orchestrator

vs = LocalVectorStore()
p('vector store ok')

from run import init_knowledge
try:
    init_knowledge(vs)
    p('knowledge seeded')
except Exception as e:
    p('seed warn: ' + repr(e))

agent = CodingAgent(vs, max_attempts=2)
p('neural_ready: ' + str(agent.neural_ready))
orch = Orchestrator(agent)

tests = [
    ("Напиши функцию факториал числа", None),
    ("Write a function that calculates factorial", 'python'),
    ("Напиши функцию проверки простого числа", None),
]
for task, lang in tests:
    p('=' * 40)
    p('TASK: ' + task)
    try:
        r = orch.execute(task, lang)
        p('success: ' + str(r.get('success')))
        p('lang: ' + str(r.get('language')))
        code = r.get('code', '')
        p('code length: ' + str(len(code)))
        p('code head: ' + repr(code[:150]))
    except Exception as e:
        import traceback
        p('ERROR: ' + repr(e))
        traceback.print_exc(file=log)

p('=' * 40)
p('TEST DONE')
log.close()
print('FINISHED, log in temp')
