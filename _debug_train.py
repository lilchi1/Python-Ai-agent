import ctypes, os, sys, traceback

log_path = os.path.join(os.environ.get('TEMP','.'), 'train_debug2.txt')
log = open(log_path, 'w', encoding='utf-8')

def p(x):
    log.write(str(x) + '\n')
    log.flush()

p('=== start ===')
try:
    os.chdir(os.path.join('src', 'model'))
    p('cwd: ' + os.getcwd())
    lib = ctypes.CDLL('./minigpt_lib.dll')
    p('dll loaded')
    lib.host_init.restype = ctypes.c_void_p
    lib.host_init.argtypes = []
    h = lib.host_init()
    p('host_init ok: ' + str(h))
    lib.host_train_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    data = b'def foo():\n    return 42\n\ndef bar(x):\n    return x*2\n'
    a = (ctypes.c_int * len(data))(*data)
    p('train data len ' + str(len(data)))
    lib.host_train_data(h, a, len(data), 8, 50, 50)
    p('train done, now generate')
    lib.host_generate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                  ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.POINTER(ctypes.c_int)]
    lib.host_generate.restype = ctypes.c_int
    prompt = (ctypes.c_int * 5)(ord('d'), ord('e'), ord('f'), ord(' '), ord('f'))
    out = (ctypes.c_int * 100)()
    n = lib.host_generate(h, prompt, 5, 100, ctypes.c_float(0.8), out)
    p('gen n=' + str(n))
    p('gen str: ' + bytes([out[i] & 0xFF for i in range(n)]).decode('utf-8', 'replace'))
    lib.host_free.argtypes = [ctypes.c_void_p]
    lib.host_free(h)
    p('freed, ALL OK')
except Exception as e:
    p('EXCEPTION: ' + repr(e))
    traceback.print_exc(file=log)
finally:
    log.close()
    print('LOG at ' + log_path)
