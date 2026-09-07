import ctypes, os, sys, time

os.chdir(os.path.join('src', 'model'))
lib = ctypes.CDLL('./minigpt_lib.dll')
lib.host_init.restype = ctypes.c_void_p
lib.host_init.argtypes = []
h = lib.host_init()
lib.host_nparams.argtypes = [ctypes.c_void_p]
lib.host_nparams.restype = ctypes.c_int
print('params:', lib.host_nparams(h))

lib.host_train_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.host_train_step.restype = ctypes.c_float

corpus = open(os.path.join('..', '..', 'data', 'training', 'code_corpus.txt'), 'rb').read()
data = list(corpus)
arr = (ctypes.c_int * len(data))(*data)

for seq in (48, 64, 96):
    n = 300
    t0 = time.time()
    for i in range(n):
        off = (i * seq) % (len(data) - seq - 1)
        ptr = ctypes.cast(ctypes.byref(arr, off * ctypes.sizeof(ctypes.c_int)),
                          ctypes.POINTER(ctypes.c_int))
        lib.host_train_step(h, ptr, seq)
    dt = time.time() - t0
    print(f'seq={seq}: {n} steps in {dt:.2f}s -> {dt/n*1000:.1f} ms/step')

lib.host_free.argtypes = [ctypes.c_void_p]
lib.host_free(h)
