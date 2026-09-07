#!/usr/bin/env python3
import os, collections
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
p = os.path.join(ROOT, 'data', 'lib', 'code_lib.txt')
with open(p, 'rb') as f:
    by = f.read()
norm = bytearray()
i = 0
while i < len(by):
    if by[i] == 13:
        if i + 1 < len(by) and by[i+1] == 10:
            norm.append(10); i += 2; continue
        i += 1; continue
    norm.append(by[i]); i += 1
ascii_set = sorted(set(b for b in norm if b < 128))
print('distinct ASCII in code_lib ONLY:', len(ascii_set))
print(sorted(chr(b) if 32 <= b <= 126 else '\\n' for b in ascii_set))
if len(ascii_set) <= 64:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'top_bytes.txt'), 'w') as f:
        for b in ascii_set:
            f.write('%d\n' % b)
    print('FITS in 64! wrote top_bytes.txt with ALL library bytes.')
else:
    print('TOO MANY')
