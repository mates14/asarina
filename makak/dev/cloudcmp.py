#!/usr/bin/env python3
"""cloudcmp.py RUN... : per-frame outcome of cloudbench.sh runs side by side (zp/IDNUM or failure)."""
import sys, glob
from pathlib import Path
from astropy.time import Time

runs = sys.argv[1:]
res = {}
for r in runs:
    d = Path('/home/mates/asarina+makak/bench') / r / 'nght'
    for f in glob.glob(str(d / 'mr????????.dat')):
        for l in open(f):
            w = l.split()
            res.setdefault(w[0][:18], {})[r] = f"{float(w[7]):6.3f} {int(w[6]):4d}"
    for f in glob.glob(str(d / 'mr*f.dat')):
        for l in open(f):
            w = l.split()
            key = Time(float(w[0]), format='unix').strftime('%Y%m%d%H%M%S')
            res.setdefault(key, {})[r] = f"{w[3]:>11s}"
    for f in glob.glob(str(d / 'mr*d.dat')):
        for l in open(f):
            key = Time(float(l.split()[0]), format='unix').strftime('%Y%m%d%H%M%S')
            res.setdefault(key, {})[r] = '       dark'
print('frame          ' + ''.join(f"{r:>13s}" for r in runs))
for k in sorted(res):
    print(k[:14] + ' ' + ''.join(f"{res[k].get(r, '-'):>13s}" for r in runs))
for r in runs:
    ok = sum(1 for k in res if r in res[k] and res[k][r][:6].strip().replace('.', '').isdigit())
    t = [float(l.split()[2][:-1]) for l in open(f'/home/mates/asarina+makak/bench/{r}/times')]
    print(f"{r}: {ok} with zeropoint, {sum(t)/len(t):.1f} s/frame, max {max(t):.0f} s")
