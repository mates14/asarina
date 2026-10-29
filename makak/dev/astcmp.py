#!/usr/bin/env python3
"""astcmp.py RUN... : astrometry and zeropoint statistics of cloudbench runs."""
import sys, glob, re
import numpy as np
from astropy.table import Table

for r in sys.argv[1:]:
    base = f'/home/mates/asarina+makak/bench/{r}'
    meta = [Table.read(f, format='ascii.ecsv').meta for f in sorted(glob.glob(f'{base}/ecsv/*/*.ecsv'))]
    logs = ''.join(open(f).read() for f in glob.glob(f'{base}/*.log'))
    zp = np.array([float(l.split()[7]) for f in glob.glob(f'{base}/nght/mr????????.dat') for l in open(f)])
    qual = [str(m.get('ASTQUAL')) for m in meta]
    sig = np.array([m.get('ASTSIGMA', np.nan) or np.nan for m in meta], float)
    chi = [float(x) for x in re.findall(r'CHI2C=([0-9.]+)', logs)]
    print(f"{r:9s} n={len(meta):3d} OK={qual.count('OK'):3d} PARTIAL={qual.count('PARTIAL'):2d} "
          f"none={qual.count('None'):2d} kept-guess={logs.count('guess as it is'):2d} "
          f"field-solve={logs.count('field-solving'):2d} "
          f"ASTSIGMA med={np.nanmedian(sig):.3f} "
          f"CHI2C med={np.median(chi) if chi else np.nan:.2f} max={max(chi) if chi else np.nan:.1f} "
          f"zp med={np.median(zp):.3f} mad={np.median(np.abs(zp-np.median(zp))):.3f} "
          f"idnum med={np.median([int(m.get('IDNUM', 0)) for m in meta]):.0f}")
