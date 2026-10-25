#!/usr/bin/env python3
"""
Drop whole frames whose smooth surface fit disagrees with the rest of the group.

The IRAF combine scripts split every frame into a smooth fit (imsurfit) and a
residual, average the fits without rejection and combine the residuals with
rejection. A per-pixel robust combine of the fits would build the smooth part
and the high-frequency part from different frame subsets (steps, artifacts),
so the fits stay a plain average - and a single frame with a large-scale
defect (saturated block of rows, light leak, residual image) bends the whole
master. This drops such frames from files.lst before both combines, so each
frame is either in both parts or in neither.

The reference is the per-pixel median of all fits, used only for this test.
  dark: max |fit - median fit|, in ADU. Must tolerate the normal bias drift
        (~30 ADU on the MI cameras); a shift of hundreds is a defect.
  flat: each fit divided by its own median first (twilight level varies),
        then max |fit - median fit| as a fraction.

Usage: fitcheck.py dark|flat MAXDEV files.lst [names.lst]
  files.lst  staged frames (in_NNNNN.fits); their fits are in_NNNNN-f.fits.
             Rewritten in place without the dropped frames.
  names.lst  original path of each staged frame, same order, for the report.
Report lines start with 'fitcheck:'.
"""

import sys

import numpy as np
from astropy.io import fits

STEP = 16      # the fits are smooth: a coarse grid is enough
MIN_FRAMES = 3


def main():
    if len(sys.argv) < 4:
        print("Usage: fitcheck.py dark|flat MAXDEV files.lst [names.lst]", file=sys.stderr)
        return 2
    kind, maxdev, listfile = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    files = [l.strip() for l in open(listfile) if l.strip()]
    names = files
    if len(sys.argv) > 4:
        names = [l.rstrip('\n') for l in open(sys.argv[4])][:len(files)]
    if len(files) < MIN_FRAMES:
        return 0

    grids = []
    for f in files:
        # IRAF writes the fits as scaled integers (BZERO), which cannot be memmapped
        with fits.open(f.replace('.fits', '-f.fits'), memmap=False) as h:
            g = np.array(h[0].data[::STEP, ::STEP], dtype=np.float64)
        if kind == 'flat':
            g /= np.median(g)
        grids.append(g)
    grids = np.array(grids)
    ref = np.median(grids, axis=0)
    dev = np.abs(grids - ref).max(axis=(1, 2))

    fmt = (lambda x: f"{100 * x:.1f}%") if kind == 'flat' else (lambda x: f"{x:.1f} ADU")
    bad = dev > maxdev
    print(f"fitcheck: {kind} fit deviation from the median fit, {len(files)} frames: "
          f"median {fmt(np.median(dev))}, max {fmt(dev.max())}, limit {fmt(maxdev)}")
    if not bad.any():
        return 0
    if bad.sum() > len(files) // 2:
        # The median reference is only meaningful with a good majority
        print(f"fitcheck: WARNING {bad.sum()}/{len(files)} frames exceed the limit, "
              f"reference unreliable - keeping all")
        return 0

    for i in np.where(bad)[0]:
        print(f"fitcheck: dropped {names[i]} (deviation {fmt(dev[i])})")
    with open(listfile, 'w') as out:
        for f, b in zip(files, bad):
            if not b:
                out.write(f + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
