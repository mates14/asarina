#!/usr/bin/env python3
"""
darkmodel - an exposure-time model of the dark frame from a set of master darks.

The D50 Andor iXon 888 has a ~1235 ADU bias and ~0.2 ADU/s of dark current at -50 C, and
the frame of a kinetic series is now the sum of its 96 readouts (COADDS): it needs 96 x the
dark of one 1.25 s readout, which no master dark is.  One temperature (all the D50 darks
are at -50 C; another temperature is another model), the dark of a pixel p after t seconds:

    D_p(t) = L(t) + A_p + B_p t,        L(t) = M0 + M1 t

  L_k   the level of master dark k: the iterative mode of its pixels (IRAF's salgorithm=mode
        as pyrt.core.aperphot.sky_mode does it, phcat's sky), M0, M1 fitted to L_k(t_k)
        weighted by the frames combined (NCOMBINE),
  A, B  per pixel, weighted least squares of D_p - L_k against t_k (weights NCOMBINE,
        capped at WCAP: a year's combine is not 3000x better than one night's, the pattern
        changes), saturated values out, outliers clipped (CLIP sigma of the pixel's own
        weighted residuals, two passes).

A frame of n readouts of t each (n = COADDS, 1 otherwise) is calibrated by n D(t).

Output: one FITS file, the primary header with M0, M1, the temperature, the inputs and the
diagnostics, image extensions A, B (ADU, ADU/s), RMS (the pixel's weighted residual rms,
ADU) and NUSED (the darks a pixel's fit kept).  --check also fits C t^2 per pixel (QUAD)
to show where a line does not do, and prints how the model predicts darks given with
--test (not in the fit).

Usage:
  asarina-darkmodel -o darkmodel-050-1x1-1024x1024.fits ~/calib/andor46/2025/dark-050-*-1024x1024.fits
  asarina-darkmodel --check --test ~/calib/andor46/2026/dark-050-*-1024x1024.fits -o m.fits 2025/dark-*
"""
import argparse
import os
import sys
import time

import numpy as np
from astropy.io import fits

WCAP = 200          # weight cap (frames combined)
CLIP = 5.0          # [sigma] outlier clip of a pixel's points
SATURATION = 65000  # [ADU] a master-dark value at or above this is saturated (16-bit readout)


def level(img):
    """The iterative mode of the pixels (phcat's sky): mode, stdev."""
    from pyrt.core.aperphot import sky_mode
    v = np.asarray(img, float).ravel()
    v = v[np.isfinite(v)][None, :]
    mode, sd, _, n, _ = sky_mode(v)
    return float(mode[0]), float(sd[0])


def load(paths):
    darks = []
    for p in paths:
        h = fits.getheader(p)
        d = fits.getdata(p).astype(np.float64)
        darks.append(dict(path=p, t=float(h['EXPTIME']), n=int(h.get('NCOMBINE') or 1),
                          temp=h.get('CCD_TEMP'), date=str(h.get('DATE-OBS', ''))[:10], data=d))
    shapes = {d['data'].shape for d in darks}
    if len(shapes) != 1:
        sys.exit(f"darkmodel: the darks differ in shape: {shapes}")
    return sorted(darks, key=lambda d: d['t'])


def wlin(t, y, w):
    """Weighted lines y = a + b t, per pixel: t (k,), y and w (k, npix) -> a, b."""
    s0, s1, s2 = w.sum(0), (w * t[:, None]).sum(0), (w * t[:, None] ** 2).sum(0)
    sy, sty = (w * y).sum(0), (w * t[:, None] * y).sum(0)
    det = s0 * s2 - s1 ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        b = np.where(det > 0, (s0 * sty - s1 * sy) / det, 0.0)
        a = np.where(s0 > 0, (sy - b * s1) / s0, np.nan)
    return a, b


def wquad(t, y, w):
    """Weighted parabolas y = a + b t + c t^2 per pixel -> a, b, c (batched 3x3 solves)."""
    T = np.stack([np.ones_like(t), t, t * t])                       # (3, k)
    N = np.einsum('ik,jk,kp->pij', T, T, w)                         # (npix, 3, 3)
    r = np.einsum('ik,kp->pi', T, w * y)                            # (npix, 3)
    ok = np.linalg.det(N) > 0
    sol = np.full(r.shape, np.nan)
    sol[ok] = np.linalg.solve(N[ok], r[ok][..., None])[..., 0]
    return sol.T


def fit(darks, wcap=WCAP, clip=CLIP, sat=SATURATION, log=print):
    t = np.array([d['t'] for d in darks])
    nk = np.array([d['n'] for d in darks], float)
    shape = darks[0]['data'].shape
    L = np.array([level(d['data'])[0] for d in darks])
    wl = np.minimum(nk, wcap)
    M1, M0 = np.polyfit(t, L, 1, w=np.sqrt(wl))
    lres = L - (M0 + M1 * t)
    for d, l, r in zip(darks, L, lres):
        log(f"  {os.path.basename(d['path'])}: t {d['t']:7.2f} s  N {d['n']:5d}  {d['date']}  "
            f"level {l:9.3f}  - L(t) {r:+.3f}")
    log(f"level L(t) = {M0:.3f} + {M1:.5f} t  (rms {np.sqrt(np.average(lres ** 2, weights=wl)):.3f} ADU)")

    Y = np.stack([d['data'].ravel() for d in darks]) - L[:, None]   # (k, npix)
    raw = np.stack([d['data'].ravel() for d in darks])
    W = np.where(np.isfinite(raw) & (raw < sat), wl[:, None], 0.0)
    del raw
    for it in range(3):
        a, b = wlin(t, Y, W)
        res = Y - (a + b * t[:, None])
        with np.errstate(invalid='ignore', divide='ignore'):
            rms = np.sqrt((W * res ** 2).sum(0) / W.sum(0) * (W > 0).sum(0) / np.maximum((W > 0).sum(0) - 2, 1))
        if it == 2:
            break
        # the pixel's scatter, but not below what the read noise of the darks allows
        floor = 0.5
        W = np.where(np.abs(res) > clip * np.maximum(rms, floor), 0.0, W)
    used = (W > 0).sum(0)
    return dict(M0=M0, M1=M1, L=L, t=t, n=nk, A=a.reshape(shape), B=b.reshape(shape),
                RMS=rms.reshape(shape), NUSED=used.reshape(shape).astype(np.int16), Y=Y, W=W)


def predict(model, t, n=1):
    """The dark of a frame of n readouts of t seconds each (header-free form)."""
    return n * (model['M0'] + model['M1'] * t + model['A'] + model['B'] * t)


def write(path, m, darks, temp):
    h = fits.Header()
    h['DMODEL'] = ('L(t)+A+B*t', 'dark of one readout of t s; n readouts: n x')
    h['M0'] = (m['M0'], '[ADU] level at t=0 (bias)')
    h['M1'] = (m['M1'], '[ADU/s] level growth')
    h['CCD_TEMP'] = (temp, '[C] the darks temperature')
    h['NDARKS'] = (len(darks), 'master darks fitted')
    h['WCAP'] = (WCAP, 'weight cap (frames combined)')
    h['TMIN'] = (min(d['t'] for d in darks), '[s] shortest dark')
    h['TMAX'] = (max(d['t'] for d in darks), '[s] longest dark')
    for i, d in enumerate(darks):
        h[f'DARK{i:02d}'] = (f"{os.path.basename(d['path'])} N{d['n']} {d['date']}"[:68])
    hl = [fits.PrimaryHDU(header=h)]
    for k, unit in (('A', 'ADU'), ('B', 'ADU/s'), ('RMS', 'ADU'), ('NUSED', 'darks')):
        e = fits.ImageHDU(m[k].astype(np.float32) if k != 'NUSED' else m[k], name=k)
        e.header['BUNIT'] = unit
        hl.append(e)
    fits.HDUList(hl).writeto(path, overwrite=True)


def load_model(path):
    with fits.open(path) as f:
        h = f[0].header
        return dict(M0=h['M0'], M1=h['M1'], A=f['A'].data.astype(np.float64),
                    B=f['B'].data.astype(np.float64), header=h)


def check(m, darks, tests, log=print):
    t = m['t']
    k = len(t)
    a, b = m['A'].ravel(), m['B'].ravel()
    hot = np.abs(b) > np.percentile(np.abs(b), 99.9)
    log(f"pixels: B median {np.median(b):+.4f} ADU/s, 99.9% |B| < {np.percentile(np.abs(b), 99.9):.3f}, "
        f"max {b.max():.2f}; residual rms median {np.median(m['RMS']):.3f} ADU, 99% {np.percentile(m['RMS'], 99):.2f}; "
        f"fewer than {k - 2} darks used: {np.mean(m['NUSED'] < k - 2) * 100:.3f}%")
    # does a line do?  the parabola's t^2 term where it is significant, at the longest dark
    qa, qb, qc = wquad(t, m['Y'], m['W'])
    tmax = t.max()
    dq = qc * tmax ** 2
    big = np.abs(dq) > 5 * np.maximum(m['RMS'].ravel(), 0.5)
    log(f"curvature: |C tmax^2| median {np.median(np.abs(dq)):.3f} ADU, > 5 rms in {big.mean() * 100:.3f}% of pixels "
        f"({big[hot].mean() * 100:.1f}% of the 0.1% hottest)")
    for d in tests:
        pred = (m['M0'] + m['M1'] * d['t'] + m['A'] + m['B'] * d['t']).ravel()
        obs = d['data'].ravel()
        lv, _ = level(obs)
        r = obs - pred
        rl = obs - lv - (m['A'] + m['B'] * d['t']).ravel()      # the pattern, its own level
        ok = obs < SATURATION
        p = np.abs(rl[ok])
        h = hot & ok
        log(f"  test {os.path.basename(d['path'])} (N {d['n']}, {d['date']}): level {lv:.2f} vs L(t) "
            f"{m['M0'] + m['M1'] * d['t']:.2f}; pattern residual median |r| {np.median(p):.3f}, 99% {np.percentile(p, 99):.2f}, "
            f"hottest 0.1%: median |r| {np.median(np.abs(rl[h])):.2f} of median {np.median(np.abs(obs[h] - lv)):.1f} ADU")
    return dict(QUAD=dq.reshape(m['A'].shape))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('darks', nargs='+', help='master darks of one camera, temperature, binning and shape')
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--check', action='store_true', help='curvature check (QUAD extension) and the tests')
    ap.add_argument('--test', nargs='*', default=[], help='master darks to predict (not fitted)')
    args = ap.parse_args(argv)
    t0 = time.time()
    darks = load(args.darks)
    temps = [d['temp'] for d in darks if d['temp'] is not None]
    temp = round(float(np.median(temps)), 1) if temps else None
    print(f"darkmodel: {len(darks)} darks, {darks[0]['data'].shape}, t {darks[0]['t']}..{darks[-1]['t']} s, "
          f"temperature {temp} C")
    m = fit(darks)
    write(args.output, m, darks, temp)
    if args.check:
        tests = load(args.test) if args.test else []
        q = check(m, darks, tests)
        with fits.open(args.output, mode='update') as f:
            f.append(fits.ImageHDU(q['QUAD'].astype(np.float32), name='QUAD'))
    print(f"darkmodel: wrote {args.output} ({time.time() - t0:.0f} s)")


if __name__ == '__main__':
    main()
