#!/usr/bin/env python3
"""
rtspix - two-faced (random telegraph signal, RTS) hot pixels: catalogue and per-frame fix.

Some pixels (radiation damage; the FRAM-ORM G2-1000BI CCD47-10 has ~1400 strong ones) have a
dark current that jumps between discrete levels and stays at one for hours to months
(FRAM-ORM: ~0.03 switches per pixel and hour, cold most of the time, hot excursions).  A
master dark or a dark model gets them right only when they happen to be in the state the
darks averaged to.  The difference is a rate (dark current), not an offset.

Build (asarina-rtspix): from the raw darks of one camera, temperature and binning
(a dark whose level is off the others' L(t) by more than LEVTOL ADU, a broken readout, left out),
  r_p = (D_p - median(D)) / t        each dark's rate relative to its own level, as
                                     darkmodel's B (no overscan needed),
  sessions: runs of darks with no gap > GAP s; a pixel keeps its state within one,
  m_sp = median of r_p over session s, its error from the scatter within the session,
  sessions whose level grows faster than the rest by more than GMAX ADU/s are left out
  (FRAM-ORM: the summer evening darks at sunset are light-struck),
  a pixel is an RTS pixel when the best two-class split of its m_sp (Otsu, >= 3 sessions a
  class) separates the classes by > SEP of their scatter and by > SEPN of the session
  error; its levels by splitting the classes again while they still split that well.
Output: FITS, IMAGETYP rtspix, the camera keys of the first dark; BINTABLE RTS with
  X, Y (0-based full-frame numpy indices: data[Y, X]), NLEV, LEVEL (ADU/s, NaN-padded to
  MAXLEV, ascending), FRAC (time in each level), SEP, SEPN, NTR (state changes seen).

Apply (fix_dark): for each catalogued pixel of a frame of total exposure T,
  tot = (data - dark)[p] - median of its 5x5 ring  +  dark[p] - median of the dark's ring
      = the pixel's dark-current excess over its neighbours, whatever the dark was;
  the nearest level b_k (tot / T) is taken when it is told from the next by > NSIG sigma of
  the ring and no star sits on the pixel (the 3x3 neighbours not above the ring), and the
  dark of that pixel becomes median(dark ring) + A0 + b_k T (A0: optional column, the pixel's
  excess at t = 0 [ADU], 0 when absent); otherwise the dark is left as it was.
  The levels are measured the same way in the build: the pixel over the median of its ring.
  KIND (column absent: all R): R as above; S a pixel that changed once (a hot pixel born or
  healed): LEVEL[0] before MJD TCH0, LEVEL[1] after TCH1, set by the frame's date, told per
  frame like R between the two; M always bad.  The pixels not told (and M) are returned as
  'bad': the caller lists them in the frame's BADPIX (pyrt.utils.badpix) and they keep the
  dark they had - the error is below NSIG sigma or a star sits on them - for single-frame
  photometry, while the stacks leave them out.

Usage:
  asarina-rtspix -o rtspix-020-1x1-1062x1026.fits ~/fram-orm/images/2026*/darks/C0/*.fits
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from astropy.io import fits

MAXLEV = 6
TMIN = 60.0       # [s] darks shorter than this do not go into the catalogue
GAP = 1800.0      # [s] a new session after a gap this long
NMIN = 3          # darks a session needs
GMAX = 0.5        # [ADU/s] a session's level growth above the typical: light-struck
SEP = 15.0        # two-class separation / class scatter
SEPN = 20.0       # two-class separation / session error
DTEMP = 0.5       # [C] darks within this of the median temperature
LEVTOL = 15.0     # [ADU] a dark's level this far below (or above, + GMAX t) the others': broken readout
NSIG = 6.0        # apply: levels told apart by this many sigma of the ring


# ---------------------------------------------------------------------------- build

def _header(path):
    h = fits.getheader(path)
    return dict(path=path, t=float(h['EXPTIME']), ctime=float(h.get('CTIME', 0)) + 1e-6 * float(h.get('USEC', 0)),
                temp=h.get('CCD_TEMP'), shape=(h['NAXIS2'], h['NAXIS1']))


def _read(path, rows=None):
    with fits.open(path, memmap=True, do_not_scale_image_data=True) as f:
        h = f[0].header
        d = f[0].data if rows is None else f[0].data[rows]
        return d.astype(np.float32) * np.float32(h.get('BSCALE', 1)) + np.float32(h.get('BZERO', 0))


def _level(path):
    return float(np.median(_read(path)))


def otsu(v, mmin=3):
    """Best two-class split of each column of v (k, npix): dict LO, HI, SD (pooled class
    scatter), FHI, THR."""
    s = np.sort(v, axis=0).astype(np.float64)
    N = s.shape[0]
    c1, c2 = np.cumsum(s, axis=0), np.cumsum(s ** 2, axis=0)
    k = np.arange(1, N)[:, None]
    lo1, lo2 = c1[:-1], c2[:-1]
    hi1, hi2 = c1[-1] - lo1, c2[-1] - lo2
    mlo, mhi = lo1 / k, hi1 / (N - k)
    vlo = np.maximum(lo2 / k - mlo ** 2, 0)
    vhi = np.maximum(hi2 / (N - k) - mhi ** 2, 0)
    w = (k * vlo + (N - k) * vhi) / N
    w[((k < mmin) | (N - k < mmin))[:, 0]] = np.inf
    kb = np.argmin(w, axis=0)
    P = np.arange(s.shape[1])
    return dict(LO=mlo[kb, P], HI=mhi[kb, P], SD=np.sqrt(w[kb, P]), FHI=(N - kb - 1) / N,
                THR=0.5 * (s[kb, P] + s[np.minimum(kb + 1, N - 1), P]))


def split_levels(m, sig, sep=SEP, sepn=SEPN, mmin=3, depth=0):
    """The levels of one pixel's session medians m: split while the split is good."""
    m = np.sort(m)
    if len(m) >= 2 * mmin and depth < 4:
        o = otsu(m[:, None], mmin)
        d = o['HI'][0] - o['LO'][0]
        if d > sep * max(o['SD'][0], 1e-6) and d > sepn * sig:
            lo, hi = m[m <= o['THR'][0]], m[m > o['THR'][0]]
            if len(lo) >= mmin and len(hi) >= mmin:
                return (split_levels(lo, sig, sep, sepn, mmin, depth + 1) +
                        split_levels(hi, sig, sep, sepn, mmin, depth + 1))
    return [(float(np.median(m)), len(m))]


_G = {}


def _block(args):
    y0, y1 = args
    fr, sess = _G['frames'], _G['sessions']
    nx = _G['shape'][1]
    S = len(sess)
    med = np.empty((S, y1 - y0, nx), np.float32)
    err = np.empty((S, y1 - y0, nx), np.float32)
    for s, idx in enumerate(sess):
        c = np.stack([(_read(fr[i]['path'], slice(y0, y1)) - fr[i]['level']) / fr[i]['t'] for i in idx])
        m = np.median(c, axis=0)
        med[s] = m
        err[s] = 1.4826 * np.median(np.abs(c - m), axis=0) / np.sqrt(len(idx))
    return y0, med, err


def build(paths, tmin=TMIN, gap=GAP, nmin=NMIN, gmax=GMAX, sep=SEP, sepn=SEPN, rows=32, nproc=4, log=print):
    with Pool(nproc) as p:
        frames = p.map(_header, paths, chunksize=32)
    temps = [f['temp'] for f in frames if f['temp'] is not None]
    temp = float(np.median(temps)) if temps else None
    shape = max({f['shape'] for f in frames}, key=lambda s: sum(f['shape'] == s for f in frames))
    frames = [f for f in frames if f['t'] >= tmin and f['shape'] == shape and
              (temp is None or f['temp'] is None or abs(f['temp'] - temp) < DTEMP)]
    if not frames:
        sys.exit('rtspix: no darks left (exposure, shape, temperature)')
    frames.sort(key=lambda f: f['ctime'])
    log(f"rtspix: {len(frames)} darks >= {tmin:g} s, {shape}, temperature {temp}")
    with Pool(nproc) as p:
        lv = p.map(_level, [f['path'] for f in frames], chunksize=8)
    for f, l in zip(frames, lv):
        f['level'] = l
    # broken readouts (FRAM-ORM 2026-08..11: bias down to ~340 or up to ~1300, no dark signal):
    # a dark whose level is far from the others' L(t) goes, before the sessions are made
    t = np.array([f['t'] for f in frames]); L = np.array(lv)
    ok = np.ones(len(t), bool)
    for _ in range(3):
        M1, M0 = np.polyfit(t[ok], L[ok], 1)
        res = L - M0 - M1 * t
        ok = (res > -LEVTOL) & (res < LEVTOL + gmax * t)
    if (~ok).any():
        log(f"rtspix: {(~ok).sum()} darks with a broken level (L(t) {M0:.1f} + {M1:.4f} t, "
            f"outside -{LEVTOL:g}..+{LEVTOL:g}+{gmax:g} t ADU) left out")
        frames = [f for f, k in zip(frames, ok) if k]
        lv = [l for l, k in zip(lv, ok) if k]
    # sessions
    sess, cur = [], [0]
    for i in range(1, len(frames)):
        if frames[i]['ctime'] - (frames[i - 1]['ctime'] + frames[i - 1]['t']) > gap:
            sess.append(cur); cur = []
        cur.append(i)
    sess.append(cur)
    sess = [s for s in sess if len(s) >= nmin]
    # light-struck sessions: level growth above the typical
    t = np.array([f['t'] for f in frames]); L = np.array(lv)
    ok = np.ones(len(t), bool)
    for _ in range(3):
        M1, M0 = np.polyfit(t[ok], L[ok], 1)
        ex = (L - M0 - M1 * t) / t
        ok = np.abs(ex - np.median(ex[ok])) < 3 * 1.4826 * np.median(np.abs(ex[ok] - np.median(ex[ok]))) + 0.05
    sx = np.array([np.median(ex[s]) for s in sess])
    lit = sx - np.median(sx) > gmax
    log(f"rtspix: {len(sess)} sessions of >= {nmin} darks, {lit.sum()} light-struck (> {gmax} ADU/s) left out")
    sess = [s for s, b in zip(sess, lit) if not b]
    if len(sess) < 2 * 3:
        sys.exit('rtspix: too few sessions for a catalogue')
    S = len(sess)
    ny, nx = shape
    med = np.empty((S, ny, nx), np.float32)
    err = np.empty((S, ny, nx), np.float32)
    _G.update(frames=frames, sessions=sess, shape=shape)
    with Pool(nproc) as p:
        for y0, m, e in p.imap_unordered(_block, [(y, min(y + rows, ny)) for y in range(0, ny, rows)]):
            med[:, y0:y0 + m.shape[1]] = m
            err[:, y0:y0 + m.shape[1]] = e
    # the session error underestimates (a MAD of a few darks): scale it so that a typical
    # pixel's session-to-session scatter is 1 sigma
    q = np.percentile(med[:, ::4, ::4].reshape(S, -1), [16, 84], axis=0)
    esc = float(np.sqrt(np.median((0.5 * (q[1] - q[0]) / np.maximum(np.median(err[:, ::4, ::4].reshape(S, -1), axis=0), 1e-6)) ** 2)))
    log(f"rtspix: {S} sessions, error scale {esc:.2f}")
    rowsout = []
    for y0 in range(0, ny, rows):
        y1 = min(y0 + rows, ny)
        m = med[:, y0:y1].reshape(S, -1)
        sig = np.maximum(np.median(err[:, y0:y1].reshape(S, -1), axis=0) * esc, 1e-4)
        o = otsu(m)
        d = o['HI'] - o['LO']
        sel = np.nonzero((d > sep * np.maximum(o['SD'], 1e-6)) & (d > sepn * sig))[0]
        for p_ in sel:
            yy, xx = divmod(int(p_), nx)
            yy += y0
            ring = [(yy + dy, xx + dx) for dy, dx in _OUTER if 0 <= yy + dy < ny and 0 <= xx + dx < nx]
            mr = med[:, yy, xx] - np.median(np.stack([med[:, a, c] for a, c in ring], axis=1), axis=1)
            lev = split_levels(mr, sig[p_], sep, sepn)
            if len(lev) < 2:                     # the split was there, the recursion's min class not
                th = o['THR'][p_] - (m[:, p_] - mr).mean()
                lev = [(float(np.median(mr[mr <= th])), int((mr <= th).sum())),
                       (float(np.median(mr[mr > th])), int((mr > th).sum()))]
            st = m[:, p_] > o['THR'][p_]
            amb = np.abs(m[:, p_] - o['THR'][p_]) < 0.25 * d[p_]
            ss = st[~amb]
            ntr = int((ss[1:] != ss[:-1]).sum())
            rowsout.append((xx, yy, lev[:MAXLEV], float(d[p_] / max(o['SD'][p_], 1e-6)), float(d[p_] / sig[p_]), ntr))
    log(f"rtspix: {len(rowsout)} RTS pixels (SEP > {sep:g}, SEPN > {sepn:g})")
    return dict(rows=rowsout, frames=frames, sessions=sess, temp=temp, shape=shape, esc=esc,
                sep=sep, sepn=sepn, tmin=tmin, gap=gap, gmax=gmax)


SKIP_KEYS = {'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'EXTEND', 'BZERO', 'BSCALE', 'EXPTIME',
             'EXPOSURE', 'NCOMBINE', 'DATE-OBS', 'DATE', 'CTIME', 'USEC', 'JD', 'IMAGETYP', 'COMMENT',
             'HISTORY', 'IRAF-TLM', 'ORIGIN', 'OBJECT', 'SCRIPT', 'CONTINUE', 'AVERAGE', 'STDEV', ''}


def write(path, b):
    h = fits.Header()
    h0 = fits.getheader(b['frames'][0]['path'])
    for c in h0.cards:
        if c.keyword not in SKIP_KEYS and not c.keyword.startswith(('NAXIS', 'MOON', 'SCR', 'WEATHER', 'UPS', 'HIERARCH')):
            try:
                h.append(c)
            except Exception:
                pass
    h['IMAGETYP'] = ('rtspix', 'RTS (two-faced) hot pixel catalogue')
    if b['temp'] is not None:
        h['CCD_TEMP'] = (round(b['temp'], 1), '[C] the darks temperature')
    h['RTSNPIX'] = (len(b['rows']), 'RTS pixels')
    h['NDARKS'] = (len(b['frames']), 'darks read')
    h['NSESS'] = (len(b['sessions']), 'sessions used')
    h['DATE0'] = (fits.getheader(b['frames'][0]['path']).get('DATE-OBS', '')[:10], 'first dark')
    h['DATE1'] = (fits.getheader(b['frames'][-1]['path']).get('DATE-OBS', '')[:10], 'last dark')
    for k in ('sep', 'sepn', 'tmin', 'gap', 'gmax', 'esc'):
        h[k.upper()[:8]] = b[k]
    n = len(b['rows'])
    lev = np.full((n, MAXLEV), np.nan, np.float32)
    frac = np.zeros((n, MAXLEV), np.float32)
    for i, r in enumerate(b['rows']):
        tot = sum(c for _, c in r[2]) or 1
        for j, (v, c) in enumerate(r[2]):
            lev[i, j] = v; frac[i, j] = c / tot
    cols = [fits.Column('X', 'J', array=np.array([r[0] for r in b['rows']], np.int32)),
            fits.Column('Y', 'J', array=np.array([r[1] for r in b['rows']], np.int32)),
            fits.Column('NLEV', 'I', array=np.array([len(r[2]) for r in b['rows']], np.int16)),
            fits.Column('LEVEL', f'{MAXLEV}E', unit='ADU/s', array=lev),
            fits.Column('FRAC', f'{MAXLEV}E', array=frac),
            fits.Column('SEP', 'E', array=np.array([r[3] for r in b['rows']], np.float32)),
            fits.Column('SEPN', 'E', array=np.array([r[4] for r in b['rows']], np.float32)),
            fits.Column('NTR', 'J', array=np.array([r[5] for r in b['rows']], np.int32))]
    t = fits.BinTableHDU.from_columns(cols, name='RTS')
    t.header['COMMENT'] = 'X, Y: 0-based full-frame indices, data[Y, X]'
    fits.HDUList([fits.PrimaryHDU(header=h), t]).writeto(path, overwrite=True)


# ---------------------------------------------------------------------------- apply

def load(path):
    with fits.open(path) as f:
        t = f['RTS'].data
        n = len(t)
        names = t.columns.names
        kind = (np.char.strip(np.asarray(t['KIND']).astype(str)) if 'KIND' in names else np.full(n, 'R'))
        tch0 = t['TCH0'].astype(np.float64) if 'TCH0' in names else np.full(n, np.nan)
        tch1 = t['TCH1'].astype(np.float64) if 'TCH1' in names else np.full(n, np.nan)
        a0 = t['A0'].astype(np.float64) if 'A0' in names else np.zeros(n)
        return dict(x=t['X'].astype(int), y=t['Y'].astype(int), level=t['LEVEL'].astype(np.float64),
                    kind=kind, tch0=tch0, tch1=tch1, a0=a0, header=f[0].header)


COVER = 30.0      # [d] a catalogue is used this far outside its DATE0..DATE1


def covers(cat, mjd, margin=COVER):
    """Whether the frame (mjd) is within the catalogue's DATE0..DATE1 (+- margin days);
    True when the header has no dates."""
    from astropy.time import Time
    h = cat.get('header') or {}
    d0, d1 = str(h.get('DATE0', '') or ''), str(h.get('DATE1', '') or '')
    try:
        if d0 and mjd < Time(d0[:10]).mjd - margin:
            return False
        if d1 and mjd > Time(d1[:10]).mjd + 1 + margin:
            return False
    except ValueError:
        return True
    return True


def write_table(path, header, x, y, level, kind, tch0=None, tch1=None, a0=None, extra=None):
    """A catalogue from any search: KIND R (switches, levels told per frame), S (one step:
    LEVEL[0] before TCH0, LEVEL[1] after TCH1, MJD; between them told per frame like R),
    M (always bad); a0: [ADU] the pixel's excess at t = 0.  extra: {name: (format, array)}."""
    n = len(x)
    lev = np.full((n, MAXLEV), np.nan, np.float32)
    for i, l in enumerate(level):
        l = np.asarray(l, np.float32)[:MAXLEV]
        lev[i, :len(l)] = l
    cols = [fits.Column('X', 'J', array=np.asarray(x, np.int32)),
            fits.Column('Y', 'J', array=np.asarray(y, np.int32)),
            fits.Column('NLEV', 'I', array=np.isfinite(lev).sum(1).astype(np.int16)),
            fits.Column('LEVEL', f'{MAXLEV}E', unit='ADU/s', array=lev),
            fits.Column('KIND', '1A', array=np.asarray(kind, 'U1')),
            fits.Column('TCH0', 'D', unit='MJD', array=np.full(n, np.nan) if tch0 is None else np.asarray(tch0, float)),
            fits.Column('TCH1', 'D', unit='MJD', array=np.full(n, np.nan) if tch1 is None else np.asarray(tch1, float)),
            fits.Column('A0', 'E', unit='ADU', array=np.zeros(n, np.float32) if a0 is None else np.asarray(a0, np.float32))]
    for k, (fmt, arr) in (extra or {}).items():
        cols.append(fits.Column(k, fmt, array=arr))
    t = fits.BinTableHDU.from_columns(cols, name='RTS')
    t.header['COMMENT'] = 'X, Y: 0-based full-frame indices, data[Y, X]'
    h = fits.Header(header)
    h['IMAGETYP'] = ('rtspix', 'RTS (two-faced) hot pixel catalogue')
    h['RTSNPIX'] = (n, 'catalogued pixels')
    fits.HDUList([fits.PrimaryHDU(header=h), t]).writeto(path, overwrite=True)


_RING = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if max(abs(dy), abs(dx)) == 2 or
         (abs(dy) <= 1 and abs(dx) <= 1 and (dy or dx))]
_OUTER = [o for o in _RING if max(abs(o[0]), abs(o[1])) == 2]
_INNER = [o for o in _RING if max(abs(o[0]), abs(o[1])) == 1]


def _gather(img, y, x, offs):
    ny, nx = img.shape
    return np.stack([img[np.clip(y + dy, 0, ny - 1), np.clip(x + dx, 0, nx - 1)] for dy, dx in offs], axis=1)


def fix_dark(data, dark, cat, exptime, origin=(0, 0), nsig=NSIG, sat=None, mjd=None):
    """The dark with each catalogued RTS pixel at the level the frame shows it in.

    data, dark: the raw frame and its dark (same shape; a window of the full frame starting
    at origin = (row, col)); exptime: the frame's total exposure [s]; mjd: the frame's time,
    for the KIND S pixels (one step: the level of the frame's side of it; told per frame like
    an R pixel when the frame falls between TCH0 and TCH1, or without mjd).
    Returns (dark', dict(n=pixels in frame, fixed=, unknown=, state=level index or -1 per
    catalogued pixel in frame, bad=(y, x) frame indices of the pixels whose dark is not known:
    R and S not told, M - for the frame's BADPIX))."""
    y = cat['y'] - origin[0]
    x = cat['x'] - origin[1]
    ny, nx = data.shape
    inside = (y >= 2) & (y < ny - 2) & (x >= 2) & (x < nx - 2)
    y, x, lev = y[inside], x[inside], cat['level'][inside]
    n_all = len(cat['x'])
    kind = np.asarray(cat.get('kind', np.full(n_all, 'R')))[inside]
    tch0 = np.asarray(cat.get('tch0', np.full(n_all, np.nan)))[inside]
    tch1 = np.asarray(cat.get('tch1', np.full(n_all, np.nan)))[inside]
    a0 = np.asarray(cat.get('a0', np.zeros(n_all)))[inside]
    out = dark.astype(np.float64, copy=True)
    if len(y) == 0:
        return out, dict(n=0, fixed=0, unknown=0, state=np.zeros(0, int), bad=(y, x))
    # S pixels with the frame clearly on one side of the step: the level by the date
    dated = np.zeros(len(y), bool); dk = np.zeros(len(y), int)
    if mjd is not None:
        s = kind == 'S'
        before = s & (mjd < tch0); after = s & (mjd > tch1)
        dated = before | after
        dk = np.where(after, 1, 0)
    masked = kind == 'M'
    r = data.astype(np.float64) - dark
    rring = _gather(r, y, x, _OUTER)
    bg = np.median(rring, axis=1)
    sig = np.maximum(1.4826 * np.median(np.abs(rring - bg[:, None]), axis=1), 1.0)
    dring = np.median(_gather(dark, y, x, _OUTER), axis=1)
    dexc = dark[y, x] - dring                                  # what the dark gives the pixel
    tot = (r[y, x] - bg) + dexc                                # the pixel's excess, ADU
    inner = _gather(r, y, x, _INNER) - bg[:, None]
    star = np.median(inner, axis=1) > 3 * sig                  # a star (or a cluster) on it
    if sat is not None:
        star |= _gather(data, y, x, _INNER + [(0, 0)]).max(axis=1) >= sat
    exp_ = lev * exptime                                       # (n, MAXLEV), NaN-padded
    dist = np.abs(tot[:, None] - exp_)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    o = np.argsort(dist, axis=1)
    k1, k2 = o[:, 0], o[:, 1]
    P = np.arange(len(y))
    gap_ = np.abs(exp_[P, k1] - exp_[P, k2])
    gap_ = np.where(np.isfinite(gap_), gap_, 0)
    good = (~star) & (gap_ > nsig * sig) & (dist[P, k1] < 0.5 * gap_ - 2 * sig) & ~masked
    k1 = np.where(dated, dk, k1)
    good = (good | dated) & ~masked
    # absolute: the ring of the dark + A0 + level T (moving the dark by a difference of levels
    # from the state it shows did worse on FRAM-ORM: its master dark is a blend of states)
    out[y[good], x[good]] = dring[good] + a0[good] + exp_[P[good], k1[good]]
    state = np.where(good, k1, -1)
    return out, dict(n=len(y), fixed=int(good.sum()), unknown=int((~good).sum()), state=state,
                     bad=(y[~good], x[~good]))


# ---------------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('darks', nargs='+', help='raw darks of one camera, temperature and binning')
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--tmin', type=float, default=TMIN, help=f'[s] shortest dark used ({TMIN:g})')
    ap.add_argument('--gap', type=float, default=GAP, help=f'[s] gap that starts a new session ({GAP:g})')
    ap.add_argument('--gmax', type=float, default=GMAX, help=f'[ADU/s] light-struck session cut ({GMAX})')
    ap.add_argument('--sep', type=float, default=SEP); ap.add_argument('--sepn', type=float, default=SEPN)
    ap.add_argument('--nproc', type=int, default=4)
    args = ap.parse_args(argv)
    t0 = time.time()
    b = build(args.darks, tmin=args.tmin, gap=args.gap, gmax=args.gmax, sep=args.sep, sepn=args.sepn,
              nproc=args.nproc)
    write(args.output, b)
    print(f"rtspix: wrote {args.output} ({time.time() - t0:.0f} s)")


if __name__ == '__main__':
    main()
