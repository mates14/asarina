#!/usr/bin/env python3
"""
kinstep - progressive light curve of a sequence of Andor kinetic series, ~12 points per step.

The EP/GRB initial sequence at D50 is up to 12 kinetic series of 96 x 1.25 s.  When
the k-th series has arrived, the whole k x 120 s are split into 12 bins of 8k
sub-frames.  Only the "trivial" steps k = 1, 2, 3, 4, 6, 12 are done here: their bins
never cross a series boundary (96 is divisible by 8k), so every bin lies on the pixel
grid of one series and takes that series' astrometric solution (the reference ECSV of
its 120 s mean, solved by the pipeline).

Per step:
  1. each series: calibrated sum of its sub-frames (with the series WCS) and its bins,
  2. pyrt-combine -u of the series sums -> deep.fits and skel.hdr (the series may be
     shifted, the telescope recentres after the first frames),
  3. deep.fits: phcat -> cat2det -> dophot (spatial response of the first series' reference
     held fixed, zeropoint free) -> the deep catalogue (faint candidates),
  4. every bin: pyrt-mproject onto skel.hdr, phcat -D deep.fits (forced photometry at the
     deep positions, SExtractor dual-image mode), cat2det, dophot with the deep response
     held fixed, zeropoint free,
  5. per-bin ECSVs in <outdir>/ for transients.py (flare search on ~12 points).

Usage (asarina-kinstep; asarina-kinseries runs it in real time, see kinseries.py):
  kinstep.py S1-kin.fits                         # step 1: 12 bins of 8
  kinstep.py S1-kin.fits S2-kin.fits S3-kin.fits S4-kin.fits     # step 4: bins of 32
  kinstep.py --step 2 S*-kin.fits                # the first 2 series only

Series inputs: the *-kin.fits cube (primary = mean, extensions = the sub-frames) or the
raw RA.fits with its KINFILE dump; the reference ECSV (WCS) is found by the frame stem
(next to the input, in its parent directory, ~/phdb-rt, ...), or given by --ref (in order).
"""

import argparse
import concurrent.futures as cf
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
from astropy.io import fits
from astropy.table import Table

from asarina.pipeline.kinphot import (stem_of, find_ref, find_raw, find_kinfile, load_calibration,
                                      ref_wcs_and_response, fixed_terms, subframe_header, WCS_RE)

log = logging.getLogger("kinstep")

POINTS = 12
TRIVIAL = (1, 2, 3, 4, 6, 12)


# ---------------------------------------------------------------- input

class Series:
    """One kinetic series: header, sub-frame access, reference ECSV."""

    def __init__(self, path, ref=None):
        self.path = path
        self.stem = stem_of(path)
        if path.endswith('-kin.fits'):
            # int16 + BZERO 32768 extensions: memmap the raw values, scale by hand
            self.hdul = fits.open(path, memmap=True, do_not_scale_image_data=True)
            self.hdr = self.hdul[0].header
            self.nfr = len(self.hdul) - 1
            self._get = lambda i: (self.hdul[1 + i].data.astype(np.float32) * self.hdul[1 + i].header.get('BSCALE', 1)
                                   + self.hdul[1 + i].header.get('BZERO', 0))
            self.raw_path = path
        else:
            # real time: the queue path the frame had in imgproc, archived by now
            raw = path if path.endswith('-RA.fits') and os.path.exists(path) else find_raw(self.stem)
            if raw is None:
                raise FileNotFoundError(f"raw RA.fits for {self.stem} not found")
            self.raw_path = raw
            self.hdr = fits.getheader(raw)
            nx, ny = self.hdr['NAXIS1'], self.hdr['NAXIS2']
            kin = np.memmap(find_kinfile(self.hdr), np.uint16, 'r')
            self.nfr = kin.size // (nx * ny)
            kin = kin[:self.nfr * nx * ny].reshape(self.nfr, ny, nx)
            self._get = lambda i: kin[i]
        self.ref = ref or self._find_ref()
        if self.ref is None:
            raise FileNotFoundError(f"reference ECSV for {self.stem} not found, use --ref")
        self.wcs, self.response, self.ref_meta = ref_wcs_and_response(self.ref)
        self.cycle = float(self.hdr['KINCYCLE'])
        self.t0 = self.hdr['CTIME'] + self.hdr.get('USEC', 0) / 1e6

    def _find_ref(self):
        d = os.path.dirname(os.path.abspath(self.path))
        for where in (d, os.path.dirname(d)):
            hits = sorted(glob.glob(f"{where}/{self.stem}-*.ecsv"))
            if hits:
                return hits[0]
        return find_ref(self.stem)

    def frame(self, i):
        return np.asarray(self._get(i), dtype=np.float32)


def _cd(wcs):
    if 'CD1_1' in wcs:
        return np.array([[wcs['CD1_1'], wcs.get('CD1_2', 0.0)], [wcs.get('CD2_1', 0.0), wcs['CD2_2']]], float)
    pc = np.array([[wcs.get('PC1_1', 1.0), wcs.get('PC1_2', 0.0)], [wcs.get('PC2_1', 0.0), wcs.get('PC2_2', 1.0)]], float)
    return np.diag([wcs.get('CDELT1', 1.0), wcs.get('CDELT2', 1.0)]) @ pc


def consistent_wcs(series, tol=0.01):
    """Drop the series whose astrometric solution (the realtime pipeline's) disagrees with the
    others in scale or rotation by more than tol: mapped onto the common grid its stars would
    smear and its bins measure nothing (53294, 2026-03-22: one series solved 10% off in scale).
    The solution most of the series agree with wins, the earliest on a tie."""
    if len(series) < 2:
        return series
    cds = [_cd(s.wcs) for s in series]
    def close(a, b):
        return np.linalg.norm(a - b) / np.linalg.norm(b) < tol
    votes = [sum(close(a, b) for b in cds) for a in cds]
    best = cds[int(np.argmax(votes))]
    keep = [s for s, cd in zip(series, cds) if close(cd, best)]
    for s, cd in zip(series, cds):
        if not close(cd, best):
            sc = np.sqrt(abs(np.linalg.det(cd)) / abs(np.linalg.det(best)))
            log.warning(f"{s.stem}: astrometric solution of {s.ref} off by {100 * (sc - 1):+.1f}% in scale "
                        f"(or rotated) from the other series, series left out of this step")
    return keep


CROP_MARGIN = 8   # [px] inside the common footprint


def crop_common(deep_path, series, skel_path):
    """Crop deep_path to the pixels covered by every series (the inner box of each series'
    footprint, CROP_MARGIN inside) and write skel_path with its grid; returns its header."""
    from astropy.wcs import WCS
    with fits.open(deep_path, mode='update') as d:
        hd = d[0].header
        wd = WCS(hd)
        ny, nx = d[0].data.shape
        x0, x1, y0, y1 = 0.0, nx - 1.0, 0.0, ny - 1.0
        for s in series:
            w = WCS(fits.Header(s.wcs))
            sx, sy = s.hdr['NAXIS1'], s.hdr['NAXIS2']
            ex = np.r_[np.linspace(0, sx - 1, 9), np.full(9, sx - 1), np.linspace(sx - 1, 0, 9), np.zeros(9)]
            ey = np.r_[np.zeros(9), np.linspace(0, sy - 1, 9), np.full(9, sy - 1), np.linspace(sy - 1, 0, 9)]
            ra, dec = w.all_pix2world(ex, ey, 0)
            x, y = wd.all_world2pix(ra, dec, 0)
            # inner box of this footprint from its corners (as framestep: any of the series may
            # be turned by 180 deg after a meridian flip, slight rotations take the inner values)
            cx, cy = np.sort(x[[0, 9, 18, 27]]), np.sort(y[[0, 9, 18, 27]])
            x0, x1 = max(x0, cx[1]), min(x1, cx[2])
            y0, y1 = max(y0, cy[1]), min(y1, cy[2])
        ix0, iy0 = max(0, int(np.ceil(x0 + CROP_MARGIN))), max(0, int(np.ceil(y0 + CROP_MARGIN)))
        ix1, iy1 = min(nx, int(np.floor(x1 - CROP_MARGIN)) + 1), min(ny, int(np.floor(y1 - CROP_MARGIN)) + 1)
        if ix1 - ix0 < 100 or iy1 - iy0 < 100:
            log.warning(f"common footprint {ix1 - ix0}x{iy1 - iy0} px only, deep.fits left uncropped")
            ix0, iy0, ix1, iy1 = 0, 0, nx, ny
        if (ix0, iy0, ix1, iy1) != (0, 0, nx, ny):
            d[0].data = d[0].data[iy0:iy1, ix0:ix1]
            hd['CRPIX1'] -= ix0
            hd['CRPIX2'] -= iy0
            log.info(f"deep.fits cropped to the common footprint of {len(series)} series: "
                     f"[{ix0}:{ix1}, {iy0}:{iy1}] of {nx}x{ny}")
        hd = d[0].header.copy()
    hd['NAXIS1'], hd['NAXIS2'] = ix1 - ix0, iy1 - iy0
    sk = WCS(hd).to_header(relax=True)
    sk['NAXIS'] = 2
    sk['NAXIS1'], sk['NAXIS2'] = hd['NAXIS1'], hd['NAXIS2']
    sk.totextfile(skel_path, overwrite=True)
    return hd


def calibrate_sum(series, a, b, dark, flat, crop):
    acc = np.zeros(dark.shape, np.float32)
    for i in range(a, b):
        acc += series.frame(i)
    acc = (acc - (b - a) * dark) / flat
    return acc[crop] if crop is not None else acc


def mark_accumulated(hdr, n):
    """A sum of n readouts, as Andor's accumulate mode: phcat scales the saturation and
    the read noise by ACCNUM (gain unchanged)."""
    hdr['ACQMODE'] = ('Accumulate', 'sum of kinetic sub-frames (kinstep)')
    hdr['ACCNUM'] = (int(n), 'readouts summed')


def centroids(img, xy, r=5):
    """Intensity-weighted centroids (0-based) in (2r+1)^2 boxes around xy, two iterations."""
    out = np.full((len(xy), 2), np.nan)
    ny, nx = img.shape
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    for n, (x, y) in enumerate(xy):
        for _ in range(2):
            ix, iy = int(round(x)), int(round(y))
            if ix < r or iy < r or ix >= nx - r or iy >= ny - r:
                break
            box = img[iy - r:iy + r + 1, ix - r:ix + r + 1]
            edge = np.concatenate([box[0], box[-1], box[:, 0], box[:, -1]])
            w = np.clip(box - np.median(edge), 0, None)
            if w.sum() <= 0:
                break
            x, y = ix + (w * xx).sum() / w.sum(), iy + (w * yy).sum() / w.sum()
        else:
            out[n] = x, y
    return out


def guide_stars(ref_path, n=40):
    """Bright, unflagged, not too bright stars of the reference ECSV (0-based x, y)."""
    t = Table.read(ref_path, format='ascii.ecsv')
    m = np.ones(len(t), bool)
    for c, f in (('FLAGS', lambda v: v == 0), ('MAGERR_AUTO', lambda v: v < 0.02)):
        if c in t.colnames:
            m &= np.asarray(f(np.asarray(t[c])), bool)
    t = t[m]
    t.sort('MAGERR_AUTO')
    t = t[len(t) // 10:][:n]    # skip the brightest tenth (saturation in a sum)
    return np.c_[np.asarray(t['X_IMAGE'], float) - 1, np.asarray(t['Y_IMAGE'], float) - 1]


def bin_shift(img, ref_cen, xy):
    """Shift (dx, dy) of a bin against the series sum, the median over the guide stars."""
    c = centroids(img, xy)
    d = c - ref_cen
    ok = np.all(np.isfinite(d), 1)
    if ok.sum() < 5:
        return 0.0, 0.0, ok.sum()
    dx, dy = np.median(d[ok], 0)
    return float(dx), float(dy), int(ok.sum())


def write_image(path, data, series, a, b, expt1, shift=(0.0, 0.0)):
    hdr = subframe_header(series.hdr, series.wcs, None, series.t0 + a * series.cycle,
                          expt1 * (b - a), a, b - a, series.ref, series.raw_path)
    for k in [k for k in hdr if k.startswith('NAXIS') or k in ('BITPIX', 'EXTEND', 'SIMPLE')]:
        del hdr[k]
    hdr['KINSTEM'] = (series.stem, 'kinetic series')
    mark_accumulated(hdr, b - a)
    # the field drifts within a series (tracking, wind): the series WCS shifted by the
    # measured offset against the series sum
    hdr['KINDX'] = (round(shift[0], 3), '[px] shift against the series sum')
    hdr['KINDY'] = (round(shift[1], 3), '[px] shift against the series sum')
    hdr['CRPIX1'] = hdr['CRPIX1'] + shift[0]
    hdr['CRPIX2'] = hdr['CRPIX2'] + shift[1]
    fits.PrimaryHDU(data=data.astype(np.float32), header=hdr).writeto(path, overwrite=True)


# ---------------------------------------------------------------- tools

def run(cmd, cwd, what, logf=None):
    t = time.time()
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if logf is not None:
        with open(logf, 'a') as f:
            f.write(f"$ {' '.join(cmd)}\n{r.stdout}\n{r.stderr}\n")
    if r.returncode != 0:
        raise RuntimeError(f"{what} failed: " + (r.stderr or r.stdout or '').strip()[-600:])
    return r, time.time() - t


SKIP = ('SIMPLE', 'BITPIX', 'EXTEND', 'BZERO', 'BSCALE', 'BLANK', 'COMMENT', 'HISTORY', '')


def copy_keys(src, dst):
    """Non-WCS keywords of src into dst (standard 8-char ones; RTS2's HIERARCH/odd cards stay out)."""
    for card in src.cards:
        kw = card.keyword
        if kw in SKIP or len(kw) > 8 or not kw.replace('-', '').replace('_', '').isalnum() \
                or WCS_RE.match(kw) or kw.startswith('NAXIS') or kw in dst:
            continue
        try:
            card.verify('exception')
            dst[kw] = (card.value, card.comment)
        except Exception:
            pass


def project(src, dst, skel, cwd, logf):
    """Reproject src onto skel.hdr, keep its non-WCS header (time, exposure, filter, ...)."""
    run(['pyrt-mproject', src, dst, skel], cwd, 'mproject', logf)
    with fits.open(f"{cwd}/{src}") as s, fits.open(f"{cwd}/{dst}", mode='update') as d:
        copy_keys(s[0].header, d[0].header)
        d[0].data = np.nan_to_num(d[0].data, nan=0.0).astype(np.float32)
    area = f"{cwd}/{dst[:-5]}_area.fits"
    if os.path.exists(area):
        os.remove(area)


class Resampler:
    """Map of the skel.hdr pixels into one series' pixels, computed once and applied to all
    its bins (bilinear interpolation times the pixel-area ratio): the same mapping that
    mProjectPP does per image, ~50x faster.  The WCS is evaluated on a coarse grid
    (every STEP px) and interpolated, it is smooth."""
    STEP = 16

    def __init__(self, series_wcs, skel_path):
        from astropy.wcs import WCS
        from scipy.ndimage import map_coordinates
        self._mc = map_coordinates
        skel = fits.Header.fromtextfile(skel_path)
        nx, ny = skel['NAXIS1'], skel['NAXIS2']
        w_out, w_in = WCS(skel), WCS(fits.Header(series_wcs))
        gx = np.arange(0, nx + self.STEP, self.STEP, dtype=float)
        gy = np.arange(0, ny + self.STEP, self.STEP, dtype=float)
        GX, GY = np.meshgrid(gx, gy)
        ra, dec = w_out.all_pix2world(GX, GY, 0)
        X, Y = w_in.all_world2pix(ra, dec, 0)
        yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
        cy, cx = yy / self.STEP, xx / self.STEP
        self.x = map_coordinates(X, [cy, cx], order=3, mode='nearest').astype(np.float32)
        self.y = map_coordinates(Y, [cy, cx], order=3, mode='nearest').astype(np.float32)
        # flux conservation: input pixels per output pixel
        dxdi, dxdj = np.gradient(self.x)
        dydi, dydj = np.gradient(self.y)
        self.jac = np.abs(dxdj * dydi - dxdi * dydj).astype(np.float32)

    def __call__(self, img, dx=0.0, dy=0.0):
        """img: a bin whose stars are shifted by (dx, dy) against the series WCS."""
        out = self._mc(img, [self.y + dy, self.x + dx], order=1, mode='constant', cval=0.0, prefilter=False)
        return (out * self.jac).astype(np.float32)


def resample_fast(rs, src, dst, skel, cwd):
    with fits.open(f"{cwd}/{src}") as s:
        hdr = fits.Header.fromtextfile(f"{cwd}/{skel}")
        for k in ('SAMPLING', 'SCALE', 'NIMAGES', 'NAXIS', 'NAXIS1', 'NAXIS2', 'BITPIX'):
            hdr.remove(k, ignore_missing=True)
        copy_keys(s[0].header, hdr)
        data = rs(s[0].data.astype(np.float32), s[0].header.get('KINDX', 0.0), s[0].header.get('KINDY', 0.0))
        for k in ('KINDX', 'KINDY'):
            hdr[k] = s[0].header.get(k, 0.0)
        fits.PrimaryHDU(data=data, header=hdr).writeto(f"{cwd}/{dst}", overwrite=True)


def photometry(name, cwd, dophot_cmd, phcat_args, logf):
    base = name[:-5]
    _, t1 = run(['pyrt-phcat'] + phcat_args + [name], cwd, 'phcat', logf)
    _, t2 = run(['pyrt-cat2det', name], cwd, 'cat2det', logf)
    r, t3 = run(dophot_cmd + [base + '.det'], cwd, 'dophot', logf)
    if not os.path.exists(f"{cwd}/{base}.ecsv"):
        raise RuntimeError("dophot wrote no ECSV: " + r.stdout.strip()[-400:])
    return t1, t2, t3


def merge_terms(terms, fterms):
    """Fixed per-filter (colour) terms win over the same terms in a response."""
    fterms = (fterms or '').strip().strip('"\'')
    if not fterms:
        return terms
    names = {t.lstrip('#&@*').split('=')[0] for t in fterms.split(',')}
    return ','.join([t for t in terms.split(',') if t and t.lstrip('#&@*').split('=')[0] not in names]
                    + [fterms])


def config_filter_terms(hdr):
    try:
        from asarina.config import load_config, filter_terms
        cfg = load_config(None, hdr.get('CCD_NAME'))
        return filter_terms(cfg).get(str(hdr.get('FILTER', '')).lower(), '')
    except Exception as e:
        log.warning(f"cannot read the asarina config: {e}")
        return ''


# ---------------------------------------------------------------- main

def main():
    os.environ.setdefault('TERM', 'xterm')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('series', nargs='+', help='kinetic series in time order (*-kin.fits or RA.fits)')
    ap.add_argument('--step', type=int, help='use the first STEP series (default: all given)')
    ap.add_argument('--points', type=int, default=POINTS, help=f'light-curve points per step (default {POINTS})')
    ap.add_argument('--nsub', type=int, default=96, help='sub-frames used per series (default 96, the rest ignored)')
    ap.add_argument('--ref', action='append', help='reference ECSV per series (repeat, in order)')
    ap.add_argument('--dark', help='dark (of the sub-frame exposure), default as asarina picks')
    ap.add_argument('--flat', help='flat, default as asarina picks')
    ap.add_argument('--filter-terms', default=None,
                    help="fixed colour terms for dophot (default filter_terms_<filter> of the asarina config)")
    ap.add_argument('-i', '--idlimit', type=int, default=2, help='dophot idlimit (default 2)')
    ap.add_argument('-o', '--outdir', help='output directory (default ./<first stem>-step<k>)')
    ap.add_argument('-j', '--jobs', type=int, default=2, help='parallel bins (default 2)')
    ap.add_argument('-I', '--noiraf', action='store_true', help='phcat -I (SExtractor magnitudes, faster)')
    ap.add_argument('-E', '--engine', choices=('numpy', 'iraf', 'auto'), default='numpy',
                    help='phcat aperture photometry: numpy (IRAF phot reimplemented, fast; default) or IRAF')
    ap.add_argument('--montage', action='store_true',
                    help='reproject every bin with pyrt-mproject (slow) instead of the per-series map')
    ap.add_argument('-k', '--keep', action='store_true', help='keep the work files (FITS, catalogues, logs)')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    for name in ('asarina', 'pyrt'):
        logging.getLogger(name).setLevel(logging.WARNING)
    t_all = time.time()

    k = args.step or len(args.series)
    if k > len(args.series):
        ap.error(f"--step {k} needs {k} series, {len(args.series)} given")
    if args.points % k or args.nsub % (args.points // k):
        ap.error(f"step {k}: bins would cross a series boundary (trivial steps: {TRIVIAL})")
    per_series = args.points // k            # bins per series
    nbin = args.nsub // per_series           # sub-frames per bin
    refs = (args.ref or []) + [None] * k
    series = [Series(p, r) for p, r in zip(args.series[:k], refs)]
    for s in series:
        if s.nfr < args.nsub:
            raise SystemExit(f"{s.stem}: only {s.nfr} sub-frames, --nsub {args.nsub}")
    series = consistent_wcs(series)
    expt1 = series[0].cycle
    log.info(f"step {k}: {k} series x {per_series} bins of {nbin} sub-frames ({nbin * expt1:.1f} s)")

    outdir = os.path.abspath(args.outdir or f"{series[0].stem}-step{k}")
    work = f"{outdir}/work"
    os.makedirs(work, exist_ok=True)
    logf = f"{work}/kinstep.log"

    # calibration (one dark/flat for the whole sequence: same camera, same setup)
    if args.dark and args.flat:
        dark = fits.getdata(args.dark).astype(np.float32)
        flat = fits.getdata(args.flat).astype(np.float32)
        crop = None
    else:
        dark, flat, crop = load_calibration(series[0].raw_path, series[0].hdr, expt1)

    # 1. series sums and bins
    t = time.time()
    sums, bins = [], []
    for si, s in enumerate(series):
        imgs = [calibrate_sum(s, bi * nbin, (bi + 1) * nbin, dark, flat, crop) for bi in range(per_series)]
        tot = np.sum(imgs, axis=0)
        sname = f"S{si + 1:02d}-{s.stem}.fits"
        write_image(f"{work}/{sname}", tot, s, 0, args.nsub, expt1)
        sums.append(sname)
        xy = guide_stars(s.ref)
        ref_cen = centroids(tot, xy)
        shifts = []
        for bi, img in enumerate(imgs):
            a, b = bi * nbin, (bi + 1) * nbin
            dx, dy, ng = bin_shift(img, ref_cen, xy) if per_series > 1 else (0.0, 0.0, 0)
            shifts.append("%+.2f,%+.2f" % (dx, dy))
            ts = s.t0 + a * s.cycle
            name = (datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y%m%d%H%M%S')
                    + f"-{int((ts % 1) * 1000):03d}-{s.hdr.get('FILTER', 'UNK')}-k{a:03d}b{nbin}")
            write_image(f"{work}/{name}-raw.fits", img, s, a, b, expt1, (dx, dy))
            bins.append(name)
        if per_series > 1:
            log.info(f"{s.stem}: bin shifts [px] {' '.join(shifts)} ({len(xy)} guide stars)")
    log.info(f"{len(bins)} bins and {len(sums)} series sums calibrated in {time.time() - t:.1f}s")

    # 2. deep image and the common frame
    t = time.time()
    for f in ('deep.fits', 'skel.hdr'):
        if os.path.exists(f"{work}/{f}"):
            os.remove(f"{work}/{f}")
    run(['pyrt-combine', '-u', '--no-selection', '-o', 'deep.fits'] + sums, work, 'combine', logf)
    hd = fits.getheader(f"{work}/deep.fits")
    h0 = fits.getheader(f"{work}/{sums[0]}")
    with fits.open(f"{work}/deep.fits", mode='update') as d:
        # the deep image is the mean of the series sums: a 96-frame exposure with k times the
        # depth; keep the time/telescope keywords of the first series for cat2det/dophot
        copy_keys(h0, d[0].header)
        d[0].header['KINSTEP'] = (k, 'kinetic series combined')
        # pyrt-combine adds the (uniformly weighted) series sums: k x nsub readouts;
        # checked on the sky level, as the normalisation is combine's business
        scale = np.nanmedian(d[0].data[d[0].data != 0]) / np.nanmedian(fits.getdata(f"{work}/{sums[0]}"))
        ns = len(series)
        if not np.isfinite(scale) or abs(scale / ns - 1) > 0.1:
            log.warning(f"deep.fits is {scale:.2f} x a series sum, expected {ns}")
            if not np.isfinite(scale):
                scale = ns
        mark_accumulated(d[0].header, int(round(scale * args.nsub)))
        d[0].data = np.nan_to_num(d[0].data, nan=0.0).astype(np.float32)
    log.info(f"deep.fits {hd['NAXIS1']}x{hd['NAXIS2']} from {len(series)} series in {time.time() - t:.1f}s")
    # the common footprint: pyrt-combine grids the union of the series, but where a series
    # does not reach (the telescope recentred between series) its bins are empty and the
    # forced photometry fakes variability (EP 53267: a 'TURNING' source 53 px from the edge);
    # deep.fits is cropped to what every series covers, and the bins go onto that grid
    hd = crop_common(f"{work}/deep.fits", series, f"{work}/skel.hdr")

    # 3. deep photometry: the response of the first series' solution, zeropoint free
    fterms = args.filter_terms if args.filter_terms is not None else config_filter_terms(series[0].hdr)
    base_cmd = ['pyrt-dophot', '-m0.5', '-S0', f'-i{args.idlimit}', '--max-stars', '1000']
    phcat_args = ['-I'] if args.noiraf else ['-E', args.engine]
    terms = merge_terms(fixed_terms(series[0].response), fterms)
    t = time.time()
    tt = photometry('deep.fits', work, base_cmd + (['-U', terms] if terms else []), phcat_args, logf)
    deep = Table.read(f"{work}/deep.ecsv", format='ascii.ecsv')
    shutil.copy(f"{work}/deep.ecsv", f"{outdir}/{series[0].stem}-deep{k}.ecsv")
    log.info(f"deep: {len(deep)} objects, ZP {deep.meta.get('MAGZERO')}, limit {deep.meta.get('MAGLIMIT')}, "
             f"response {deep.meta.get('RESPONSE')} ({time.time() - t:.1f}s: phcat {tt[0]:.1f} "
             f"cat2det {tt[1]:.1f} dophot {tt[2]:.1f})")
    bin_terms = merge_terms(fixed_terms(deep.meta.get('RESPONSE', '')), fterms)
    bin_cmd = base_cmd + (['-U', bin_terms] if bin_terms else [])

    # 4. bins: reproject, forced photometry, zeropoint
    resamplers = {}
    if not args.montage:
        t = time.time()
        for s in series:
            resamplers[s.stem] = Resampler(s.wcs, f"{work}/skel.hdr")
        log.info(f"pixel maps of {len(series)} series onto skel.hdr in {time.time() - t:.1f}s")

    def do_bin(name):
        t = time.time()
        try:
            if args.montage:
                project(f"{name}-raw.fits", f"{name}.fits", 'skel.hdr', work, None)
            else:
                stem = fits.getval(f"{work}/{name}-raw.fits", 'KINSTEM')
                resample_fast(resamplers[stem], f"{name}-raw.fits", f"{name}.fits", 'skel.hdr', work)
            tp = time.time() - t
            tt = photometry(f"{name}.fits", work, bin_cmd, phcat_args + ['-D', 'deep.fits'], None)
            shutil.copy(f"{work}/{name}.ecsv", f"{outdir}/{name}.ecsv")
            m = Table.read(f"{work}/{name}.ecsv", format='ascii.ecsv').meta
            return name, None, (tp,) + tt, m
        except Exception as e:
            return name, str(e), None, None

    t = time.time()
    nok = 0
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for name, err, tt, m in ex.map(do_bin, bins):
            if err:
                log.warning(f"{name}: {err}")
                continue
            nok += 1
            log.info(f"{name}: ZP {float(m.get('MAGZERO', np.nan)):.3f} lim {float(m.get('MAGLIMIT', np.nan)):.2f} "
                     f"N {m.get('IDNUM')} (proj {tt[0]:.1f} phcat {tt[1]:.1f} cat2det {tt[2]:.1f} "
                     f"dophot {tt[3]:.1f} s)")
    log.info(f"{nok}/{len(bins)} bins in {time.time() - t:.1f}s; total {time.time() - t_all:.1f}s -> {outdir}")
    with open(f"{outdir}/step.json", 'w') as f:
        json.dump({'step': k, 'series': [s.stem for s in series], 'bins': len(bins), 'bins_ok': nok,
                   'bin_seconds': round(nbin * expt1, 2), 'deep_objects': len(deep),
                   'deep_limit': float(deep.meta.get('MAGLIMIT', np.nan)),
                   'seconds': round(time.time() - t_all, 1)}, f)
    # the deep image stays for the finding chart (trigreport); the runner keeps only the latest
    if os.path.exists(f"{work}/deep.fits"):
        shutil.copy(f"{work}/deep.fits", f"{outdir}/deep.fits")
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if nok == len(bins) else 1


if __name__ == '__main__':
    sys.exit(main())
