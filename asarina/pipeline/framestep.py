#!/usr/bin/env python3
"""
framestep - progressive light curve of a sequence of short frames, ~12 points per step.

The frame counterpart of kinstep (Andor kinetic series at D50): at SBT, the two
synchronised cameras C1 and C2 take alternating 5 s (or 12 s) frames of overlapping
fields, every frame is calibrated and solved by asarina-imgproc on its own.  When
12k frames of an observation are in, they are split into 12 bins of k consecutive
frames (k = 1: the 12 frames as they are; k = 2: a C1 + C2 pair; ...).

Per step:
  1. a common TAN grid (skel.hdr) over the union of the frames' footprints (the first
     frame's orientation and pixel scale),
  2. every frame resampled onto it with its own solution (the per-frame pixel map of
     kinstep, flux conserving) and scaled to the first frame's zeropoint: the cameras
     differ, and so does the transparency from frame to frame,
  3. bin = mean of its frames where they cover the pixel, times k (a k-frame exposure
     everywhere, the background stays continuous across the edge of one camera's field),
  4. deep = the same over all 12k frames, x 12k: phcat -> cat2det -> dophot (the first
     frame's spatial response fixed, zeropoint free) -> the deep catalogue,
  5. every bin: phcat -D deep.fits (forced photometry at the deep positions), cat2det,
     dophot with the deep response fixed, zeropoint free -> per-bin ECSVs for
     transients.py (flare search on ~12 points).

Inputs: calibrated frames (the df.fits imgproc copies to the kinseries spool) with their
ECSVs (solution, zeropoint), in pairs FITS:ECSV, or FITS alone with the ECSV next to it.

Usage:
  framestep.py --step 2 f1.fits:f1.ecsv f2.fits:f2.ecsv ...
"""

import argparse
import concurrent.futures as cf
import json
import logging
import os
import shutil
import sys
import threading
import time

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from asarina.pipeline.kinphot import ref_wcs_and_response, fixed_terms, WCS_RE, STALE_TIME
from asarina.pipeline.kinstep import (Resampler, copy_keys, mark_accumulated, photometry, merge_terms,
                                      config_filter_terms)

log = logging.getLogger("framestep")

POINTS = 12
MARGIN = 8   # [px] around the union of the footprints


class Frame:
    def __init__(self, spec):
        fpath, _, epath = spec.partition(':')
        if not epath:
            base = fpath[:-5] if fpath.endswith('.fits') else fpath
            epath = base + '.ecsv'
        self.path, self.ecsv = fpath, epath
        self.wcs, self.response, self.meta = ref_wcs_and_response(epath)
        self.hdr = fits.getheader(fpath)
        self.zp = float(self.meta.get('MAGZERO', np.nan))
        self.t0 = float(self.hdr['CTIME']) + float(self.hdr.get('USEC', 0)) / 1e6
        self.exptime = float(self.hdr['EXPTIME'])
        self.shape = (self.hdr['NAXIS2'], self.hdr['NAXIS1'])
        self.camera = self.hdr.get('CCD_NAME', '?')
        self.stem = os.path.basename(fpath)[:18]

    def data(self):
        return fits.getdata(self.path).astype(np.float32)


def make_skel(frames, path):
    """TAN grid: the first frame's linear orientation and scale, centred on and covering the
    union of all footprints."""
    w0 = WCS(fits.Header(frames[0].wcs))
    cd = w0.pixel_scale_matrix
    # footprints in the first frame's pixels -> bounding box
    xs, ys = [], []
    for f in frames:
        w = WCS(fits.Header(f.wcs))
        ny, nx = f.shape
        ex = np.r_[np.linspace(0, nx - 1, 9), np.full(9, nx - 1), np.linspace(nx - 1, 0, 9), np.zeros(9)]
        ey = np.r_[np.zeros(9), np.linspace(0, ny - 1, 9), np.full(9, ny - 1), np.linspace(ny - 1, 0, 9)]
        ra, dec = w.all_pix2world(ex, ey, 0)
        x, y = w0.all_world2pix(ra, dec, 0)
        xs.append(x)
        ys.append(y)
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    x0, x1, y0, y1 = xs.min() - MARGIN, xs.max() + MARGIN, ys.min() - MARGIN, ys.max() + MARGIN
    nx, ny = int(np.ceil(x1 - x0)), int(np.ceil(y1 - y0))
    ra_c, dec_c = w0.all_pix2world([(x0 + x1) / 2], [(y0 + y1) / 2], 0)
    h = fits.Header()
    h['NAXIS'] = 2
    h['NAXIS1'], h['NAXIS2'] = nx, ny
    h['CTYPE1'], h['CTYPE2'] = 'RA---TAN', 'DEC--TAN'
    h['CRVAL1'], h['CRVAL2'] = float(ra_c[0]), float(dec_c[0])
    h['CRPIX1'], h['CRPIX2'] = (nx + 1) / 2, (ny + 1) / 2
    h['CD1_1'], h['CD1_2'], h['CD2_1'], h['CD2_2'] = cd[0, 0], cd[0, 1], cd[1, 0], cd[1, 1]
    h['EQUINOX'] = 2000.0
    h.totextfile(path, overwrite=True)
    return h


def bin_header(skel, frames, k, first_hdr):
    hdr = skel.copy()
    for key in ('NAXIS', 'NAXIS1', 'NAXIS2'):
        hdr.remove(key, ignore_missing=True)
    src = first_hdr.copy()
    for key in list(src.keys()):
        if WCS_RE.match(key) or key in STALE_TIME or key.startswith('NAXIS'):
            del src[key]
    copy_keys(src, hdr)
    t0 = frames[0].t0
    t1 = frames[-1].t0 + frames[-1].exptime
    ctime = int(np.floor(t0))
    hdr['CTIME'] = (ctime, 'bin start (unix)')
    hdr['USEC'] = (int(round((t0 - ctime) * 1e6)), 'bin start micro seconds')
    hdr['JD'] = (2440587.5 + t0 / 86400.0, 'bin start')
    hdr['DATE-OBS'] = (time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(t0)) + f".{int((t0 % 1) * 1000):03d}", 'bin start')
    # the light curve point is at the middle of the exposure: EXPTIME = the time span
    # (a k-frame sum has k exposures' worth of signal, the zeropoint takes that)
    hdr['EXPTIME'] = (round(t1 - t0, 3), '[s] span of the binned frames')
    hdr['FRMEXPT'] = (round(sum(f.exptime for f in frames), 3), '[s] summed exposure of the frames')
    hdr['NFRAMES'] = (len(frames), 'frames in this bin')
    hdr['FRAMES'] = (' '.join(f.stem[-6:] + f.camera for f in frames)[:68], 'stems (tail) and cameras')
    mark_accumulated(hdr, k)
    return hdr


def main():
    os.environ.setdefault('TERM', 'xterm')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('frames', nargs='+', help='calibrated frames in time order, FITS[:ECSV]')
    ap.add_argument('--step', type=int, help='frames per bin (default: all frames // points)')
    ap.add_argument('--points', type=int, default=POINTS, help=f'light-curve points (default {POINTS})')
    ap.add_argument('--filter-terms', default=None,
                    help="fixed colour terms for dophot (default filter_terms_<filter> of the asarina config)")
    ap.add_argument('-i', '--idlimit', type=int, default=2, help='dophot idlimit (default 2)')
    ap.add_argument('-o', '--outdir', help='output directory (default ./<first stem>-fstep<k>)')
    ap.add_argument('-j', '--jobs', type=int, default=4, help='parallel bins (default 4)')
    ap.add_argument('-I', '--noiraf', action='store_true', help='phcat -I (SExtractor magnitudes, faster)')
    ap.add_argument('-k', '--keep', action='store_true', help='keep the work files')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    for name in ('asarina', 'pyrt'):
        logging.getLogger(name).setLevel(logging.WARNING)
    t_all = time.time()

    k = args.step or len(args.frames) // args.points
    n = k * args.points
    if k < 1 or len(args.frames) < n:
        ap.error(f"step {k} needs {n} frames, {len(args.frames)} given")
    frames = sorted((Frame(s) for s in args.frames[:n]), key=lambda f: f.t0)
    bad = [f.stem for f in frames if not np.isfinite(f.zp)]
    if bad:
        raise SystemExit(f"no MAGZERO in the ECSV of {bad}")
    cams = sorted(set(f.camera for f in frames))
    log.info(f"step {k}: {n} frames ({'+'.join(cams)}) -> {args.points} bins of {k}")

    outdir = os.path.abspath(args.outdir or f"{frames[0].stem}-fstep{k}")
    work = f"{outdir}/work"
    os.makedirs(work, exist_ok=True)
    logf = f"{work}/framestep.log"

    # 1.-3. common grid, resampled and scaled frames, bins
    t = time.time()
    skel = make_skel(frames, f"{work}/skel.hdr")
    shape = (skel['NAXIS2'], skel['NAXIS1'])
    zref = frames[0].zp

    # every frame resampled once and added to its bin and to the deep image, then dropped:
    # memory 2 x (12 + 1) grids, not one per frame (12 s SBT frames are 4k x 4k)
    acc = np.zeros((args.points + 1,) + shape, np.float32)
    cnt = np.zeros((args.points + 1,) + shape, np.uint16)
    lock = threading.Lock()

    def add(i_f):
        i, f = i_f
        rs = Resampler(f.wcs, f"{work}/skel.hdr")
        img = rs(f.data()) * 10 ** (0.4 * (zref - f.zp))
        ny, nx = f.shape
        cov = (rs.x >= 0) & (rs.x <= nx - 1) & (rs.y >= 0) & (rs.y <= ny - 1)
        img[~cov] = 0
        b = i // k
        with lock:
            acc[b] += img
            cnt[b] += cov
            acc[-1] += img
            cnt[-1] += cov

    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        list(ex.map(add, enumerate(frames)))
    log.info(f"{n} frames resampled onto {shape[1]}x{shape[0]} in {time.time() - t:.1f}s "
             f"(zeropoints {min(f.zp for f in frames):.2f}..{max(f.zp for f in frames):.2f})")

    def mean_of(j, scale):
        c = cnt[j]
        return np.where(c > 0, acc[j] / np.maximum(c, 1) * scale, 0).astype(np.float32), c

    bins = []
    for b in range(args.points):
        fr = frames[b * k:(b + 1) * k]
        img, _ = mean_of(b, k)
        ts = fr[0].t0
        name = (time.strftime('%Y%m%d%H%M%S', time.gmtime(ts)) + f"-{int((ts % 1) * 1000):03d}"
                + f"-{fr[0].hdr.get('FILTER', 'UNK')}-f{b * k:03d}b{k}")
        fits.PrimaryHDU(data=img, header=bin_header(skel, fr, k, fr[0].hdr)).writeto(f"{work}/{name}.fits",
                                                                                   overwrite=True)
        bins.append(name)

    deep, cnt = mean_of(-1, n)
    hdr = bin_header(skel, frames, n, frames[0].hdr)
    hdr['KINSTEP'] = (k, 'frames per bin of this step')
    fits.PrimaryHDU(data=deep, header=hdr).writeto(f"{work}/deep.fits", overwrite=True)
    del acc
    log.info(f"{len(bins)} bins and the deep image in {time.time() - t:.1f}s "
             f"(coverage {int(cnt.max())} frames, {np.mean(cnt >= n / 2) * 100:.0f}% of the grid by half or more)")

    # 4. deep photometry
    fterms = args.filter_terms if args.filter_terms is not None else config_filter_terms(frames[0].hdr)
    base_cmd = ['pyrt-dophot', '-m0.5', '-S0', f'-i{args.idlimit}', '--max-stars', '1000']
    phcat_args = ['-I'] if args.noiraf else []
    terms = merge_terms(fixed_terms(frames[0].response), fterms)
    t = time.time()
    tt = photometry('deep.fits', work, base_cmd + (['-U', terms] if terms else []), phcat_args, logf)
    dcat = Table.read(f"{work}/deep.ecsv", format='ascii.ecsv')
    shutil.copy(f"{work}/deep.ecsv", f"{outdir}/{frames[0].stem}-deep{k}.ecsv")
    log.info(f"deep: {len(dcat)} objects, ZP {dcat.meta.get('MAGZERO')}, limit {dcat.meta.get('MAGLIMIT')} "
             f"({time.time() - t:.1f}s: phcat {tt[0]:.1f} cat2det {tt[1]:.1f} dophot {tt[2]:.1f})")
    bin_terms = merge_terms(fixed_terms(dcat.meta.get('RESPONSE', '')), fterms)
    bin_cmd = base_cmd + (['-U', bin_terms] if bin_terms else [])

    # 5. bins
    def do_bin(name):
        try:
            tt = photometry(f"{name}.fits", work, bin_cmd, phcat_args + ['-D', 'deep.fits'], None)
            shutil.copy(f"{work}/{name}.ecsv", f"{outdir}/{name}.ecsv")
            return name, None, tt, Table.read(f"{work}/{name}.ecsv", format='ascii.ecsv').meta
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
                     f"N {m.get('IDNUM')} (phcat {tt[0]:.1f} cat2det {tt[1]:.1f} dophot {tt[2]:.1f} s)")
    log.info(f"{nok}/{len(bins)} bins in {time.time() - t:.1f}s; total {time.time() - t_all:.1f}s -> {outdir}")
    with open(f"{outdir}/step.json", 'w') as f:
        json.dump({'step': k, 'frames': [fr.stem for fr in frames], 'bins': len(bins), 'bins_ok': nok,
                   'bin_seconds': round(frames[k - 1].t0 + frames[k - 1].exptime - frames[0].t0, 2),
                   'deep_objects': len(dcat), 'deep_limit': float(dcat.meta.get('MAGLIMIT', np.nan)),
                   'seconds': round(time.time() - t_all, 1)}, f)
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if nok == len(bins) else 1


if __name__ == '__main__':
    sys.exit(main())
