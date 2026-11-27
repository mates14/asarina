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
  3. bin = exposure-weighted mean of its frames where they cover the pixel, times its summed
     exposure (in the first frame's), with outliers rejected over the whole series
     (stack_strip: cosmic rays, satellites, bad columns; a source lasting two or more
     frames is kept; --stack mean for the plain mean),
  4. deep = the same over all 12k frames, x 12k: phcat -> cat2det -> dophot (the first
     frame's spatial response fixed, zeropoint free) -> the deep catalogue,
  5. every bin: phcat -D deep.fits (forced photometry at the deep positions), cat2det,
     dophot with the deep response fixed, zeropoint free -> per-bin ECSVs for
     transients.py (flare search on ~12 points).

Sky frame (--sky-frame; auto: when the frames carry RTSPIX, i.e. the camera has two-faced
hot pixels, FRAM-ORM): before 2., per camera, the median over the series of every pixel's
sky-subtracted rate (stars masked by their 3x3 neighbourhood, so a lone hot pixel stays),
subtracted from every frame times its exposure where it is significant (> SKY_NSIG of its
scatter over the frame).  An RTS pixel keeps its state for hours, so this takes out what a
frame alone cannot decide (short exposures, bright sky) and the traces it leaves in the
stacks.  Only for a dithered series (the stars move by > SKY_DITHER px), else the stars would
go too; nothing is subtracted elsewhere, so no noise is added.

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
import time

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from pyrt.core.stack import _sky, _sky_at, MESH
from pyrt.utils.camera_config import readouts
from pyrt.utils import badpix

from asarina.pipeline.kinphot import ref_wcs_and_response, fixed_terms, WCS_RE, STALE_TIME
from asarina.pipeline.kinstep import (Resampler, copy_keys, mark_accumulated, photometry, merge_terms,
                                      config_filter_terms)

log = logging.getLogger("framestep")

POINTS = 12
MARGIN = 8   # [px] inside the common footprint
RESAMPLE_MEM = 2.5e9   # [B] for the pixel maps being built at the same time
STACK_MEM = 1.5e9      # [B] for a strip of the frame cube and its work arrays
# outlier rejection (as pyrt.core.stack, which pyrt-combine uses): against the per-pixel
# median of the sky-subtracted frames, noise from each frame's own rms and its signal
CLIP_HI, CLIP_LO = 4.0, 6.0
CLIP_AMP = 1.0         # tolerance as a fraction of the median: the cores of bright stars
                       # differ from frame to frame with the seeing
CLIP_NEAR = 2.0        # [sigma] a high pixel with a neighbour in time this high is a source
                       # lasting more than one frame (a fading afterglow, a flare), not a hit
CLIP_MIN = 5           # frames on a pixel to test it at all
SKY_MIN_FRAMES = 8     # sky frame: frames of one camera it needs
SKY_NSIG = 5.0         # sky frame: subtracted where the median rate exceeds this x its scatter
SKY_DITHER = 2.0       # [px] sky frame: median offset of the frames from the series' median pointing
SKY_MEM = 3e9          # [B] sky frame: the frame cube of one camera at most
GROW = 1               # [px] a rejected pixel's neighbours go with it (the resampling spreads a hit)


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
        self.skyframe = None     # [ADU/s] see sky_frames()

    def data(self):
        """The frame, the sky frame off, its BADPIX pixels (pyrt.utils.badpix) NaN: left out
        of the stacks (the bilinear resampling blanks at most their 2x2 output neighbours)."""
        d = fits.getdata(self.path).astype(np.float32)
        if self.skyframe is not None:
            d -= self.skyframe * np.float32(self.exptime)
        by, bx = badpix.from_header(self.hdr)
        if len(by):
            d[by, bx] = np.nan
        return d


def sky_frames(frames, mode='auto', nsig=SKY_NSIG, dither_min=SKY_DITHER):
    """Set Frame.skyframe per camera: the significant part of the median over the series of
    each pixel's sky-subtracted rate (see the module docstring).  Returns log lines."""
    from scipy.ndimage import median_filter, binary_dilation, label
    msgs = []
    if mode == 'off':
        return msgs
    for cam in sorted(set(f.camera for f in frames)):
        fr = [f for f in frames if f.camera == cam]
        if mode == 'auto' and not all('RTSPIX' in f.hdr for f in fr):
            continue
        shapes = {f.shape for f in fr}
        if len(fr) < (SKY_MIN_FRAMES if mode == 'auto' else 3) or len(shapes) > 1:
            msgs.append(f"sky frame {cam}: {len(fr)} frames, {len(shapes)} shapes - none")
            continue
        ny, nx = fr[0].shape
        if len(fr) * ny * nx * 4 > SKY_MEM:
            msgs.append(f"sky frame {cam}: {len(fr)} frames of {nx}x{ny} too big - none")
            continue
        # dithering: the frame centre of each frame in the first frame's pixels
        w0 = WCS(fits.Header(fr[0].wcs))
        xy = np.array([w0.all_world2pix(WCS(fits.Header(f.wcs)).all_pix2world([[nx / 2, ny / 2]], 0), 0)[0]
                       for f in fr])
        dith = float(np.median(np.hypot(*(xy - np.median(xy, axis=0)).T)))
        if dith < dither_min:
            msgs.append(f"sky frame {cam}: not dithered ({dith:.1f} px) - none")
            continue
        cube = np.empty((len(fr), ny, nx), np.float32)
        for i, f in enumerate(fr):
            d = f.data() if f.skyframe is None else fits.getdata(f.path).astype(np.float32)
            med, rms = _sky(d)
            x = d - _sky_at(med, MESH, 0, ny, 0, nx)
            star = binary_dilation(median_filter(np.nan_to_num(x), size=3) > 3 * rms, iterations=2)
            cube[i] = np.where(star | ~np.isfinite(x), np.nan, x / np.float32(f.exptime))
        ok = np.isfinite(cube)
        cnt = ok.sum(axis=0)
        R = _nanmedian0(cube, ok)
        del cube
        good = np.isfinite(R) & (cnt >= max(3, len(fr) // 3))
        r = R[good]
        sr = 1.4826 * float(np.median(np.abs(r - np.median(r)))) if r.size else np.inf
        use = good & (np.abs(R) > nsig * sr)
        # only isolated pixels (1-2 px): a hot/RTS pixel; a larger group is a star remnant, a
        # column, the overscan - left to the stack's rejection
        lab, nl = label(use)
        if nl:
            size = np.bincount(lab.ravel())
            use &= (size <= 2)[lab]
        S = np.where(use, R, 0).astype(np.float32)
        for f in fr:
            f.skyframe = S
        msgs.append(f"sky frame {cam}: {len(fr)} frames, dither {dith:.0f} px, scatter {sr:.4f} ADU/s, "
                    f"{int(use.sum())} px subtracted ({int((S > 0).sum())} high, {int((S < 0).sum())} low)")
    return msgs


def make_skel(frames, path):
    """TAN grid: the first frame's linear orientation and scale, over the footprint common to
    all frames (their intersection, MARGIN inside): every pixel is covered by every frame.
    Over the union, the edge rows were covered by only some frames of a bin when the
    pointing wandered, and faked variability there (GRB 230818A: 'flares' along an edge)."""
    w0 = WCS(fits.Header(frames[0].wcs))
    cd = w0.pixel_scale_matrix
    # footprints in the first frame's pixels -> bounding box
    bx0, bx1, by0, by1 = -np.inf, np.inf, -np.inf, np.inf
    for f in frames:
        w = WCS(fits.Header(f.wcs))
        ny, nx = f.shape
        ex = np.r_[np.linspace(0, nx - 1, 9), np.full(9, nx - 1), np.linspace(nx - 1, 0, 9), np.zeros(9)]
        ey = np.r_[np.zeros(9), np.linspace(0, ny - 1, 9), np.full(9, ny - 1), np.linspace(ny - 1, 0, 9)]
        ra, dec = w.all_pix2world(ex, ey, 0)
        x, y = w0.all_world2pix(ra, dec, 0)
        # the inner box of this footprint (slightly rotated frames: the inner two values)
        sx, sy = np.sort(x[[0, 9, 18, 27]]), np.sort(y[[0, 9, 18, 27]])
        bx0, bx1 = max(bx0, sx[1]), min(bx1, sx[2])
        by0, by1 = max(by0, sy[1]), min(by1, sy[2])
    x0, x1, y0, y1 = bx0 + MARGIN, bx1 - MARGIN, by0 + MARGIN, by1 - MARGIN
    if x1 - x0 < 100 or y1 - y0 < 100:
        raise SystemExit(f"the frames have (almost) no common footprint: {x1 - x0:.0f} x {y1 - y0:.0f} px")
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


def exposure_bins(exptimes, points, tol=0.01):
    """Split frames (their exposure times, in time order) into `points` consecutive bins of
    equal summed exposure, on frame boundaries: [[i, ...], ...], or None when they do not
    split exactly (20 s and 60 s frames: 12 x 20 s, or 12 x 20 + 8 x 60 s into 60 s bins).
    Equal exposures: bins of len // points frames, as --step."""
    e = np.asarray(exptimes, float)
    if len(e) < points or np.any(e <= 0):
        return None
    target = e.sum() / points
    bins, cur, acc = [], [], 0.0
    for i, x in enumerate(e):
        cur.append(i)
        acc += x
        if acc >= target * (1 - tol):
            if abs(acc - target) > tol * target:
                return None
            bins.append(cur)
            cur, acc = [], 0.0
    return bins if len(bins) == points and not cur else None


def _nanmedian0(x, ok):
    """np.nanmedian(x, axis=0) for a frame cube: a sort puts the NaNs last, the median is
    taken from the finite ones (np.nanmedian goes through masked arrays here, ~5x slower)."""
    xs = np.sort(x, axis=0)
    c = ok.sum(axis=0)
    lo = np.clip((c - 1) // 2, 0, None)[None]
    hi = np.clip(c // 2, 0, x.shape[0] - 1)[None]
    m = 0.5 * (np.take_along_axis(xs, lo, 0)[0] + np.take_along_axis(xs, hi, 0)[0])
    return np.where(c > 0, m, np.nan).astype(x.dtype)


def _grow(b, it):
    """binary_dilation of every frame plane (the cross element, it iterations) by shifts."""
    for _ in range(it):
        g = b.copy()
        g[:, 1:] |= b[:, :-1]
        g[:, :-1] |= b[:, 1:]
        g[:, :, 1:] |= b[:, :, :-1]
        g[:, :, :-1] |= b[:, :, 1:]
        b = g
    return b


def stack_strip(X, Sk, rms, kg, wts, groups, clip=True):
    """Rows of the frame cube X (frames x rows x cols, flux on the first frame's zeropoint, NaN
    off a frame) and their skies Sk -> the weighted means of the frame groups (the bins and
    the deep image, wts the frames' exposure weights), the coverage of all the frames
    (weights summed), the rejected frames per pixel.

    Rejection (clip): x = X - sky against the per-pixel median over all the frames,
    s^2 = rms^2 + kg max(x, 0); rejected above median + CLIP_HI s + CLIP_AMP |median| (cosmic
    rays, satellites, planes) unless a frame next to it in time is CLIP_NEAR s high as well
    (a real source changing over the series stays: the afterglow in the early frames), or below
    median - CLIP_LO s - CLIP_AMP |median| (bad columns); grown by GROW px; second pass about
    the weighted mean of what passed.  A group whose frames are all rejected on a pixel keeps
    them there (a one-frame bin is never clipped).  The mean: of x over what passed, plus the
    weighted mean sky of all the frames covering the pixel (a rejection does not step the
    level by that frame's sky offset)."""
    import warnings
    n = X.shape[0]
    ok = np.isfinite(X)
    x = np.where(ok, X - Sk, np.nan)
    w = np.asarray(wts, np.float32)[:, None, None]
    keep = ok.copy()
    if clip:
        S = np.sqrt(np.asarray(rms, np.float32)[:, None, None] ** 2
                    + np.asarray(kg, np.float32)[:, None, None] * np.clip(np.nan_to_num(x), 0, None))
        test = ok.sum(axis=0) >= CLIP_MIN
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            centre = _nanmedian0(x, ok)
            for it in range(2):
                tol = CLIP_AMP * np.abs(centre)
                hi = ok & ((x > centre + CLIP_HI * S + tol))
                near = ok & (x > centre + CLIP_NEAR * S + tol)
                nb = np.zeros_like(near)
                nb[1:] |= near[:-1]
                nb[:-1] |= near[1:]
                bad = test & (hi & ~nb | ok & (x < centre - CLIP_LO * S - tol))
                if GROW:
                    bad = _grow(bad, GROW) & ok
                keep = ok & ~bad
                if it == 0:
                    wk = np.where(keep, w, 0)
                    ws = wk.sum(axis=0)
                    centre = np.where(ws > 0, np.nansum(np.where(keep, x, 0) * wk, axis=0) / np.maximum(ws, 1e-30),
                                      centre)
    out = np.zeros((len(groups),) + X.shape[1:], np.float32)
    for j, g in enumerate(groups):
        okg, kg_ = ok[g], keep[g]
        kg_ = np.where(((okg.sum(axis=0) > 0) & (kg_.sum(axis=0) == 0))[None], okg, kg_)
        wg = w[g]
        wk = np.where(kg_, wg, 0)
        wa = np.where(okg, wg, 0)
        ws, wsa = wk.sum(axis=0), wa.sum(axis=0)
        num = np.nansum(np.where(kg_, x[g], 0) * wk, axis=0)
        sky = np.nansum(np.where(okg, Sk[g], 0) * wa, axis=0)
        out[j] = np.where(ws > 0, num / np.maximum(ws, 1e-30) + sky / np.maximum(wsa, 1e-30), 0)
    cov = np.where(ok, w, 0).sum(axis=0)
    return out, cov, (ok & ~keep).sum(axis=0).astype(np.int16)


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
    # the frames' own sums count: a D50 kinetic series frame is 96 scans (COADDS) since
    # 2026-11, and with ACCNUM=k alone every star of a bin of them was "saturated"
    mark_accumulated(hdr, sum(readouts(f.hdr) for f in frames))
    return hdr


def main():
    os.environ.setdefault('TERM', 'xterm')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('frames', nargs='+', help='calibrated frames in time order, FITS[:ECSV]')
    ap.add_argument('--step', type=int, help='frames per bin, equal exposures (default: all the frames, '
                                             'in bins of equal summed exposure, see exposure_bins)')
    ap.add_argument('--label', type=int, help='the step number for step.json (default: frames in the first bin)')
    ap.add_argument('--points', type=int, default=POINTS, help=f'light-curve points (default {POINTS})')
    ap.add_argument('--filter-terms', default=None,
                    help="fixed colour terms for dophot (default filter_terms_<filter> of the asarina config)")
    ap.add_argument('-i', '--idlimit', type=int, default=2, help='dophot idlimit (default 2)')
    ap.add_argument('-o', '--outdir', help='output directory (default ./<first stem>-fstep<k>)')
    ap.add_argument('-j', '--jobs', type=int, default=4, help='parallel bins (default 4)')
    ap.add_argument('-I', '--noiraf', action='store_true', help='phcat -I (SExtractor magnitudes, faster)')
    ap.add_argument('-E', '--engine', choices=('numpy', 'iraf', 'auto'), default='numpy',
                    help='phcat aperture photometry: numpy (IRAF phot reimplemented, fast; default) or IRAF')
    ap.add_argument('--stack', choices=('clip', 'mean'), default='clip',
                    help='clip: weighted mean with outlier rejection (default); mean: plain weighted mean')
    ap.add_argument('--sky-frame', choices=('auto', 'on', 'off'), default='auto',
                    help='subtract the series sky frame where significant (auto: frames with RTSPIX)')
    ap.add_argument('--sky-dither', type=float, default=SKY_DITHER,
                    help=f'[px] least median dither for the sky frame (default {SKY_DITHER:g})')
    ap.add_argument('-k', '--keep', action='store_true', help='keep the work files')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    for name in ('asarina', 'pyrt'):
        logging.getLogger(name).setLevel(logging.WARNING)
    t_all = time.time()

    if args.step:
        k = args.step
        n = k * args.points
        if k < 1 or len(args.frames) < n:
            ap.error(f"step {k} needs {n} frames, {len(args.frames)} given")
        frames = sorted((Frame(s) for s in args.frames[:n]), key=lambda f: f.t0)
        members = [list(range(b * k, (b + 1) * k)) for b in range(args.points)]
    else:
        frames = sorted((Frame(s) for s in args.frames), key=lambda f: f.t0)
        members = exposure_bins([f.exptime for f in frames], args.points)
        if members is None:
            ap.error(f"{len(frames)} frames ({sorted(set(f.exptime for f in frames))} s) do not split into "
                     f"{args.points} bins of equal exposure")
        n, k = len(frames), len(members[0])
    bad = [f.stem for f in frames if not np.isfinite(f.zp)]
    if bad:
        raise SystemExit(f"no MAGZERO in the ECSV of {bad}")
    # exposure weights in units of the shortest frame (integers: the coverage stays uint16)
    emin = min(f.exptime for f in frames)
    wts = [f.exptime / emin for f in frames]
    if any(abs(w - round(w)) > 0.02 for w in wts):
        raise SystemExit(f"exposures {sorted(set(f.exptime for f in frames))} s are not multiples of {emin} s")
    wts = [int(round(w)) for w in wts]
    bin_of = {i: b for b, m in enumerate(members) for i in m}
    exp0 = frames[0].exptime
    cams = sorted(set(f.camera for f in frames))
    sizes = sorted(set(len(m) for m in members))
    log.info(f"step {args.label if args.label is not None else k}: {n} frames ({'+'.join(cams)}, "
             f"{'+'.join(f'{e:g}' for e in sorted(set(f.exptime for f in frames)))} s) -> {args.points} bins of "
             f"{sum(frames[i].exptime for i in members[0]):g} s ({'/'.join(map(str, sizes))} frames)")

    outdir = os.path.abspath(args.outdir or f"{frames[0].stem}-fstep{k}")
    work = f"{outdir}/work"
    os.makedirs(work, exist_ok=True)
    logf = f"{work}/framestep.log"

    # 1.-3. common grid, resampled and scaled frames, bins
    t = time.time()
    for m in sky_frames(frames, args.sky_frame, dither_min=args.sky_dither):
        log.info(m)
    skel = make_skel(frames, f"{work}/skel.hdr")
    shape = (skel['NAXIS2'], skel['NAXIS1'])
    zref = frames[0].zp

    # every frame resampled once (flux on the first frame's zeropoint) into a disk-backed
    # cube; then, strip by strip, the outlier rejection over all the frames and the
    # exposure-weighted means of the bins and the deep image (stack_strip)
    n_all = len(frames)
    cube = np.lib.format.open_memmap(f"{work}/cube.npy", mode='w+', dtype=np.float32, shape=(n_all,) + shape)
    skies = [None] * n_all

    def add(i_f):
        i, f = i_f
        rs = Resampler(f.wcs, f"{work}/skel.hdr")
        k_i = 10 ** (0.4 * (zref - f.zp))
        img = rs(f.data()) * k_i
        ny, nx = f.shape
        cov = (rs.x >= 0) & (rs.x <= nx - 1) & (rs.y >= 0) & (rs.y <= ny - 1)
        img[~cov] = np.nan
        med, rms = _sky(img)
        gain = float(f.hdr.get('GAIN') or 1.0)
        skies[i] = (med, rms, k_i / gain)   # var of x: rms^2 + (k/gain) (x - sky)
        cube[i] = img

    # a pixel map takes ~60 B/px while it is built: 1k frames in parallel, 4k ones (~1 GB
    # each) two at a time
    npix = max(f.shape[0] * f.shape[1] for f in frames)
    workers = max(1, min(args.jobs, int(RESAMPLE_MEM // (60 * npix))))
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(add, enumerate(frames)))
    cube.flush()
    log.info(f"{n} frames resampled onto {shape[1]}x{shape[0]} in {time.time() - t:.1f}s "
             f"(zeropoints {min(f.zp for f in frames):.2f}..{max(f.zp for f in frames):.2f})")

    groups = [list(m) for m in members] + [list(range(n_all))]
    out = np.zeros((len(groups),) + shape, np.float32)
    cnt = np.zeros(shape, np.float32)
    nrej = np.zeros(shape, np.int16)
    rows = max(8, int(STACK_MEM // (n_all * shape[1] * 4 * 6)))
    clip = args.stack == 'clip'
    for y0 in range(0, shape[0], rows):
        y1 = min(shape[0], y0 + rows)
        ya, yb = max(0, y0 - GROW), min(shape[0], y1 + GROW)
        X = np.asarray(cube[:, ya:yb, :])
        Sk = np.stack([_sky_at(skies[i][0], MESH, ya, yb, 0, shape[1]) for i in range(n_all)])
        r = slice(y0 - ya, y0 - ya + (y1 - y0))
        o, c, nr = stack_strip(X, Sk, [s[1] for s in skies], [s[2] for s in skies], wts, groups, clip)
        out[:, y0:y1] = o[:, r]
        cnt[y0:y1] = c[r]
        nrej[y0:y1] = nr[r]
    del cube
    os.unlink(f"{work}/cube.npy")
    log.info(f"stack ({args.stack}): {int(nrej.sum())} frame pixels rejected "
             f"({nrej.sum() / max(1, n_all * nrej.size) * 1e6:.0f} ppm), {np.mean(nrej > 0) * 100:.2f}% of the grid touched")

    # the weighted mean times the summed exposure in units of the first frame's: a bin has
    # the signal of its whole exposure (equal exposures: the mean times k, as before)
    def stack_keys(h):
        h['STACK'] = (args.stack, 'framestep: clip = weighted mean with outlier rejection')
        if clip:
            h['CLIPHI'] = (CLIP_HI, 'rejection above the median [sigma]')
            h['CLIPLO'] = (CLIP_LO, 'rejection below the median [sigma]')
            h['CLIPAMP'] = (CLIP_AMP, 'tolerance as a fraction of the median')
        return h

    bins = []
    for b, m in enumerate(members):
        fr = [frames[i] for i in m]
        img = out[b] * (sum(f.exptime for f in fr) / exp0)
        ts = fr[0].t0
        name = (time.strftime('%Y%m%d%H%M%S', time.gmtime(ts)) + f"-{int((ts % 1) * 1000):03d}"
                + f"-{fr[0].hdr.get('FILTER', 'UNK')}-f{m[0]:03d}b{len(m)}")
        fits.PrimaryHDU(data=img, header=stack_keys(bin_header(skel, fr, len(fr), fr[0].hdr))).writeto(
            f"{work}/{name}.fits", overwrite=True)
        bins.append(name)

    deep = out[-1] * (sum(f.exptime for f in frames) / exp0)
    cnt = cnt / max(1, min(wts))   # coverage in frames of the shortest exposure
    hdr = stack_keys(bin_header(skel, frames, n, frames[0].hdr))
    hdr['KINSTEP'] = (k, 'frames per bin of this step')
    if clip:
        hdr['NCLIPPED'] = (int(nrej.sum()), 'frame pixels rejected')
    fits.PrimaryHDU(data=deep, header=hdr).writeto(f"{work}/deep.fits", overwrite=True)
    if args.keep:
        fits.PrimaryHDU(data=nrej, header=hdr).writeto(f"{work}/rejected.fits", overwrite=True)
    del out
    log.info(f"{len(bins)} bins and the deep image in {time.time() - t:.1f}s "
             f"(coverage {int(cnt.max())} frame-weights, {np.mean(cnt >= cnt.max() / 2) * 100:.0f}% of the grid by half or more)")

    # 4. deep photometry
    fterms = args.filter_terms if args.filter_terms is not None else config_filter_terms(frames[0].hdr)
    base_cmd = ['pyrt-dophot', '-m0.5', '-S0', f'-i{args.idlimit}', '--max-stars', '1000']
    phcat_args = ['-I'] if args.noiraf else ['-E', args.engine]
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
        last = frames[members[0][-1]]
        json.dump({'step': args.label if args.label is not None else k, 'frames': [fr.stem for fr in frames],
                   'bins': len(bins), 'bins_ok': nok,
                   'bin_seconds': round(last.t0 + last.exptime - frames[0].t0, 2),
                   'bin_exposure': round(sum(frames[i].exptime for i in members[0]), 2),
                   'exptimes': sorted(set(f.exptime for f in frames)),
                   'deep_objects': len(dcat), 'deep_limit': float(dcat.meta.get('MAGLIMIT', np.nan)),
                   'seconds': round(time.time() - t_all, 1)}, f)
    # the deep image stays for the finding chart (trigreport); the runner keeps only the latest
    if os.path.exists(f"{work}/deep.fits"):
        shutil.copy(f"{work}/deep.fits", f"{outdir}/deep.fits")
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if nok == len(bins) else 1


if __name__ == '__main__':
    sys.exit(main())
