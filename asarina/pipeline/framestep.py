#!/usr/bin/env python3
"""
framestep - progressive light curve of a sequence of short frames, ~12 points per step.

The frame counterpart of kinstep (Andor kinetic series at D50): at SBT, the two
synchronised cameras C1 and C2 take alternating 5 s (or 12 s) frames of overlapping
fields, every frame is calibrated and solved by asarina-imgproc on its own.  When
12k frames of an observation are in, they are split into 12 bins of k consecutive
frames (k = 1: the 12 frames as they are; k = 2: a C1 + C2 pair; ...).

Two ways (--mode):
  combine (default): one TAN grid per observation (--skel; around the trigger: max(BOX_MIN,
     BOX_ERR error radii), within the union of the footprints), every frame projected onto it
     once (pyrt-combine --skel, cached in <skel dir>/proj), a bin = the clipped, zeropoint-
     scaled, exposure-weighted mean of its frames where they cover (blank below MIN_COVER),
     the deep image = the bins stacked pixel by pixel; then 4. and 5. below.  Memory: a bin's
     frames over a strip of rows.
  cube: per step, as follows (the whole step's frames in one cube).

Per step (cube):
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
BOX_MIN = 15.0         # [arcmin] least half-size of a grid cut to the error box (calibration stars)
BOX_ERR = 3.0          # the grid's half-size in error-box radii (GRB_ERR): the search goes beyond the box
MIN_COVER = 0.5        # a bin is blank where less than this fraction of its frames covers the pixel
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


def make_skel(frames, path, box=None, union=False):
    """TAN grid: the first frame's linear orientation and scale, over the footprint common to
    all frames (their intersection, MARGIN inside): every pixel is covered by every frame.
    Over the union, the edge rows were covered by only some frames of a bin when the
    pointing wandered, and faked variability there (GRB 230818A: 'flares' along an edge).
    union=True (--mode combine, which normalises every bin by its coverage and blanks what too
    few of its frames cover): the bounding box of all the footprints, so a camera pointing
    off the field of the other (SBT C1/C2) still counts.  box=(ra, dec, half-size arcmin):
    cut to that square around the trigger."""
    w0 = WCS(fits.Header(frames[0].wcs))
    cd = w0.pixel_scale_matrix
    # footprints in the first frame's pixels -> bounding box
    bx0, bx1, by0, by1 = -np.inf, np.inf, -np.inf, np.inf
    ux0, ux1, uy0, uy1 = np.inf, -np.inf, np.inf, -np.inf
    for f in frames:
        w = WCS(fits.Header(f.wcs))
        ny, nx = f.shape
        ex = np.r_[np.linspace(0, nx - 1, 9), np.full(9, nx - 1), np.linspace(nx - 1, 0, 9), np.zeros(9)]
        ey = np.r_[np.zeros(9), np.linspace(0, ny - 1, 9), np.full(9, ny - 1), np.linspace(ny - 1, 0, 9)]
        ra, dec = w.all_pix2world(ex, ey, 0)
        x, y = w0.all_world2pix(ra, dec, 0)
        # the inner box of this footprint (slightly rotated frames: the inner two values)
        sx, sy = np.sort(x[[0, 9, 18, 27]]), np.sort(y[[0, 9, 18, 27]])
        if union:
            ux0, ux1 = min(ux0, x.min()), max(ux1, x.max())
            uy0, uy1 = min(uy0, y.min()), max(uy1, y.max())
        bx0, bx1 = max(bx0, sx[1]), min(bx1, sx[2])
        by0, by1 = max(by0, sy[1]), min(by1, sy[2])
    if union:
        x0, x1, y0, y1 = ux0 + MARGIN, ux1 - MARGIN, uy0 + MARGIN, uy1 - MARGIN
    else:
        x0, x1, y0, y1 = bx0 + MARGIN, bx1 - MARGIN, by0 + MARGIN, by1 - MARGIN
    if box is not None:
        # only around the trigger: (ra, dec, half-size arcmin), inside the common footprint
        ra, dec, half = box
        cx, cy = w0.all_world2pix([[ra, dec]], 0)[0]
        hp = half * 60 / (np.sqrt(abs(np.linalg.det(cd))) * 3600)
        x0, x1, y0, y1 = max(x0, cx - hp), min(x1, cx + hp), max(y0, cy - hp), min(y1, cy + hp)
    if x1 - x0 < 100 or y1 - y0 < 100:
        raise SystemExit(f"the frames have (almost) no common footprint: {x1 - x0:.0f} x {y1 - y0:.0f} px")
    nx, ny = int(np.ceil(x1 - x0)), int(np.ceil(y1 - y0))
    ra_c, dec_c = w0.all_pix2world([(x0 + x1) / 2], [(y0 + y1) / 2], 0)
    h = fits.Header()
    h['SIMPLE'] = True     # a Montage template (pyrt-combine --skel) needs them
    h['BITPIX'] = -32
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


def project_cached(f, skel, work, cache, keep=False):
    """Frame f on the grid (pyrt-combine --skel of the frame alone: mProjectPP, flux
    conserving), NaN where it does not cover: cache/<stem>.fits, made once per observation and
    grid, reused by every later step.  Returns its path."""
    import subprocess
    path = f"{cache}/{f.stem}.fits"
    if os.path.exists(path):
        return path
    tmp, wtmp = f"{work}/{f.stem}-proj.fits", f"{work}/{f.stem}-projw.fits"
    cmd = ['pyrt-combine', '--skel', f"{work}/skel.hdr", '-u', '--no-selection', '--num-processes', '1',
           '-W', wtmp, '-o', tmp, f.path]
    r = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    if r.returncode or not os.path.exists(tmp):
        with open(f"{work}/{f.stem}-proj.log", 'w') as lf:
            lf.write(' '.join(cmd) + '\n' + r.stdout + r.stderr)
        raise RuntimeError(f"pyrt-combine (projection) failed for {f.stem}, see {f.stem}-proj.log")
    img = fits.getdata(tmp).astype(np.float32)
    if os.path.exists(wtmp):
        img[~(fits.getdata(wtmp) > 0)] = np.nan
    h = fits.Header()
    h['PROJOF'] = (os.path.basename(f.path), 'frame projected onto the observation grid')
    fits.PrimaryHDU(data=img, header=h).writeto(path + '.tmp', overwrite=True, output_verify='silentfix')
    os.replace(path + '.tmp', path)
    for x in (tmp, wtmp):
        if os.path.exists(x) and not keep:
            os.remove(x)
    return path


def combine_step(args, frames, members, wts, skel, shape, work, k, n, t):
    """1.-3. on a fixed grid (--mode combine): every frame projected onto it once per
    observation and cached (project_cached; --skel's directory/proj), then per bin the
    exposure-weighted mean of its frames with outlier rejection (stack_strip, as the cube mode,
    over the bin's own frames), each frame scaled to the first frame's zeropoint and NaN off its
    footprint: a camera pointing off the field of the other leaves no step in level, and where
    less than MIN_COVER of the bin's exposure covers a pixel the bin is blank there, not
    biased.  The deep image: the bins stacked with rejection, pixel by pixel.  Memory ~ one
    bin's frames over a strip of rows, not the step's cube.  Writes work/<bin>.fits, deep.fits."""
    zref = frames[0].zp
    exp0 = frames[0].exptime
    clip = args.stack == 'clip'
    # the cache belongs to the grid: a different grid, a fresh cache
    cache = os.path.join(os.path.dirname(os.path.abspath(args.skel)) if args.skel else work, 'proj')
    os.makedirs(cache, exist_ok=True)
    stamp = f"{cache}/skel.hdr"
    cur = open(f"{work}/skel.hdr").read()
    if not os.path.exists(stamp) or open(stamp).read() != cur:
        for x in os.listdir(cache):
            os.remove(os.path.join(cache, x))
        with open(stamp, 'w') as fh:
            fh.write(cur)
    t0 = time.time()
    new = [f for f in frames if not os.path.exists(f"{cache}/{f.stem}.fits")]
    with cf.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        paths = dict(zip([f.stem for f in frames],
                         ex.map(lambda f: project_cached(f, skel, work, cache, args.keep), frames)))
    log.info(f"{len(new)} frames projected onto {shape[1]}x{shape[0]} ({len(frames) - len(new)} from the cache) "
             f"in {time.time() - t0:.1f}s")

    def stack_keys(h, what):
        h['STACK'] = (what, 'framestep --mode combine')
        return h

    def strips(idx, wgt, groups):
        """stack_strip over the frames idx (scaled, NaN off their footprints), by rows."""
        hs = [fits.open(paths[frames[i].stem], memmap=True) for i in idx]
        ks = [np.float32(10 ** (0.4 * (zref - frames[i].zp))) for i in idx]
        skies = []
        for i, h, k_i in zip(idx, hs, ks):
            med, rms = _sky(h[0].data * k_i)
            gain = float(frames[i].hdr.get('GAIN') or 1.0)
            skies.append((med, rms, k_i / gain))
        out = np.zeros((len(groups),) + shape, np.float32)
        cov = np.zeros(shape, np.float32)
        nrej = np.zeros(shape, np.int16)
        rows = max(8, int(STACK_MEM // (len(idx) * shape[1] * 4 * 6)))
        for y0 in range(0, shape[0], rows):
            y1 = min(shape[0], y0 + rows)
            ya, yb = max(0, y0 - GROW), min(shape[0], y1 + GROW)
            X = np.stack([np.asarray(h[0].data[ya:yb], np.float32) * k_i for h, k_i in zip(hs, ks)])
            Sk = np.stack([_sky_at(sk[0], MESH, ya, yb, 0, shape[1]) for sk in skies])
            r = slice(y0 - ya, y0 - ya + (y1 - y0))
            o, c, nr = stack_strip(X, Sk, [sk[1] for sk in skies], [sk[2] for sk in skies], wgt, groups, clip)
            out[:, y0:y1] = o[:, r]
            cov[y0:y1] = c[r]
            nrej[y0:y1] = nr[r]
        for h in hs:
            h.close()
        return out, cov, nrej

    names = []
    t1 = time.time()
    for b, m in enumerate(members):
        fr = [frames[i] for i in m]
        ts = fr[0].t0
        name = (time.strftime('%Y%m%d%H%M%S', time.gmtime(ts)) + f"-{int((ts % 1) * 1000):03d}"
                + f"-{fr[0].hdr.get('FILTER', 'UNK')}-f{m[0]:03d}b{len(m)}")
        w = [wts[i] for i in m]
        o, cov, _ = strips(list(m), w, [list(range(len(m)))])
        good = cov >= MIN_COVER * sum(w)
        img = np.where(good, o[0] * np.float32(sum(f.exptime for f in fr) / exp0), np.nan).astype(np.float32)
        hdr = stack_keys(bin_header(skel, fr, len(fr), fr[0].hdr), 'clip' if clip else 'mean')
        hdr['COVERMIN'] = (MIN_COVER, 'blank where less of the exposure covers')
        hdr['COVERED'] = (round(float(good.mean()), 4), 'fraction of the grid with data')
        fits.PrimaryHDU(data=img, header=hdr).writeto(f"{work}/{name}.fits", overwrite=True)
        names.append(name)
    log.info(f"{len(names)} bins stacked in {time.time() - t1:.1f}s")

    # the deep image: the bins (each the whole exposure of its frames) stacked with rejection
    t2 = time.time()
    nb = len(names)
    hs = [fits.open(f"{work}/{nm}.fits", memmap=True) for nm in names]
    skies = []
    gain = float(frames[0].hdr.get('GAIN') or 1.0)
    for h in hs:
        med, rms = _sky(h[0].data)
        skies.append((med, rms, 1.0 / gain))
    out = np.zeros(shape, np.float32)
    nrej = np.zeros(shape, np.int16)
    rows = max(8, int(STACK_MEM // (nb * shape[1] * 4 * 6)))
    for y0 in range(0, shape[0], rows):
        y1 = min(shape[0], y0 + rows)
        ya, yb = max(0, y0 - GROW), min(shape[0], y1 + GROW)
        X = np.stack([np.asarray(h[0].data[ya:yb], np.float32) for h in hs])
        Sk = np.stack([_sky_at(sk[0], MESH, ya, yb, 0, shape[1]) for sk in skies])
        r = slice(y0 - ya, y0 - ya + (y1 - y0))
        o, _, nr = stack_strip(X, Sk, [sk[1] for sk in skies], [sk[2] for sk in skies], [1] * nb,
                               [list(range(nb))], clip)
        out[y0:y1] = o[0][r]
        nrej[y0:y1] = nr[r]
    for h in hs:
        h.close()
    hdr = stack_keys(bin_header(skel, frames, n, frames[0].hdr), 'bins stacked with rejection, sum')
    hdr['KINSTEP'] = (k, 'frames per bin of this step')
    hdr['NCLIPPED'] = (int(nrej.sum()), 'bin pixels rejected')
    fits.PrimaryHDU(data=out * np.float32(nb), header=hdr).writeto(f"{work}/deep.fits", overwrite=True)
    log.info(f"deep image: {nb} bins stacked in {time.time() - t2:.1f}s, {int(nrej.sum())} bin pixels rejected")
    return names


def cube_step(args, frames, members, wts, skel, shape, zref, work, k, n, t):
    """1.-3. of the module docstring: every frame resampled into one cube, the rejection over
    the whole series, the bins and the deep image (work/<bin>.fits, work/deep.fits)."""
    exp0 = frames[0].exptime
    for m in sky_frames(frames, args.sky_frame, dither_min=args.sky_dither):
        log.info(m)

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

    return bins


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
    ap.add_argument('--mode', choices=('combine', 'cube'), default='combine',
                    help='combine: every bin by pyrt-combine --clip onto the grid, the deep image '
                         'the clipped stack of the bins (default); cube: every frame resampled, one '
                         'rejection over the whole series (memory ~ frames x grid)')
    ap.add_argument('--skel', help='the grid (a FITS header file): used if it exists, else written '
                                   'there (one grid for all the steps of an observation)')
    ap.add_argument('--box', default='auto',
                    help="a new grid only around the trigger: 'auto' (GRB_RA/GRB_DEC/GRB_ERR of the "
                         f"frames, half-size max({BOX_MIN:g}', {BOX_ERR:g} x error)), 'off', or RA,DEC,ARCMIN")
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
    if args.skel and os.path.exists(args.skel):
        skel = fits.Header.fromtextfile(args.skel)
        shutil.copy(args.skel, f"{work}/skel.hdr")
        log.info(f"grid {skel['NAXIS1']}x{skel['NAXIS2']} from {args.skel}")
    else:
        box = None
        h0 = frames[0].hdr
        if args.box == 'auto' and h0.get('GRB_RA') is not None and h0.get('GRB_DEC') is not None:
            box = (float(h0['GRB_RA']), float(h0['GRB_DEC']),
                   max(BOX_MIN, BOX_ERR * 60 * float(h0.get('GRB_ERR') or 0)))
        elif args.box not in ('auto', 'off'):
            box = tuple(float(v) for v in args.box.split(','))
        skel = make_skel(frames, f"{work}/skel.hdr", box, union=args.mode == 'combine')
        if args.skel:
            shutil.copy(f"{work}/skel.hdr", args.skel)
        log.info(f"grid {skel['NAXIS1']}x{skel['NAXIS2']}" + (f" around {box[0]:.4f} {box[1]:+.4f}, "
                 f"half-size {box[2]:.1f}' at most" if box else "") +
                 (" within all the footprints" if args.mode == 'combine' else " within the common footprint"))
    shape = (skel['NAXIS2'], skel['NAXIS1'])
    zref = frames[0].zp
    if args.mode == 'combine':
        bins = combine_step(args, frames, members, wts, skel, shape, work, k, n, t)
    else:
        bins = cube_step(args, frames, members, wts, skel, shape, zref, work, k, n, t)

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
