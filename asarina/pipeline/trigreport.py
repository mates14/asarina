#!/usr/bin/env python3
"""
trigreport - the human side of a trigger follow-up: after every kinseries step a state
record, a light curve, a finding chart, a draft GCN and a web page, published on lascaux,
and a Telegram message when something new turns up.

  build   <obsdir>        report/ in the observation directory of asarina-kinseries:
                            report.json   the state record: trigger, steps (depth), the best
                                          candidates (class, position, magnitude now, trend,
                                          extrapolation +5/+15/+60 min, light curve)
                            lc.png        light curves of the best candidates, the bins' limits,
                                          the extrapolation
                            chart.png     the deep image around the best candidate (or the
                                          trigger position), the error box
                            gcn.txt       a draft GCN circular (never sent)
                            index.html    all of it, reloading itself
  publish <obsdir>        report/ -> lascaux:~/public_html/trigger/<site>-<target>/ (a local copy
                          and notify on lascaux; elsewhere rsync with the upload key ~/.ssh/l50,
                          restricted on lascaux to rrsync -wo ~/public_html/)
  scan    <rootdir>       on lascaux, from cron: notify for the pages updated since the last scan
  notify  <pagedir>       on lascaux: a Telegram message (bot of ~/gcn/config.json) when the best
                          candidate is new or changed its class; --dry-run prints it
  update  <obsdir>        build + publish (what the kinseries runner calls after a step)

Directory names: d50-<target>, sbt-<target>, orm-<target>, ll-<target> (the RTS2 target id is
only unique per telescope).  The page: https://lascaux.asu.cas.cz/m/trigger/<name>/
"""

import argparse
import datetime
import glob
import html
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import warnings

import numpy as np

SITE_DIR = {'D50': 'd50', 'SBT': 'sbt', 'FRAM-ORM': 'orm', 'FRAM-Auger': 'll'}
REPORT_HOST = os.environ.get('TRIGREPORT_HOST', 'lascaux.asu.cas.cz')
REPORT_ROOT = os.environ.get('TRIGREPORT_ROOT', '~/public_html/trigger')
REPORT_URL = os.environ.get('TRIGREPORT_URL', 'https://lascaux.asu.cas.cz/m/trigger')
SSH_KEY = os.environ.get('TRIGREPORT_SSH_KEY', '~/.ssh/l50')   # upload key (rrsync-restricted on lascaux)
TELEGRAM_CONFIG = os.environ.get('TRIGREPORT_TELEGRAM_CONFIG', '~/gcn/config.json')
TELEGRAM_CHAT = os.environ.get('TRIGREPORT_TELEGRAM_CHAT', 'private_chat_id')   # key in config.json 'telegram'
ERRATIC_ZVAR = 10.0     # an ERRATIC source inside the box this significant and
ERRATIC_MARGIN = 2.0    # [mag] this far above the limit is 'likely'
HIGH_AIRMASS = 2.5      # from here on red stars brighten against a calibration made higher up
                        # (FRAM-ORM 101827, X 3.8: five 'FLARE's 0.5-1 mag over quiescence)
SCHEMA = 4          # report.json layout (2: fit, error parts, points, conditions, ids; 3: fit chi2_ndf, moon_illum/alt, repeatability floor;
                    # 4: candidates' lightcurve_steps, limits_steps)
N_CAND = 5          # candidates in the record
N_PLOT = 3          # of them in the light-curve plot
MATCH = 1.5         # [arcsec] a candidate in a bin catalogue
CHART = 4.0         # [arcmin] finding chart size
KNOWN = os.environ.get('TRIGREPORT_KNOWN', '1') != '0'   # look the candidates up in Gaia / PS1 / LS-DR10
KNOWN_R = 4.0       # [arcsec] known objects summed within this of a candidate (a blended pair)
KNOWN_DMAG = 1.5    # [mag] a steady candidate this close to the known objects' flux is them
KNOWN_WAIT = 20.0   # [s] the report waits this long for the online query, then goes without it
                    # (the query finishes in the background and caches for the next step)
CHIP_MEM = 2e9      # [B] frames held for the chip-or-sky test (SBT 4k frames: ~14 of them)
CHIP_FRAMES = 24    # at most this many, spread over the step
CUT = 1.5           # [arcmin] candidate cutout side
CUT_FRAMES = 6      # single frames shown next to the deep cutout
NEAR_BOX = 3.0      # [error-box radii] or
NEAR_ARCMIN = 0.5   # [arcmin] "near the trigger" for the order of the candidates
WIDE_BOX_ARCMIN = 30.0   # [arcmin] error-box radius above which (Fermi GBM) only a source that
                         # varied during our observation counts: a brightened but steady star is a
                         # regular sight in a box of degrees, worth a mention, never an alert


# ---------------------------------------------------------------- reading

def _steps(obs):
    """Step directories whose transients run has finished (flares.txt written), in order."""
    out = []
    for d in sorted(glob.glob(os.path.join(obs, 'step[0-9][0-9]'))):
        if os.path.exists(os.path.join(d, 'flares.txt')) and os.path.exists(os.path.join(d, 'step.json')):
            out.append(d)
    return out


def _bins(stepdir):
    return sorted(glob.glob(os.path.join(stepdir, '*-[kf][0-9]*b*.ecsv')))


def _meta(stepdir):
    from astropy.table import Table
    return dict(Table.read(_bins(stepdir)[0], format='ascii.ecsv').meta)


def _trigger(meta):
    from pyrt.utils import grbinfo
    tr = grbinfo.get_trigger(meta.get('TARGET'), meta=meta)
    return tr if tr is not None and tr.get('t0') is not None else None


def _lightcurve(bins_tables, ra, dec, t0):
    """[[t - T0 (s), mag, err], ...] of the object at ra, dec in the bin catalogues (t0: JD)."""
    lc = []
    for t in bins_tables:
        if len(t) == 0:
            continue
        d = np.hypot((np.asarray(t['ALPHA_J2000']) - ra) * np.cos(np.radians(dec)),
                     np.asarray(t['DELTA_J2000']) - dec) * 3600
        i = int(np.argmin(d))
        tm = (t.meta['JD'] + t.meta.get('EXPTIME', 0) / 2 / 86400 - t0) * 86400
        if d[i] < MATCH:
            e = t['MAGERR_CALIB'][i]
            e = float(e) if e is not np.ma.masked and np.isfinite(e) else np.nan
            lc.append([round(tm, 1), round(float(t['MAG_CALIB'][i]), 3), round(e, 3)])
    return lc


def _points(bins_tables, ra, dec, t0, frames):
    """The light curve with everything per point: t - T0 [s] of the bin's middle, its half width [s],
    the calibrated magnitude and its photometric error, the frame's systematic floor and common mode
    (the afterglow fit uses mag + cm with sqrt(err^2 + floor^2)), the bin's limit and filter;
    a bin without the object gives mag None (the limit is then an upper limit)."""
    fjd = np.array([v[0] for v in frames.values()]) if frames else np.array([])
    fv = list(frames.values())
    out = []
    for t in bins_tables:
        half = t.meta.get('EXPTIME', 0) / 2
        jd = t.meta['JD'] + half / 86400
        pt = dict(t=round((jd - t0) * 86400, 1), half_width=round(half, 1), mag=None, err=None,
                  floor=None, cm=None, limit=_f(t.meta.get('MAGLIMIT'), 2), filter=str(t.meta.get('FILTER', '')))
        if len(fjd):
            i = int(np.argmin(np.abs(fjd - jd)))
            if abs(fjd[i] - jd) < 2 * half / 86400 + 1e-5:
                pt['cm'], pt['floor'] = _f(fv[i][1]), _f(fv[i][2])
        if len(t):
            d = np.hypot((np.asarray(t['ALPHA_J2000']) - ra) * np.cos(np.radians(dec)),
                         np.asarray(t['DELTA_J2000']) - dec) * 3600
            i = int(np.argmin(d))
            if d[i] < MATCH:
                pt['mag'], pt['err'] = _f(t['MAG_CALIB'][i]), _f(t['MAGERR_CALIB'][i])
        out.append(pt)
    return out


def _conditions(bins_tables):
    """Median of the bins: FWHM [arcsec], airmass, zeropoint, sky [mag/arcsec^2] (MAGZERO - 2.5 log
    MEDIAN/pixel^2, the bin's level as the zeropoint's counts), moon distance [deg], phase (RTS2 MOONPHA:
    the phase angle / 1.8, 0 full, 100 new), illuminated fraction and altitude [deg]."""
    def med(f):
        v = [f(t.meta) for t in bins_tables]
        v = [x for x in v if x is not None and np.isfinite(x)]
        return float(np.median(v)) if v else np.nan
    def sky(m):
        try:
            return m['MAGZERO'] - 2.5 * np.log10(m['MEDIAN'] / m['PIXEL']**2)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return None
    return dict(fwhm_arcsec=_f(med(lambda m: m.get('FWHM', np.nan) * m.get('PIXEL', np.nan)), 2),
                airmass=_f(med(lambda m: m.get('AIRMASS', np.nan)), 2),
                magzero=_f(med(lambda m: m.get('MAGZERO', np.nan)), 2),
                sky_mag_arcsec2=_f(med(sky), 2),
                moon_dist_deg=_f(med(lambda m: m.get('MOONDIST', np.nan)), 1),
                moon_phase=_f(med(lambda m: m.get('MOONPHA', np.nan)), 1),
                moon_illum=_f(med(lambda m: (1 + np.cos(np.radians(1.8 * m.get('MOONPHA', np.nan)))) / 2), 3),
                moon_alt_deg=_f(med(lambda m: m.get('MOONALT', np.nan)), 1))


def _frames(stepdir):
    """The afterglow search's per-frame common mode and floor (pyrt.core.afterglow.frame_stats)."""
    try:
        from astropy.table import Table
        from pyrt.core.afterglow import frame_stats
        return frame_stats(Table.read(os.path.join(stepdir, 'rec.ecsv'), format='ascii.ecsv'))
    except Exception:
        return {}


def _fit(r, t0):
    """The afterglow candidate's power-law fits m = M + 2.5 alpha (x - x_ref), x = log10((t - T0)/min),
    with the covariance of (alpha, M); 'used' says which one extrapolates (recent, whole, or mean for
    a steady source)."""
    if 'X_REF' not in r.colnames or not np.isfinite(r['X_REF']):
        return None
    out = dict(x_ref=_f(r['X_REF'], 4), t_ref_s=_f(60 * 10**r['X_REF'], 1), used=str(r['FIT_USED']))
    for k, a, ae in (('whole', 'ALPHA', 'ALPHA_ERR'), ('recent', 'ALPHA_RECENT', 'ALPHA_RECENT_ERR')):
        K = k.upper()
        out[k] = dict(alpha=_f(r[a], 4), mag=_f(r[K + '_M'], 4),
                      cov=[[_f(r[K + '_CAA'], 6), _f(r[K + '_CAM'], 6)], [_f(r[K + '_CAM'], 6), _f(r[K + '_CMM'], 6)]],
                      chi2_ndf=_f(r[K + '_CHI2']) if K + '_CHI2' in r.colnames else None,
                      ndf=int(r[K + '_NDF']) if K + '_NDF' in r.colnames else None)
    return out


def _extrap_list(r, t_last_s):
    out = []
    for k in (5, 15, 60):
        e = {p: _f(r['MAG_%d_E%s' % (k, p)]) if 'MAG_%d_E%s' % (k, p) in r.colnames else None
             for p in ('STAT', 'DIFF', 'CURV', 'CAP', 'FLOOR')}
        model = [v for p, v in e.items() if p != 'STAT' and v is not None]
        out.append(dict(dt_min=k, t_since_t0_s=_f(t_last_s + 60 * k, 0) if t_last_s is not None else None,
                        mag=_f(r['MAG_%d' % k]), err=_f(r['MAG_%d_ERR' % k]), err_stat=e['STAT'],
                        err_model=_f(np.sqrt(np.sum(np.square(model)))) if model else None,
                        err_parts=dict(diff=e['DIFF'], curv=e['CURV'], cap=e['CAP'], floor=e['FLOOR'])))
    return out


def _limits(bins_tables, t0):
    return [[round((t.meta['JD'] + t.meta.get('EXPTIME', 0) / 2 / 86400 - t0) * 86400, 1),
             round(float(t.meta.get('MAGLIMIT', np.nan)), 2)] for t in bins_tables]


def _f(x, nd=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


# ---------------------------------------------------------------- build

def _known(cands, obs, stepdir):
    """Mark the steady candidates that are known objects the comparison catalogue (ATLAS-RefCat2:
    stars only) lacks - galaxies above all (GRB 190919A: three steady 'NEW' R ~ 17 objects 5-10
    box radii out, all Legacy Survey DR10 galaxies).  pyrt.core.deepfield.known_sources (Gaia DR3,
    Pan-STARRS or LS-DR10, ATLAS) for the whole field, cached in <obs>/known.ecsv; a candidate is
    'known' when the brightest source's objects within KNOWN_R sum to within KNOWN_DMAG of it.
    Only steady ones: a variable light curve on a known object stays a candidate."""
    if not KNOWN or not cands:
        return
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
        from pyrt.core.deepfield import known_sources
        h = fits.getheader(os.path.join(stepdir, 'deep.fits'))
        w = WCS(h)
        ny, nx = h['NAXIS2'], h['NAXIS1']
        ra0, de0 = w.all_pix2world([[nx / 2, ny / 2]], 0)[0]
        corners = w.all_pix2world([[0, 0], [nx, 0], [0, ny], [nx, ny]], 0)
        rad = max(np.hypot(((c[0] - ra0 + 180) % 360 - 180) * np.cos(np.radians(de0)), c[1] - de0)
                  for c in corners) + 0.01
        # online (Gaia, Pan-STARRS / Legacy Survey): never hold the real-time report up for long
        import threading
        box = {}

        def query():
            try:
                box['t'] = known_sources(float(ra0), float(de0), float(rad), cache=os.path.join(obs, 'known.ecsv'))
            except Exception as e:
                box['e'] = e
        th = threading.Thread(target=query)
        th.start()
        th.join(KNOWN_WAIT)
        if th.is_alive():
            print(f"trigreport: known objects not in after {KNOWN_WAIT:.0f} s, this report goes without them",
                  file=sys.stderr)
            return
        if 'e' in box:
            raise box['e']
        t = box.get('t')
    except Exception as e:
        print(f"trigreport: no known-object check ({e})", file=sys.stderr)
        return
    if t is None or not len(t):
        return
    kra, kde = np.asarray(t['RA'], float), np.asarray(t['DEC'], float)
    kmag, ksrc = np.asarray(t['MAG'], float), np.array([str(x) for x in t['SRC']])
    for c in cands:
        if _variable(c) or c.get('mag_mean') is None:
            continue
        d = np.hypot((kra - c['ra']) * np.cos(np.radians(c['dec'])), kde - c['dec']) * 3600
        best = None
        for src in set(ksrc[d < KNOWN_R]):
            m = (d < KNOWN_R) & (ksrc == src) & np.isfinite(kmag)
            if m.any():
                tot = -2.5 * np.log10(np.sum(10 ** (-0.4 * kmag[m])))
                if best is None or tot < best[1]:
                    best = (src, tot, float(d[m].min()))
        if best is not None and best[1] - c['mag_mean'] <= KNOWN_DMAG:
            c['known'] = dict(src=best[0], mag=_f(best[1], 2), dist_arcsec=_f(best[2], 1))


def _defects(cands, obs, stepdir):
    """Mark the candidates that stay on the chip, not on the sky: pyrt.core.deepfield.chip_or_sky
    on the step's frames (the deep search's DEFECT test).  A hot pixel the dark left survives the
    bins' clipped stacks while the pointing holds (FRAM shifts it every 3 frames) and looks like a
    source in the deep image too (GRB 190919A N: two sharp NEW 17.1-17.4 a few pixels in size,
    the deep search's "5 staying on the chip")."""
    if not cands:
        return
    try:
        from astropy.io import fits
        from pyrt.core.deepfield import chip_or_sky, CHIP_Z, CHIP_DRIFT
        sj = json.load(open(os.path.join(stepdir, 'step.json')))
        fr = [p for p in (os.path.join(obs, 'frames', n + '.fits') for n in sj.get('frames', [])) if os.path.exists(p)]
        if fr:
            # chip_or_sky holds the frames whole (float64): a bounded, evenly spread subset
            h = fits.getheader(fr[0])
            nmax = min(CHIP_FRAMES, int(CHIP_MEM // (8 * h['NAXIS1'] * h['NAXIS2'])))
            if nmax < 4:
                return
            fr = [fr[i] for i in sorted(set(np.linspace(0, len(fr) - 1, min(nmax, len(fr))).round().astype(int)))]
        ra = np.array([c['ra'] for c in cands], float)
        de = np.array([c['dec'] for c in cands], float)
        z, drift, _, _ = chip_or_sky(fr, ra, de, np.ones(len(fr)))
    except Exception as e:
        print(f"trigreport: no chip-or-sky test ({e})", file=sys.stderr)
        return
    for c, zz in zip(cands, z):
        if np.isfinite(zz):
            c['chip_z'] = _f(zz, 1)
            if drift >= CHIP_DRIFT and zz < -CHIP_Z:
                c['defect'] = dict(chip_z=_f(zz, 1), drift_px=_f(drift, 1))


def _cut(img, x, y, r):
    """(2r+1)^2 cutout centred on the pixel nearest (x, y), NaN outside the image."""
    ix, iy = int(round(x)), int(round(y))
    out = np.full((2 * r + 1, 2 * r + 1), np.nan)
    ny, nx = img.shape
    x0, x1, y0, y1 = max(0, ix - r), min(nx, ix + r + 1), max(0, iy - r), min(ny, iy + r + 1)
    if x0 < x1 and y0 < y1:
        out[y0 - iy + r:y1 - iy + r, x0 - ix + r:x1 - ix + r] = img[y0:y1, x0:x1]
    return out, x - ix, y - iy


def plot_cutouts(items, deep, frames, path, t0=None, size=CUT, nframes=CUT_FRAMES):
    """One row per (label, ra, dec): the deep image and nframes single frames spread over the
    step, size arcmin square, each scaled on its own.  A real source is in the deep cutout and
    (bright enough) in the frames at the circle; a bad pixel sits off it or in only a few frames
    (the pointing shifts every few frames), a blend or a galaxy shows its shape.  deep: FITS path;
    frames: FITS paths (the step's, in time order); t0: JD for the frame labels."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.wcs import WCS
    if not items:
        return False
    paths = [deep] + ([frames[i] for i in sorted(set(np.linspace(0, len(frames) - 1, min(nframes, len(frames)))
                                                     .round().astype(int)))] if frames else [])
    # one image at a time, only its cutouts kept (SBT 4k frames)
    cols = []
    for p in paths:
        if p is None or not os.path.exists(p):
            continue
        with fits.open(p, memmap=True) as h:
            hd = h[0].header
            jd = hd.get('JD')
            lab = 'deep' if p == deep else (f"+{(jd - t0) * 86400:.0f} s" if t0 and jd else os.path.basename(p)[:14])
            w = WCS(hd)
            img = h[0].data
            scale = np.sqrt(abs(np.linalg.det(w.pixel_scale_matrix))) * 3600
            r = max(5, int(size * 30 / scale))
            cuts = []
            for _, ra, dec in items:
                x, y = w.all_world2pix([[ra, dec]], 0)[0]
                cuts.append(_cut(img, x, y, r))
            cols.append((lab, scale, r, [(c.astype(float), dx, dy) for c, dx, dy in cuts]))
            del img
    if not cols:
        return False
    nr, nc = len(items), len(cols)
    fig, axs = plt.subplots(nr, nc, figsize=(1.35 * nc + 0.3, 1.45 * nr + 0.2), squeeze=False)
    for i, (label, ra, dec) in enumerate(items):
        for j, (lab, scale, r, cuts) in enumerate(cols):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            cut, dx, dy = cuts[i]
            v = cut[np.isfinite(cut) & (cut != 0)]
            if len(v) > 10:
                # the sky's noise sets the scale, the candidate's own peak its top (a bright
                # neighbour saturates rather than washing a faint candidate out)
                med = np.median(v)
                sig = 1.4826 * np.median(np.abs(v - med)) or 1.0
                ix, iy = int(round(dx)) + r, int(round(dy)) + r
                pk = cut[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3]
                pk = np.nanmax(pk) if np.isfinite(pk).any() else med
                lo, hi = med - 2 * sig, max(med + 6 * sig, pk)
                ax.imshow(cut, origin='lower', cmap='gray_r', vmin=lo, vmax=max(hi, lo + 1e-6),
                          extent=(-r - 0.5, r + 0.5, -r - 0.5, r + 0.5), interpolation='nearest')
            ax.add_patch(plt.Circle((dx, dy), 8 / scale, fill=False, color='#d62728', lw=1))
            ax.set_xlim(-r - 0.5, r + 0.5)
            ax.set_ylim(-r - 0.5, r + 0.5)
            if i == 0:
                ax.set_title(lab, fontsize=7)
            if j == 0:
                ax.set_ylabel(label, fontsize=7)
    fig.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.3)
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return True


def build(obs):
    from astropy.table import Table
    warnings.simplefilter('ignore')
    steps = _steps(obs)
    if not steps:
        return None
    last = steps[-1]
    meta = _meta(last)
    trig = _trigger(meta)
    from pyrt.utils import grbinfo
    site = grbinfo.site_key(meta) or str(meta.get('TELESCOP', 'unknown'))
    name = f"{SITE_DIR.get(site, site.lower())}-{meta.get('TARGET')}"
    if trig is not None:
        t0 = trig['jd0']
    else:
        t0 = min(Table.read(_bins(s)[0], format='ascii.ecsv').meta['JD'] for s in steps)
    out = os.path.join(obs, 'report')
    os.makedirs(out, exist_ok=True)

    step_info, bins_steps = [], []
    for s in steps:
        sj = json.load(open(os.path.join(s, 'step.json')))
        bt = [Table.read(b, format='ascii.ecsv') for b in _bins(s)]
        bins_steps.append((sj['step'], bt))
        tt = [(t.meta['JD'] - t0) * 86400 for t in bt]
        step_info.append(dict(step=sj['step'], bins=sj.get('bins_ok'), bin_seconds=sj.get('bin_seconds'),
                              deep_limit=_f(sj.get('deep_limit'), 2), seconds=sj.get('seconds'),
                              bin_exposure_s=_f(sj.get('bin_exposure'), 1),
                              t_first=_f(min(tt), 0), t_last=_f(max(tt) + bt[-1].meta.get('EXPTIME', 0), 0),
                              bin_limit=_f(np.nanmedian([t.meta.get('MAGLIMIT', np.nan) for t in bt]), 2),
                              exposure_bin_s=_f(np.median([t.meta.get('EXPTIME', np.nan) for t in bt]), 2),
                              exposure_total_s=_f(np.nansum([t.meta.get('EXPTIME', np.nan) for t in bt]), 1),
                              **_conditions(bt)))
    bins_last = bins_steps[-1][1]

    cands = []
    agf = os.path.join(last, 'afterglow.ecsv')
    if os.path.exists(agf):
        # all of them: the cut comes after the ordering below (near the trigger first), a candidate
        # next to the trigger may rank low by its score (230818A at SBT: 11th behind catalogue gaps)
        for r in Table.read(agf, format='ascii.ecsv'):
            t_last_s = _f((r['T_LAST'] - t0) * 86400, 1)
            cands.append(dict(kind='afterglow', cls=str(r['CLASS']), ra=_f(r['RA'], 6), dec=_f(r['DEC'], 6),
                              n=int(r['N']), chi2=_f(r['CHI2']), curv_z=_f(r['CURV_Z'], 2), rate_err=_f(r['RATE_ERR']),
                              t_last_s=t_last_s, span_min=_f(r['SPAN'], 2), frac=_f(r['FRAC'], 2),
                              n_cloudy=int(r['N_CLOUDY']) if 'N_CLOUDY' in r.colnames else None,
                              fit=_fit(r, t0), extrap_list=_extrap_list(r, t_last_s),
                              mag_mean=_f(r['MAG_MEAN']), mag_last=_f(r['MAG_LAST']), z_var=_f(r['Z_VAR'], 1),
                              cont=_f(r['CONT'], 2), alpha=_f(r['ALPHA']), alpha_err=_f(r['ALPHA_ERR']),
                              alpha_recent=_f(r['ALPHA_RECENT']), alpha_recent_err=_f(r['ALPHA_RECENT_ERR']),
                              rate=_f(r['RATE']), margin=_f(r['MARGIN'], 2),
                              dist_arcmin=_f(r['DIST'], 2), dist_box=_f(r['DIST_BOX'], 2), score=_f(r['S'], 1),
                              extrap={str(k): [_f(r['MAG_%d' % k], 2), _f(r['MAG_%d_ERR' % k], 2)] for k in (5, 15, 60)}))
    flf = os.path.join(last, 'flares.ecsv')
    if os.path.exists(flf):
        ft = Table.read(flf, format='ascii.ecsv')
        cls = np.array([str(c) for c in (ft['CLASS'].filled('') if hasattr(ft['CLASS'], 'filled') else ft['CLASS'])])
        for r, c in zip(ft, cls):
            if c not in ('FLARE', 'FLARE?'):
                continue
            cands.append(dict(kind='flare', cls=c, ra=_f(r['ALPHA_J2000'], 6), dec=_f(r['DELTA_J2000'], 6),
                              mag_mean=_f(r['MAG_CALIB']), catmag=_f(r['CATMAG']), excess=_f(r['EXCESS']),
                              excess_err=_f(r['EXCESS_ERR']), excess_last=_f(r['EXCESS_LAST']),
                              z_decay=_f(r['Z_DECAY'], 1), tau_min=_f(r['TAU'], 0), red=bool(r['RED']),
                              dwarf=bool(r['DWARF']), dist_arcmin=_f(r['DIST'], 2), dist_box=_f(r['DIST_BOX'], 2),
                              score=_f(r['S'] + r['PRIOR'], 1),
                              trend=_f(r['TREND']), trend_err=_f(r['TREND_ERR']), excess_max=_f(r['EXCESS_MAX']),
                              colx=_f(r['COLX']), colr=_f(r['COLR']), pm=_f(r['PM']),
                              excess_extrap={'0': [_f(r['FLARE_NOW'], 2), None], '30': [_f(r['FLARE_NEXT'], 2), None]}))
    # order: near the trigger first (within NEAR_BOX error-box radii or NEAR_ARCMIN), then the clear
    # classes (a decaying flare, an ordered afterglow), then the score; a flare far out in the
    # field is not what the trigger was about
    clear = ('FLARE', 'FADING', 'RISING', 'TURNING')
    def near(c):
        db, da = c.get('dist_box'), c.get('dist_arcmin')
        return (db is None and da is None) or (db is not None and db <= NEAR_BOX) or (da is not None and da <= NEAR_ARCMIN)
    wide = trig is not None and trig['errorbox'] * 60 > WIDE_BOX_ARCMIN
    _known(cands, obs, last)
    _defects(cands, obs, last)
    cands.sort(key=lambda c: ('defect' in c or 'known' in c, wide and not _variable(c), not near(c), c['cls'] not in clear,
                              -(c['score'] or 0)))
    cands = cands[:N_CAND]
    frames = _frames(last)
    for c in cands:
        c['id'] = _key(c)     # rounded position: the same for the same source in every step and telescope
        c['lightcurve'] = _lightcurve(bins_last, c['ra'], c['dec'], t0)
        c['points'] = _points(bins_last, c['ra'], c['dec'], t0, frames)
        # every step's light curve: an early step often shows what the last one has averaged
        # away (or a hot pixel's early points, where the last one shows a steady star)
        c['lightcurve_steps'] = [[k, _lightcurve(bt, c['ra'], c['dec'], t0)] for k, bt in bins_steps]
    best = cands[0] if cands and 'known' not in cands[0] and 'defect' not in cands[0] and not (wide and not _variable(cands[0])) else None
    status = 'nothing' if best is None else ('candidate' if best['cls'] in ('NEW', 'ERRATIC', 'SPIKE', 'FLARE?') else 'likely')
    if status == 'likely' and not near(best):
        # ordered, but far outside the error box: a variable star, not this trigger's
        # counterpart (FRAM LL 52007: TURNING 11' out of a 0.08' box; GRB 190919A: FADING
        # 13' out of 1.5', the afterglow itself NEW inside the box)
        status = 'candidate'
    if status == 'candidate' and not near(best):
        # nothing near the trigger: an unordered or steady source far outside the box is not
        # its counterpart and must not steer the follow-up (still listed below)
        status = 'nothing'
    if best is not None and best['cls'] == 'ERRATIC' and (best.get('dist_box') or 99) <= 1 and \
            (best.get('z_var') or 0) >= ERRATIC_ZVAR and (best.get('margin') or 0) >= ERRATIC_MARGIN:
        # inside the box, uncatalogued, bright and strongly variable without order: an afterglow
        # with flares (GRB 250702F at D50, 19 sigma, 3.7 mag above the limit), not noise
        status = 'likely'
    caution = []
    x = step_info[-1].get('airmass') if step_info else None
    if x is not None and x >= HIGH_AIRMASS:
        caution.append(f"high airmass ({x:.1f}): the colour of the extinction is not calibrated, "
                       "red stars look brighter than their catalogue magnitudes")
        if status == 'likely' and best['cls'] == 'FLARE':
            status = 'candidate'

    rec = dict(caution=caution, schema=SCHEMA, units=dict(t='s since T0 (bin middle)', alpha='m = M + 2.5 alpha log10(t - T0), > 0 fading',
                                           rate='mag/h, linear fit', margin='median bin limit - mean mag',
                                           cont='normalised light-curve length, 1 = noise, < 0.6 ordered',
                                           extrap='+dt min after the last point: [mag, total error]',
                                           excess='flare: catalogue mag - measured, positive = brighter',
                                           chi2='candidate: chi2/(n-1) about the weighted mean (the variability test, '
                                                'not the power law: fit.whole/recent.chi2_ndf is that)',
                                           chi2_ndf='fit: chi2/ndf of the power law; cov is scaled by it when > 1',
                                           err='mag, photometric, already scaled by sqrt(WSSRNDF) of the calibration',
                                           floor='mag, the frame\'s repeatability beyond err (field stars against their '
                                                 'own median), add in quadrature; 0 = err covers it, null = not measured',
                                           cm='mag, the frame\'s common mode, added to mag in the fit',
                                           moon_phase='RTS2 MOONPHA: lunar phase angle / 1.8, 0 full, 100 new',
                                           moon_illum='illuminated fraction, 0..1', moon_alt_deg='deg, < 0 below the horizon',
                                           moon_dist_deg='deg from the field'),
               name=name, site=site, target=meta.get('TARGET'), object=str(meta.get('OBJECT', '')),
               filter=str(meta.get('FILTER', '')), updated=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S'),
               trigger=(dict(name=trig['name'], t0=trig['t0'], ra=_f(trig['ra'], 5), dec=_f(trig['dec'], 5),
                             errorbox_arcmin=_f(trig['errorbox'] * 60, 2), type=trig.get('type'))
                        if trig is not None else None),
               step=step_info[-1]['step'], steps=step_info, status=status, best=best, candidates=cands,
               wide=wide,
               limits=_limits(bins_last, t0),
               limits_steps=[[k, _limits(bt, t0)] for k, bt in bins_steps])
    with open(os.path.join(out, 'report.json.tmp'), 'w') as f:
        json.dump(rec, f, indent=1)
    os.replace(os.path.join(out, 'report.json.tmp'), os.path.join(out, 'report.json'))

    for fn, what in ((plot_lc, 'lc.png'), (plot_chart, 'chart.png')):
        try:
            fn(rec, last, os.path.join(out, what))
        except Exception as e:
            print(f"trigreport: {what}: {e}", file=sys.stderr)
    try:
        sj = json.load(open(os.path.join(last, 'step.json')))
        fr = [p for p in (os.path.join(obs, 'frames', n + '.fits') for n in sj.get('frames', [])) if os.path.exists(p)]
        items = [(f"{i + 1} {c['cls']}{' known' if 'known' in c else ''}{' DEFECT' if 'defect' in c else ''} "
                  f"{_mag(c):.1f}", c['ra'], c['dec'])
                 for i, c in enumerate(rec['candidates'])]
        if rec['trigger']:
            items.insert(0, ('trigger', rec['trigger']['ra'], rec['trigger']['dec']))
        cpath = os.path.join(out, 'cutouts.png')
        rec['cutouts'] = plot_cutouts(items, os.path.join(last, 'deep.fits'), fr, cpath, t0=t0)
        if not rec['cutouts'] and os.path.exists(cpath):
            os.remove(cpath)
    except Exception as e:
        print(f"trigreport: cutouts.png: {e}", file=sys.stderr)
    with open(os.path.join(out, 'gcn.txt'), 'w') as f:
        f.write(gcn_text(rec))
    with open(os.path.join(out, 'index.html'), 'w') as f:
        f.write(page(rec))
    return out


# ---------------------------------------------------------------- products

STEP_MARKERS = ['o', 's', '^', 'D', 'P', 'X', 'h', '<', '>', 'p', '*', 'd']   # no 'v': the limits


def plot_lc(rec, stepdir, path):
    """The best candidates' light curves from every step (marker = step, colour = candidate; the
    last step filled, the earlier ones open) on a logarithmic time axis, the bins' limits, the
    extrapolation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colours = ['#d62728', '#1f77b4', '#2ca02c']
    steps = [k for k, _ in rec.get('limits_steps') or []] or [rec['step']]
    mk = {k: STEP_MARKERS[i % len(STEP_MARKERS)] for i, k in enumerate(steps)}
    last = steps[-1]

    def arr(lc):
        return np.array([x for x in lc if x[2] is not None and x[1] is not None and x[0] > 0], dtype=float).reshape(-1, 3)

    # the magnitude range follows the best candidate (all its steps' points, its quiescence, its
    # extrapolation); other candidates and the limits are drawn only where they fall into it
    window, times, cand_handles, offs = None, [], [], []
    for i, (c, col) in enumerate(zip(rec['candidates'][:N_PLOT], colours)):
        per_step = [(k, arr(lc)) for k, lc in (c.get('lightcurve_steps') or [[last, c['lightcurve']]])]
        per_step = [(k, lc) for k, lc in per_step if len(lc)]
        if not per_step:
            continue
        allp = np.vstack([lc for _, lc in per_step])
        ex = []
        if c['kind'] == 'afterglow' and c['cls'] != 'NEW':
            lcl = arr(c['lightcurve'])
            if len(lcl):
                ex = [(lcl[-1, 0] + 60 * int(k), v[0], v[1]) for k, v in c['extrap'].items() if v[0] is not None]
        quiet = c.get('catmag') if c['kind'] == 'flare' else None
        label = f"{c['cls']} {c['ra']:.5f} {c['dec']:+.5f}"
        if window is None:
            good = allp[allp[:, 2] < 1.0] if np.any(allp[:, 2] < 1.0) else allp
            # the extrapolation only where it is meaningful (a SPIKE's runs off to 23 mag)
            ys = (list(good[:, 1] - good[:, 2]) + list(good[:, 1] + good[:, 2])
                  + [e[1] for e in ex if e[2] is not None and e[2] < 0.5])
            if quiet is not None:
                ys.append(quiet)
            lo, hi = min(ys), max(ys)
            pad = max(0.1, 0.1 * (hi - lo), (0.4 - (hi - lo)) / 2)
            window = (lo - pad, hi + pad)
        elif not (window[0] <= np.median(allp[:, 1]) <= window[1]):
            # far brighter or fainter than the best one: in the panel below, relative to its
            # quiescence (the catalogue's for a flare, else its own median)
            ref = quiet if quiet is not None else float(np.median(allp[:, 1]))
            offs.append((per_step, ref, col))
            cand_handles.append(Line2D([], [], ls='-', lw=6, color=col, label=f"{label} (below, {ref:.1f} + Δ)"))
            continue
        cand_handles.append(Line2D([], [], ls='-', lw=6, color=col, label=label))
        for k, lc in per_step:
            filled = k == last
            ax.errorbar(lc[:, 0], lc[:, 1], yerr=lc[:, 2], fmt=mk.get(k, 'o'), ms=5 if filled else 4.5,
                        color=col, mfc=col if filled else 'none', alpha=1.0 if filled else 0.7,
                        elinewidth=1 if filled else 0.7, capsize=0, zorder=3 if filled else 2)
            times += list(lc[:, 0])
        if quiet is not None:
            ax.axhline(quiet, ls='--', lw=1, color=col, alpha=0.7)
            cand_handles.append(Line2D([], [], ls='--', lw=1, color=col, label=f"quiescence (catalogue) {quiet:.2f}"))
        if len(ex):
            ex = np.array(ex, dtype=float)
            times += list(ex[:, 0])
            ax.errorbar(ex[:, 0], ex[:, 1], yerr=ex[:, 2], fmt='_', ms=10, color=col, alpha=0.6, capsize=3)
    for k, lim in (rec.get('limits_steps') or [[last, rec['limits']]]):
        lim = np.array([x for x in lim if x[0] > 0 and x[1] is not None], dtype=float).reshape(-1, 2)
        if window is not None and len(lim):
            lim = lim[lim[:, 1] <= window[1]]
        if len(lim):
            ax.plot(lim[:, 0], lim[:, 1], 'v', color='grey', ms=4 if k == last else 3,
                    alpha=0.6 if k == last else 0.25, zorder=1)
            times += list(lim[:, 0])
    if window is not None:
        ax.set_ylim(window[0], window[1])
    if times and min(times) > 0:
        from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter
        ax.set_xscale('log')
        span = max(times) / min(times)
        if span < 30:   # under a decade and a half: label more than the powers of ten
            ax.xaxis.set_major_locator(LogLocator(subs=range(1, 10) if span < 4 else (1, 2, 3, 5)))
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.invert_yaxis()
    xax = ax
    if offs:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        xax = make_axes_locatable(ax).append_axes('bottom', size='40%', pad=0.1, sharex=ax)
        ax.tick_params(labelbottom=False)
        xax.axhline(0, ls='--', lw=1, color='grey', alpha=0.7)
        for per_step, ref, col in offs:
            for k, lc in per_step:
                filled = k == last
                xax.errorbar(lc[:, 0], lc[:, 1] - ref, yerr=lc[:, 2], fmt=mk.get(k, 'o'), ms=5 if filled else 4.5,
                             color=col, mfc=col if filled else 'none', alpha=1.0 if filled else 0.7,
                             elinewidth=1 if filled else 0.7, capsize=0)
        xax.invert_yaxis()
        if times and min(times) > 0:
            ax.set_xlim(min(times) / 1.03, max(times) * 1.03)
        xax.set_ylabel('Δmag', fontsize=9)
        xax.grid(alpha=0.3, which='both')
    xax.set_xlabel('time since trigger [s]' if rec['trigger'] else 'time since the first frame [s]')
    ax.set_ylabel(f"magnitude ({rec['filter']}, calibrated to r)" if rec['filter'] in ('N', 'clear') else 'magnitude')
    t = rec['trigger']['name'] if rec['trigger'] else rec['object']
    ax.set_title(f"{rec['site']} {t}, steps {steps[0]}..{last}" if len(steps) > 1 else f"{rec['site']} {t}, step {last}")
    ax.grid(alpha=0.3, which='both')
    bsec = {s['step']: s.get('bin_exposure_s') or s.get('exposure_bin_s') for s in rec.get('steps', [])}
    step_handles = [Line2D([], [], ls='none', marker=mk[k], color='k', mfc='k' if k == last else 'none', ms=5,
                           label=f"step {k}" + (f" ({bsec[k]:g} s bins)" if bsec.get(k) else ''))
                    for k in steps]
    step_handles.append(Line2D([], [], ls='none', marker='v', color='grey', ms=4, label='bin limit'))
    ax.legend(handles=cand_handles + step_handles, fontsize=7.5, loc='center left', bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_chart(rec, stepdir, path):
    """The deep image around the best candidate (or the trigger position)."""
    deep = os.path.join(stepdir, 'deep.fits')
    if not os.path.exists(deep):
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.wcs import WCS
    with fits.open(deep) as h:
        img = h[0].data.astype(float)
        w = WCS(h[0].header)
    best = rec['best']
    if best is not None:
        ra, dec = best['ra'], best['dec']
    elif rec['trigger']:
        ra, dec = rec['trigger']['ra'], rec['trigger']['dec']
    else:
        return
    x, y = w.all_world2pix([[ra, dec]], 0)[0]
    scale = np.sqrt(abs(np.linalg.det(w.pixel_scale_matrix))) * 3600   # arcsec/px
    size = CHART
    if best is None and rec['trigger'] and rec['trigger']['errorbox_arcmin']:
        size = max(CHART, 2.2 * rec['trigger']['errorbox_arcmin'])   # nothing found: the whole error box
    r = int(size * 60 / scale / 2)
    ny, nx = img.shape
    x0, x1, y0, y1 = max(0, int(x) - r), min(nx, int(x) + r), max(0, int(y) - r), min(ny, int(y) + r)
    cut = img[y0:y1, x0:x1]
    v = cut[np.isfinite(cut) & (cut != 0)]
    lo, hi = (np.percentile(v, 5), np.percentile(v, 99.5)) if len(v) else (0, 1)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cut, origin='lower', cmap='gray_r', vmin=lo, vmax=hi,
              extent=(x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5))   # pixel centres on the integers, as world2pix(.., 0)
    for c, col in zip(rec['candidates'][:N_PLOT], ('#d62728', '#1f77b4', '#2ca02c')):
        cx, cy = w.all_world2pix([[c['ra'], c['dec']]], 0)[0]
        ax.add_patch(plt.Circle((cx, cy), 8 / scale * 2, fill=False, color=col, lw=1.5))
    if rec['trigger'] and rec['trigger']['errorbox_arcmin']:
        tx, ty = w.all_world2pix([[rec['trigger']['ra'], rec['trigger']['dec']]], 0)[0]
        ax.add_patch(plt.Circle((tx, ty), rec['trigger']['errorbox_arcmin'] * 60 / scale, fill=False,
                                color='orange', ls='--', lw=1))
    # north and east, from the corner they point away from
    arm = 0.6 / 60   # 0.6' arrows
    cra, cdec = w.all_pix2world([[(x0 + x1) / 2, (y0 + y1) / 2]], 0)[0]
    vec = []
    for dra, ddec, lab in ((0, arm, 'N'), (arm / np.cos(np.radians(cdec)), 0, 'E')):
        ex, ey = w.all_world2pix([[cra + dra, cdec + ddec]], 0)[0]
        vec.append((ex - (x0 + x1) / 2, ey - (y0 + y1) / 2, lab))
    sx = x0 + 0.08 * (x1 - x0) if sum(v[0] for v in vec) >= 0 else x1 - 0.08 * (x1 - x0)
    sy = y0 + 0.08 * (y1 - y0) if sum(v[1] for v in vec) >= 0 else y1 - 0.08 * (y1 - y0)
    for vx, vy, lab in vec:
        ax.annotate('', xy=(sx + vx, sy + vy), xytext=(sx, sy), arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5))
        ax.text(sx + 1.25 * vx, sy + 1.25 * vy, lab, color='#1f77b4', ha='center', va='center', fontweight='bold')
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{size:.0f}' around {ra:.5f} {dec:+.5f} (deep, step {rec['step']})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _variable(c):
    """the source varied during our observation: an ordered afterglow light curve or a FLARE (a
    significant excess that decays; a FLARE? with a decay is a weak excess drifting)"""
    return c['cls'] in ('FADING', 'RISING', 'TURNING', 'FLARE')


def _alpha(c):
    """The decay index that counts: the recent half's when significant (2 sigma), else the whole curve's."""
    ar, er = c.get('alpha_recent'), c.get('alpha_recent_err')
    if ar is not None and er is not None and abs(ar) >= 2 * er:
        return ar, er
    return c.get('alpha'), c.get('alpha_err')


def _trend(c):
    if c['kind'] == 'flare':
        tags = [t for t, on in (('red', c.get('red')), ('dwarf', c.get('dwarf'))) if on]
        return (f"excess {c['excess']:+.2f}±{c['excess_err']:.2f} mag over the catalogue "
                f"{c['catmag']:.2f}, decay {c['z_decay']:.1f}σ" + (f", {'+'.join(tags)} star" if tags else ''))
    if c.get('alpha') is not None:
        a, e = c['alpha'], c['alpha_err']
        ar, er = c.get('alpha_recent'), c.get('alpha_recent_err')
        s = f"alpha {a:+.2f}±{e:.2f}"
        if ar is not None:
            s += f" (recent {ar:+.2f}±{er:.2f})"
        return s
    return f"{c['rate']:+.2f} mag/h" if c.get('rate') is not None else ''


def _deg2hms(ra, dec):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    return SkyCoord(ra * u.deg, dec * u.deg).to_string('hmsdms', sep=':', precision=2)


def gcn_text(rec):
    """A draft, to be edited: the facts, not the prose."""
    tr = rec['trigger']
    tname = tr['name'] if tr else rec['object']
    st = rec['steps'][-1]
    tel = {'D50': 'the 0.5m telescope D50 at Ondrejov (Czech Republic)',
           'SBT': 'the Small Binocular Telescope (SBT) at Ondrejov (Czech Republic)',
           'FRAM-Auger': 'the FRAM telescope at the Pierre Auger Observatory (Argentina)',
           'FRAM-ORM': 'the FRAM telescope at the Roque de los Muchachos Observatory (La Palma, Spain)'}.get(rec['site'], rec['site'])
    filt = str(rec.get('filter', ''))
    if filt in ('N', 'clear', ''):
        how, band = 'unfiltered, calibrated to r', 'r'
    else:
        band = filt.replace('Johnson_', '').replace('Sloan_', '')
        how = f'{band} filter'
    lines = [f"TITLE: {tname}: {rec['site']} optical observations", "",
             "[authors] report:", ""]
    lines.append(f"We observed the field of {tname} with {tel}, starting {st['t_first']:.0f} s after the trigger "
                 f"({how}" + (f", airmass {st['airmass']:.1f}" if st.get('airmass') else '') + ").  "
                 f"The sum of the frames to T0+{st['t_last']:.0f} s reaches "
                 f"{band} ~ {st['deep_limit']}.")
    best = rec['best']
    lines.append("")
    if best is None:
        lines.append(f"We do not detect any " + ("source varying during our observation within the error box"
                                                  if rec.get('wide') else
                                                  "new source within the error box, nor a brightening of any "
                                                  "catalogued star (a stellar flare)")
                     + f" in {st['bins']} frames of {st['bin_seconds']:.0f} s "
                     f"(limit {band} ~ {st['bin_limit']} each) from T0+{st['t_first']:.0f} s to T0+{st['t_last']:.0f} s.")
        # a wide box: what is brighter than the catalogue but steady gets a mention, not a claim
        for c in [c for c in rec['candidates'] if rec.get('wide') and c['kind'] == 'flare' and (c['excess'] or 0) >= 0.2
                  and (c['excess'] or 0) >= 5 * (c['excess_err'] or np.inf)][:2]:
            lines.append(f"The catalogued star at {_deg2hms(c['ra'], c['dec'])} (r = {c['catmag']:.2f}) is "
                         f"{c['excess']:.2f} mag brighter than its catalogue magnitude but constant during our "
                         f"observation; we do not consider it related.")
    elif best['kind'] == 'afterglow':
        lines.append(f"We detect a new source at {_deg2hms(best['ra'], best['dec'])} (J2000, +-0.5\"), "
                     + (f"{best['dist_arcmin']:.1f}' from the trigger position, " if best.get('dist_arcmin') else "")
                     + f"not present in the reference catalogue.")
        lc = best['lightcurve']
        if lc:
            lines.append(f"It was {lc[0][1]:.2f} +- {lc[0][2]:.2f} mag at T0+{lc[0][0]:.0f} s and "
                         f"{lc[-1][1]:.2f} +- {lc[-1][2]:.2f} at T0+{lc[-1][0]:.0f} s"
                         + (f", fading with alpha = {_alpha(best)[0]:.2f} +- {_alpha(best)[1]:.2f}"
                            if best['cls'] == 'FADING' and _alpha(best)[0] is not None else f" ({best['cls'].lower()})") + ".")
        lines.append("We consider it the likely optical afterglow." if best['cls'] in ('FADING', 'TURNING')
                     else "Its nature is to be confirmed.")
    else:
        lines.append(f"A catalogued star at {_deg2hms(best['ra'], best['dec'])} "
                     f"(r = {best['catmag']:.2f}"
                     + (", red" if best.get('red') else "") + (", a dwarf by its proper motion" if best.get('dwarf') else "")
                     + f") is {best['excess']:.2f} mag brighter than quiescence"
                     + (f" and fading (decay at {best['z_decay']:.1f} sigma)" if (best.get('z_decay') or 0) >= 2 else "")
                     + "; a stellar flare is a possible counterpart of the trigger.")
    if best is not None:
        lines += ["", "Photometry (T0 = trigger time):", "  T-T0[s]   mag     err"]
        for t, m, e in best['lightcurve']:
            lines.append(f"  {t:7.0f}  {m:6.2f}  {e if e is not None else float('nan'):.3f}")
    lines += ["", "[closing sentence]", ""]
    return "\n".join(lines)


def _na(v):
    return '' if v is None else f"{v:g}"


def page(rec):
    e = html.escape
    tr = rec['trigger']
    title = f"{rec['site']} {tr['name'] if tr else rec['object']}"
    rows = []
    for i, c in enumerate(rec['candidates']):
        ex = ' '.join(f"+{k}': {v[0]:.2f}" + (f"±{v[1]:.2f}" if v[1] is not None else '')
                      for k, v in c.get('extrap', c.get('excess_extrap', {})).items() if v[0] is not None) if c['cls'] != 'NEW' else 'steady'

        kn = c.get('known')
        cls = e(c['cls']) + (f"<br><span class=mut>known: {e(kn['src'])} {kn['mag']}, {kn['dist_arcsec']}\"</span>"
                             if kn else '')
        if c.get('defect'):
            cls += f"<br><span class=mut>stays on the chip (z {c['defect']['chip_z']}): a pixel defect</span>"
        rows.append(f"<tr><td>{i + 1}</td><td>{cls}</td><td>{c['ra']:.5f} {c['dec']:+.5f}</td>"
                    f"<td>{_mag(c):.2f}</td><td>{e(_trend(c))}</td>"
                    f"<td>{e(ex)}</td><td>{(str(c['dist_arcmin']) + chr(39)) if c.get('dist_arcmin') is not None else ''}</td>"
                    f"<td>{c['score']}</td></tr>")
    srows = ''.join(f"<tr><td>{s['step']}</td><td>{s['bins']}</td><td>{s['bin_seconds']}</td>"
                    f"<td>{s['t_first']:.0f}–{s['t_last']:.0f}</td><td>{s['bin_limit']}</td><td>{s['deep_limit']}</td>"
                    f"<td>{_na(s.get('airmass'))}</td><td>{_na(s.get('fwhm_arcsec'))}</td></tr>"
                    for s in rec['steps'])
    trig = (f"T0 {e(str(tr['t0']))}, {tr['ra']:.4f} {tr['dec']:+.4f}, error box {tr['errorbox_arcmin']}'"
            if tr else "no trigger information")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="60"><title>{e(title)}</title>
<style>
:root {{ --bg:#fff; --fg:#222; --mut:#666; --line:#ddd; --acc:#b3261e; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#16181c; --fg:#e8e8e8; --mut:#999; --line:#333; --acc:#ff8a80; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.4 system-ui,sans-serif; margin:0 auto; max-width:1000px; padding:16px; }}
h1 {{ font-size:1.4em; margin:0 0 4px; }} .mut {{ color:var(--mut); }}
.status {{ font-weight:600; color:var(--acc); }}
table {{ border-collapse:collapse; width:100%; margin:8px 0 16px; font-size:0.92em; }}
td,th {{ border-bottom:1px solid var(--line); padding:4px 6px; text-align:left; vertical-align:top; }}
.imgs {{ display:flex; flex-wrap:wrap; gap:12px; }} .imgs img {{ max-width:100%; height:auto; background:#fff; }}
pre {{ white-space:pre-wrap; border:1px solid var(--line); padding:8px; font-size:0.85em; overflow-x:auto; }}
.wrap {{ overflow-x:auto; }}
</style></head><body>
<h1>{e(title)}</h1>
<div class="mut">{e(rec['object'])} · {trig} · updated {e(rec['updated'])} UT · step {rec['step']}</div>
<p class="status">{e({'nothing': 'Nothing new near the trigger.' if rec['candidates'] else 'Nothing new found.', 'candidate': 'Candidate(s), unconfirmed.', 'likely': 'Likely counterpart.'}[rec['status']])}</p>
{''.join(f'<p class="status">Caution: {e(c)}.</p>' for c in rec.get('caution', []))}
<div class="imgs"><img src="lc.png" alt="light curve"><img src="chart.png" alt="finding chart"></div>
<h2>Candidates</h2><div class="wrap"><table><tr><th>#</th><th>class</th><th>RA Dec (J2000)</th><th>mag</th><th>trend</th><th>expected</th><th>dist</th><th>S</th></tr>{''.join(rows) or '<tr><td colspan=8>none</td></tr>'}</table></div>
{'<h2>Cutouts</h2><p class="mut">' + str(CUT) + "' square, the trigger position and every candidate: the deep image, then single frames across the step (each scaled on its own; north is roughly up).</p><div class=wrap><img src=cutouts.png alt=cutouts style='max-width:none'></div>" if rec.get('cutouts') else ''}
<h2>Steps</h2><div class="wrap"><table><tr><th>step</th><th>bins</th><th>bin [s]</th><th>T-T0 [s]</th><th>bin limit</th><th>deep limit</th><th>airmass</th><th>FWHM ["]</th></tr>{srows}</table></div>
<h2>Draft GCN</h2><pre>{e(gcn_text(rec))}</pre>
<p class="mut"><a href="report.json">report.json</a> · asarina-kinseries / pyrt transients</p>
</body></html>
"""


# ---------------------------------------------------------------- publish / notify

def _on_report_host():
    return socket.getfqdn().split('.')[0] == REPORT_HOST.split('.')[0] or socket.gethostname() == REPORT_HOST.split('.')[0]


def publish(obs, notify_args=()):
    rep = os.path.join(obs, 'report')
    rec = json.load(open(os.path.join(rep, 'report.json')))
    name = rec['name']
    if _on_report_host():
        dest = os.path.join(os.path.expanduser(REPORT_ROOT), name)
        os.makedirs(dest, exist_ok=True)
        for f in os.listdir(rep):
            shutil.copy2(os.path.join(rep, f), os.path.join(dest, f + '.tmp'))
            os.replace(os.path.join(dest, f + '.tmp'), os.path.join(dest, f))
        cmd = [sys.executable, '-m', 'asarina.pipeline.trigreport', 'notify', dest] + list(notify_args)
        subprocess.run(cmd, timeout=60)
    else:
        # the upload key is restricted on lascaux to rrsync -wo ~/public_html/ (write only, no
        # shell): paths are relative to public_html, and the notification is lascaux's own
        # business (trigreport scan, from cron, picks up the new report.json)
        key = os.path.expanduser(SSH_KEY)
        if os.path.exists(key):   # the restricted key: rrsync roots the path at public_html
            e = f'ssh -o BatchMode=yes -o ConnectTimeout=10 -i {key} -o IdentitiesOnly=yes'
            rel = REPORT_ROOT.replace('~/public_html/', '', 1).strip('/')
        else:                     # an ordinary login (by hand): the path from home
            e = 'ssh -o BatchMode=yes -o ConnectTimeout=10'
            rel = REPORT_ROOT.rstrip('/')
        # no -t: the files get their arrival time on lascaux, which the scan goes by
        subprocess.run(['rsync', '-rlD', '--mkpath', '--chmod=Fa+r,Da+rx', '-e', e, rep + '/',
                        f'{REPORT_HOST}:{rel}/{name}/'], timeout=120, check=True)
    return f"{REPORT_URL}/{name}/"


def _mag(c):
    """The magnitude to quote: the latest, but the mean of a steady (NEW) source."""
    if c['cls'] == 'NEW':
        return c.get('mag_mean') or c.get('mag_last') or 0.0
    return c.get('mag_last') or c.get('mag_mean') or 0.0


def _key(c):
    return f"{c['ra']:.4f} {c['dec']:+.4f}"


def message(rec, previous):
    """The Telegram text, or None when nothing changed that a human must know at once:
    the best candidate is new, or a known one changed its class."""
    best = rec['best']
    seen = previous.get('seen', {})
    if best is None:
        return None
    # a message wakes people up: only a light curve that says something (an ordered afterglow,
    # a decaying flare) or a flare-star candidate inside the error box; a steady NEW, an ERRATIC
    # or a one-bin SPIKE stay on the page
    inbox = best.get('dist_box') is not None and best['dist_box'] <= 1
    if not (best['cls'] in ('FADING', 'RISING', 'TURNING', 'FLARE') or (best['cls'] == 'FLARE?' and inbox)):
        return None
    if rec.get('wide') and not _variable(best):
        return None   # a wide (GBM) box: only what varied while we watched
    k = _key(best)
    if k in seen and (seen[k] == best['cls'] or (seen[k], best['cls']) == ('FLARE', 'FLARE?')):
        return None   # nothing new (a flare whose decay fit weakened for a step is still the flare)
    what = 'changed to' if k in seen else 'new'
    tr = rec['trigger']
    st = rec['steps'][-1]
    t = tr['name'] if tr else rec['object']
    ex = ''
    if best['kind'] == 'afterglow' and best['cls'] != 'NEW' and best['extrap'].get('15', [None])[0] is not None:
        ex = f", expected {best['extrap']['15'][0]:.1f}±{best['extrap']['15'][1]:.1f} at +15 min"
    mag = _mag(best)
    return (f"{rec['site']} {t}: {what} {best['cls']} at {best['ra']:.5f} {best['dec']:+.5f}, "
            f"{mag:.2f} mag ({_trend(best)}){ex}"
            + (f", {best['dist_arcmin']}' from the trigger" if best.get('dist_arcmin') is not None else '')
            + f". Step {rec['step']}, T0+{st['t_last']:.0f} s, deep limit {st['deep_limit']}. "
            + ''.join(f"Caution: {c}. " for c in rec.get('caution', []))
            + f"{REPORT_URL}/{rec['name']}/")


def send_telegram(text, chat_key=TELEGRAM_CHAT):
    import urllib.parse
    import urllib.request
    cfg = json.load(open(os.path.expanduser(TELEGRAM_CONFIG)))['telegram']
    data = urllib.parse.urlencode({'chat_id': cfg[chat_key], 'text': text[:4000],
                                   'disable_web_page_preview': 'true'}).encode()
    with urllib.request.urlopen(f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage", data, timeout=20) as r:
        return json.load(r).get('ok', False)


def notify(pagedir, dry_run=False, chat_key=TELEGRAM_CHAT):
    pagedir = os.path.expanduser(pagedir)
    rec = json.load(open(os.path.join(pagedir, 'report.json')))
    stf = os.path.join(pagedir, 'notified.json')
    prev = json.load(open(stf)) if os.path.exists(stf) else {}
    text = message(rec, prev)
    if text is None:
        return None
    if dry_run:
        print(text)
        return text
    try:
        ok = send_telegram(text, chat_key)
    except Exception as e:
        print(f"trigreport: telegram: {e}", file=sys.stderr)
        ok = False
    if ok:
        prev.setdefault('seen', {})[_key(rec['best'])] = rec['best']['cls']
        prev['last'] = dict(time=rec['updated'], step=rec['step'], text=text)
        with open(stf + '.tmp', 'w') as f:
            json.dump(prev, f, indent=1)
        os.replace(stf + '.tmp', stf)
    return text


def scan(root, dry_run=False, chat_key=TELEGRAM_CHAT):
    """On lascaux, from cron: notify for every page whose report.json changed since the last
    scan (the uploads from the telescopes cannot run anything here)."""
    root = os.path.expanduser(root)
    os.makedirs(root, exist_ok=True)   # before the first trigger the uploads have not made it yet
    stamp = os.path.join(root, '.scan.stamp')
    last = os.path.getmtime(stamp) if os.path.exists(stamp) else 0.0
    now = time.time()
    for rj in sorted(glob.glob(os.path.join(root, '*', 'report.json'))):
        if os.path.getmtime(rj) > last:
            try:
                notify(os.path.dirname(rj), dry_run, chat_key)
            except Exception as e:
                print(f"trigreport: {rj}: {e}", file=sys.stderr)
    if not dry_run:
        with open(stamp, 'w'):
            pass
        os.utime(stamp, (now, now))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=('build', 'publish', 'notify', 'update', 'scan'))
    ap.add_argument('dir')
    ap.add_argument('--dry-run', action='store_true', help='notify: print the message instead of sending it')
    ap.add_argument('--chat', default=TELEGRAM_CHAT, help=f"key of the chat in {TELEGRAM_CONFIG} 'telegram' (default {TELEGRAM_CHAT})")
    a = ap.parse_args()
    nargs = (['--dry-run'] if a.dry_run else []) + ['--chat', a.chat]
    if a.action == 'build':
        print(build(os.path.abspath(a.dir)))
    elif a.action == 'publish':
        print(publish(os.path.abspath(a.dir), nargs))
    elif a.action == 'notify':
        notify(a.dir, a.dry_run, a.chat)
    elif a.action == 'scan':
        scan(a.dir, a.dry_run, a.chat)
    else:
        if build(os.path.abspath(a.dir)):
            print(publish(os.path.abspath(a.dir), nargs))


if __name__ == '__main__':
    main()
