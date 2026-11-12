#!/usr/bin/env python3
"""
kinphot - dophot photometry of the individual frames of an Andor kinetic series.

The D50/C0 kinetic-series exposure is stored by RTS2 as an RA.fits holding the
*mean* of all sub-frames, while the sub-frames themselves are dumped as raw
uint16 into KINFILE (/var/tmp/kin-*.out, later moved to /images/kinmode).
The mean frame goes through asarina-imgproc as usual; its ECSV holds the final
(ZPN) astrometric solution and the photometric response.

For every sub-frame (or every sum of --bin consecutive sub-frames) this script
  1. dark/flat corrects it with the same master frames asarina picks for the
     mean frame (dark of the sub-frame exposure, subtracted per sub-frame),
  2. writes it with the raw header, the sub-frame timing and the WCS of the
     reference ECSV (no solving, the telescope does not move noticeably),
  3. runs pyrt-phcat -> pyrt-cat2det -> pyrt-dophot (without -a, i.e. no
     astrometric refit); by default the spatial photometric response of the
     reference is held fixed, so only the zeropoint is fitted per frame,
  4. collects the per-frame ECSVs and a summary table.

With --forced the objects are detected once, on the mean of the used sub-frames
(the deep image), and every sub-frame/bin is measured at those positions
(pyrt-phcat -D, SExtractor dual-image mode): a faint source keeps its whole
light curve instead of dropping out of the bins where it is below the threshold.

Usage (on d50):
  kinphot.py /images/2026/20261106/C0/53486/20261106214336-886-RA.fits
  kinphot.py ~/phdb-rt/26/1106/20261106214336-886-N-000-df.ecsv
  kinphot.py --bin 10 -j 4 RAW.fits
  kinphot.py --bin 8 --forced RAW.fits     # 96 x 1.25 s -> 12 points, forced photometry
"""

import argparse
import concurrent.futures as cf
import glob
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

log = logging.getLogger("kinphot")

# WCS keywords taken from the reference ECSV (and wiped from the raw header,
# which carries the rough TAN solution written back by the realtime pipeline)
WCS_RE = re.compile(r'^(WCSAXES|EQUINOX|EPOCH|RADESYSa?|RADECSYS|LONPOLE|LATPOLE|'
                    r'CTYPE\d|CUNIT\d|CRVAL\d|CRPIX\d|CDELT\d|CROTA\d|CD\d_\d|PC\d_\d|'
                    r'PV\d_\d+|[AB]P?_\d+_\d+|[AB]P?_ORDER|[AB]P?_DMAX)$')

# time keywords that would be stale for a sub-frame (cat2det rebuilds what it needs)
STALE_TIME = ('MJD-OBS', 'MJD', 'JD_START', 'JD_HELIO', 'BJD', 'DATE-END')

IMAGES_ROOT = '/images'
REF_ROOTS = ('~/phdb-rt', '~/ecsv', '~/phdb')


# ---------------------------------------------------------------- locating

def stem_of(path):
    """20261106214336-886 from any of the RA/df/ecsv names."""
    m = re.match(r'(\d{14}-\d{3})', os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot get the frame stem from {path}")
    return m.group(1)


def find_raw(stem):
    year = stem[:4]
    day = datetime.strptime(stem[:8], '%Y%m%d')
    nights = [day.strftime('%Y%m%d'),
              datetime.fromordinal(day.toordinal() - 1).strftime('%Y%m%d')]
    for night in nights:
        hits = glob.glob(f"{IMAGES_ROOT}/{year}/{night}/*/*/{stem}-RA.fits")
        if hits:
            return hits[0]
    hits = glob.glob(f"{IMAGES_ROOT}/{year}/*/*/*/{stem}-RA.fits")
    return hits[0] if hits else None


def find_ref(stem):
    yy = stem[2:4]
    for root in REF_ROOTS:
        hits = sorted(glob.glob(os.path.expanduser(f"{root}/{yy}/*/{stem}-*.ecsv")))
        hits += sorted(glob.glob(os.path.expanduser(f"{root}/{yy}*/{stem}-*.ecsv")))
        if hits:
            return hits[0]
    return None


def find_kinfile(header):
    kinfile = header['KINFILE']
    for cand in (kinfile, kinfile.replace('/var/tmp', '/images/kinmode')):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Kinetic data {kinfile} not found (nor in /images/kinmode)")


# ---------------------------------------------------------------- calibration

def load_calibration(raw_path, header, expt):
    """Dark (for the sub-frame exposure expt) and flat, chosen as asarina does."""
    from asarina.pipeline.image import ImageProcessor, CAMERA_CROPS
    from pyrt.utils.chip_id import load_chip_id

    proc = ImageProcessor()
    proc.load_calibration_frames([raw_path])
    chip = load_chip_id(header)
    binx = header.get('BINX', 1)
    shape = (header['NAXIS2'], header['NAXIS1'])
    window = proc._parse_ccdsec(header)
    origin = (window[0].start, window[1].start) if window is not None else None
    # asarina picks by shape and window position: a full frame (cropped below) or a dark
    # of exactly this window
    dark_path = proc.find_best_dark(chip, header['CCD_TEMP'], binx, expt, shape=shape, origin=origin)
    flat_path = proc.find_flat(chip, header['FILTER'], binx)
    if dark_path is None or flat_path is None:
        raise RuntimeError(f"No dark ({dark_path}) or flat ({flat_path}) for {raw_path}")
    log.info(f"dark: {dark_path}")
    log.info(f"flat: {flat_path}")
    dark = fits.getdata(dark_path).astype(np.float32)
    flat = fits.getdata(flat_path).astype(np.float32)

    if window is not None:
        if dark.shape != shape:   # a window-sized dark needs no crop
            dark = dark[window]
        flat = flat[window]
    crop = None if window is not None else CAMERA_CROPS.get(chip)
    return dark, flat, crop


# ---------------------------------------------------------------- header

def ref_wcs_and_response(ref_path):
    meta = Table.read(ref_path, format='ascii.ecsv').meta
    wcs = {k: v for k, v in meta.items() if WCS_RE.match(k)}
    if 'CTYPE1' not in wcs:
        raise RuntimeError(f"No WCS in {ref_path}")
    resp = meta.get('RESPONSE', '')
    return wcs, resp, meta


def fixed_terms(response):
    """'Z=23.6,PY=0.11,PX=-0.002' -> '#PY=0.11,#PX=-0.002' (zeropoint left free)."""
    terms = []
    for item in response.split(','):
        if '=' not in item:
            continue
        name, val = item.split('=', 1)
        name = name.strip()
        if name in ('Z', ''):
            continue
        terms.append(f"#{name}={val.strip()}")
    return ','.join(terms)


def subframe_header(raw_hdr, wcs, crop, t_start, exptime, idx, nbin, ref_path, raw_path):
    h = raw_hdr.copy()
    for k in list(h.keys()):
        if WCS_RE.match(k) or k in STALE_TIME:
            del h[k]
    for k in ('BZERO', 'BSCALE'):
        h.remove(k, ignore_missing=True)
    for k, v in wcs.items():
        h[k] = v
    if crop is not None:
        # the reference WCS is for the cropped frame already: nothing to shift
        pass

    ctime = int(np.floor(t_start))
    usec = int(round((t_start - ctime) * 1e6))
    if usec >= 1000000:
        ctime, usec = ctime + 1, usec - 1000000
    h['CTIME'] = (ctime, 'sub-frame exposure start (unix)')
    h['USEC'] = (usec, 'sub-frame exposure start micro seconds')
    h['DATE-OBS'] = (datetime.fromtimestamp(t_start, tz=timezone.utc)
                     .strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3], 'sub-frame start')
    h['JD'] = (2440587.5 + t_start / 86400.0, 'sub-frame start')
    h['EXPTIME'] = (exptime, 'sub-frame exposure (sum of KINBIN frames)')
    h['EXPOSURE'] = (exptime, 'sub-frame exposure (sum of KINBIN frames)')
    h['KINIDX'] = (idx, 'first kinetic sub-frame index (0-based)')
    h['KINBIN'] = (nbin, 'number of sub-frames summed')
    h['KINRAW'] = (os.path.basename(raw_path), 'kinetic series mean frame')
    h['KINREF'] = (os.path.basename(ref_path), 'WCS/response taken from')
    return h


# ---------------------------------------------------------------- per frame

def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def process_frame(job):
    """Run phcat, cat2det, dophot on one prepared sub-frame. Returns a dict."""
    tmp, name, outdir, dophot_cmd, phcat_cmd, keep = job
    base = name[:-5]
    res = {'name': base, 'ok': False, 'line': '', 'error': ''}
    t = time.time()
    try:
        for step, cmd in (('phcat', phcat_cmd + [name]),
                          ('cat2det', ['pyrt-cat2det', name]),
                          ('dophot', dophot_cmd + [base + '.det'])):
            r = run(cmd, tmp)
            if r.returncode != 0 or (step == 'dophot' and not os.path.exists(f"{tmp}/{base}.ecsv")):
                res['error'] = f"{step} failed: " + (r.stderr or r.stdout or '').strip()[-600:]
                if keep:
                    with open(f"{tmp}/{step}.log", 'w') as f:
                        f.write(r.stdout + '\n' + r.stderr)
                return res
            if step == 'dophot':
                lines = [l for l in r.stdout.splitlines() if l.startswith(base)]
                res['line'] = lines[-1] if lines else ''
                if keep:
                    with open(f"{tmp}/dophot.log", 'w') as f:
                        f.write(r.stdout + '\n' + r.stderr)
        shutil.copy(f"{tmp}/{base}.ecsv", f"{outdir}/{base}.ecsv")
        res['ok'] = True
        return res
    finally:
        res['time'] = time.time() - t
        if keep:
            for f in glob.glob(f"{tmp}/{base}*"):
                if f.endswith(('.fits', '.det', '.cat')):
                    shutil.copy(f, outdir)
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- summary

SUMMARY_KEYS = ('KINIDX', 'KINBIN', 'CTIME', 'USEC', 'JD', 'EXPTIME', 'FWHM', 'MEDIAN', 'BGSIGMA',
                'IDNUM', 'MAGZERO', 'DMAGZERO', 'MAGLIMIT', 'WSSRNDF', 'PHQUAL')


def _num(x):
    try:
        v = float(x)
        return np.nan if v >= 99.9 else v
    except (TypeError, ValueError):
        return np.nan


def write_summary(outdir, results, path):
    rows = []
    for r in results:
        if not r['ok']:
            continue
        meta = Table.read(f"{outdir}/{r['name']}.ecsv", format='ascii.ecsv').meta
        row = {k: meta.get(k, np.nan) for k in SUMMARY_KEYS}
        row['FILE'] = r['name']
        # target magnitude as reported by dophot (cols: ... mag magerr target obsid status)
        f = r['line'].split()
        row['TMAG'] = _num(f[11]) if len(f) > 12 else np.nan
        row['TMAGERR'] = _num(f[12]) if len(f) > 12 else np.nan
        for k in SUMMARY_KEYS:
            if k != 'PHQUAL' and not isinstance(row[k], (int, float, np.number)):
                row[k] = _num(row[k])
        rows.append(row)
    if not rows:
        return None
    cols = ['FILE'] + list(SUMMARY_KEYS) + ['TMAG', 'TMAGERR']
    t = Table(rows=[[row[c] for c in cols] for row in rows], names=cols)
    t.sort('KINIDX')
    t.write(path, format='ascii.ecsv', overwrite=True)
    return t


# ---------------------------------------------------------------- main

def parse_frames(spec, n):
    if not spec:
        return 0, n
    a, _, b = spec.partition(':')
    return int(a or 0), min(int(b) if b else n, n)


def main():
    os.environ.setdefault('TERM', 'xterm')   # IRAF cl (in pyrt-phcat) dies without it
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+', help='kinetic RA.fits (raw mean frame) or its -df.ecsv')
    ap.add_argument('--raw', help='raw RA.fits (when not found automatically)')
    ap.add_argument('--ref', help='reference ECSV with WCS (when not found automatically)')
    ap.add_argument('-b', '--bin', type=int, default=1, help='sum this many consecutive sub-frames (default 1)')
    ap.add_argument('--frames', help='sub-frame range a:b (0-based, python-like), default all')
    ap.add_argument('--t-offset', type=float, default=0.0,
                    help='shift of the first dumped sub-frame start w.r.t. CTIME+USEC [s] '
                         '(the dump has KINNUM-1 frames; which one is missing is not known yet)')
    ap.add_argument('--sub-exptime', type=float, default=None,
                    help='sub-frame exposure [s] (default KINCYCLE); also picks the dark')
    ap.add_argument('-o', '--outdir', help='output directory (default ./<stem>-kin[-bN])')
    ap.add_argument('-j', '--jobs', type=int, default=2, help='parallel frames (default 2; d50 is short of RAM)')
    ap.add_argument('-U', '--terms', default=None,
                    help="dophot -U terms; default: the reference RESPONSE held fixed ('#PX=..,...'), "
                         "use e.g. '&.p1' to fit, or '' for a bare zeropoint")
    ap.add_argument('--filter-terms', default=None,
                    help="fixed colour terms appended to -U (default: filter_terms_<filter> "
                         "of the asarina config; '' for none)")
    ap.add_argument('-i', '--idlimit', type=int, default=2, help='dophot idlimit (default 2)')
    ap.add_argument('--phcat-fwhm', default='auto',
                    help="pyrt-phcat -f: 'auto' (phcat estimates), 'ref' (FWHM of the reference), or a number")
    ap.add_argument('-F', '--forced', action='store_true',
                    help='forced photometry: detect on the mean of the used sub-frames, measure each bin there')
    ap.add_argument('--dophot-args', default='', help='extra arguments for pyrt-dophot (one string)')
    ap.add_argument('-k', '--keep', action='store_true', help='keep the sub-frame FITS/.det/.cat and logs')
    ap.add_argument('-f', '--force', action='store_true', help='redo frames whose ECSV exists')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    for name in ('asarina', 'pyrt'):
        logging.getLogger(name).setLevel(logging.WARNING if not args.verbose else logging.DEBUG)

    rc = 0
    for arg in args.files:
        try:
            if not do_series(arg, args):
                rc = 1
        except Exception as e:
            log.error(f"{arg}: {e}")
            if args.verbose:
                raise
            rc = 1
    return rc


def do_series(arg, args):
    stem = stem_of(arg)
    raw_path = args.raw or (arg if arg.endswith('-RA.fits') else find_raw(stem))
    ref_path = args.ref or (arg if arg.endswith('.ecsv') else find_ref(stem))
    if not raw_path:
        raise FileNotFoundError(f"raw RA.fits for {stem} not found, use --raw")
    if not ref_path:
        raise FileNotFoundError(f"reference ECSV for {stem} not found, use --ref")
    log.info(f"{stem}: raw {raw_path}")
    log.info(f"{stem}: ref {ref_path}")

    raw_hdr = fits.getheader(raw_path)
    if 'KINFILE' not in raw_hdr:
        raise RuntimeError(f"{raw_path} is not a kinetic series (no KINFILE)")
    kinfile = find_kinfile(raw_hdr)
    nx, ny = raw_hdr['NAXIS1'], raw_hdr['NAXIS2']
    kin = np.memmap(kinfile, np.uint16, 'r')
    nfr = kin.size // (nx * ny)
    kin = kin[:nfr * nx * ny].reshape(nfr, ny, nx)
    cycle = float(raw_hdr['KINCYCLE'])
    # EXPTIME in the header is the nominal setting (0.1s); the CCD integrates
    # for the whole kinetic cycle, so a sub-frame is a KINCYCLE-long exposure
    expt1 = args.sub_exptime or cycle
    log.info(f"{stem}: {kinfile}: {nfr} frames (KINNUM={raw_hdr.get('KINNUM')}), "
             f"{expt1:.4f}s exposures every {cycle:.4f}s")
    if nfr != raw_hdr.get('KINNUM'):
        log.warning(f"{stem}: dump holds {nfr} frames, KINNUM={raw_hdr.get('KINNUM')}; "
                    f"timing assumes the dump starts at CTIME+USEC{args.t_offset:+.3f}s")

    dark, flat, crop = load_calibration(raw_path, raw_hdr, expt1)
    wcs, response, ref_meta = ref_wcs_and_response(ref_path)

    # sanity: the mean of the dump must be the RA frame (RTS2 stores the integer mean)
    rawmean = fits.getdata(raw_path).astype(np.float32)
    stack = (rawmean - dark) / flat
    if crop is not None:
        stack = stack[crop]
    log.info(f"{stem}: calibrated mean frame median {np.median(stack):.2f} "
             f"(reference MEDIAN {ref_meta.get('MEDIAN')})")

    terms = args.terms if args.terms is not None else fixed_terms(response)
    # fixed per-filter (colour) terms, as asarina-imgproc appends them:
    # filter_terms_<filter> from /etc/asarina/config (+ ~/.config), camera section
    fterms = args.filter_terms
    if fterms is None:
        try:
            from asarina.config import load_config, filter_terms
            cfg = load_config(None, raw_hdr.get('CCD_NAME'))
            fterms = filter_terms(cfg).get(str(raw_hdr.get('FILTER', '')).lower(), '')
        except Exception as e:
            log.warning(f"{stem}: cannot read asarina config: {e}")
            fterms = ''
        if not fterms:
            log.warning(f"{stem}: no filter_terms_{str(raw_hdr.get('FILTER', '')).lower()} "
                        f"in the asarina config, no fixed colour terms")
    fterms = (fterms or '').strip().strip('"\'')   # a quoted config value keeps its quotes
    if fterms:
        # the reference RESPONSE may hold the same terms (it was fitted with
        # them): the config value wins
        names = {t.lstrip('#&@*').split('=')[0] for t in fterms.split(',')}
        terms = ','.join([t for t in terms.split(',')
                          if t and t.lstrip('#&@*').split('=')[0] not in names] + [fterms])
    log.info(f"{stem}: dophot terms: '{terms}'")
    dophot_cmd = ['pyrt-dophot', '-m0.5', '-S0', f'-i{args.idlimit}', '--max-stars', '1000']
    if terms:
        dophot_cmd += ['-U', terms]
    dophot_cmd += args.dophot_args.split()
    phcat_cmd = ['pyrt-phcat']
    if args.phcat_fwhm == 'ref':
        phcat_cmd += ['-f', str(ref_meta['FWHM'])]
    elif args.phcat_fwhm != 'auto':
        phcat_cmd += ['-f', args.phcat_fwhm]

    first, last = parse_frames(args.frames, nfr)
    nbin = max(1, args.bin)
    outdir = args.outdir or f"{stem}-kin" + (f"-b{nbin}" if nbin > 1 else '') + ('-forced' if args.forced else '')
    os.makedirs(outdir, exist_ok=True)
    workroot = os.path.abspath(tempfile.mkdtemp(prefix='kinphot-', dir=outdir))

    t0 = raw_hdr['CTIME'] + raw_hdr.get('USEC', 0) / 1e6 + args.t_offset
    if args.forced:
        # the deep (detection) image: mean of the sub-frames that are used, same grid as the bins
        if (first, last) == (0, nfr):
            deep = stack
        else:
            deep = np.zeros_like(dark)
            for i in range(first, last):
                deep += kin[i]
            deep = (deep / (last - first) - dark) / flat
            if crop is not None:
                deep = deep[crop]
        detect = f"{workroot}/{stem}-deep.fits"
        dhdr = subframe_header(raw_hdr, wcs, crop, t0 + first * cycle, expt1, first, last - first, ref_path, raw_path)
        fits.PrimaryHDU(data=deep.astype(np.float32), header=dhdr).writeto(detect)
        phcat_cmd = phcat_cmd + ['-D', detect]
        log.info(f"{stem}: forced photometry, detection on the mean of sub-frames {first}:{last}")

    fltr = raw_hdr.get('FILTER', 'UNK')

    def jobs():
        for idx in range(first, last - nbin + 1, nbin):
            t_start = t0 + idx * cycle
            name = (datetime.fromtimestamp(t_start, tz=timezone.utc).strftime('%Y%m%d%H%M%S')
                    + f"-{int((t_start % 1) * 1000):03d}-{fltr}-k{idx:03d}"
                    + (f"b{nbin}" if nbin > 1 else '') + '.fits')
            if not args.force and os.path.exists(f"{outdir}/{name[:-5]}.ecsv"):
                log.info(f"{name[:-5]}: done already")
                yield None, name
                continue
            data = (kin[idx:idx + nbin].astype(np.float32) - dark).sum(axis=0) / flat
            if crop is not None:
                data = data[crop]
            hdr = subframe_header(raw_hdr, wcs, crop, t_start, expt1 * nbin, idx, nbin,
                                  ref_path, raw_path)
            tmp = tempfile.mkdtemp(prefix=f"k{idx:03d}-", dir=workroot)
            fits.PrimaryHDU(data=data.astype(np.float32), header=hdr).writeto(f"{tmp}/{name}")
            yield (tmp, name, outdir, dophot_cmd, phcat_cmd, args.keep), name

    results = []
    tstart = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {}
        for job, name in jobs():
            if job is None:
                results.append({'name': name[:-5], 'ok': True, 'line': '', 'error': ''})
                continue
            futs[ex.submit(process_frame, job)] = name
            # keep at most 2*jobs frames prepared at a time (memory, disk)
            while len([f for f in futs if not f.done()]) >= 2 * args.jobs:
                cf.wait(futs, return_when=cf.FIRST_COMPLETED)
        for fut in cf.as_completed(futs):
            r = fut.result()
            results.append(r)
            if r['ok']:
                log.info(f"{r['name']}: ok {r['time']:.1f}s  {' '.join(r['line'].split()[5:12])}")
            else:
                log.warning(f"{r['name']}: {r['error']}")
    shutil.rmtree(workroot, ignore_errors=True)

    nok = sum(r['ok'] for r in results)
    summ = f"{outdir}/{stem}.kinphot.ecsv"
    t = write_summary(outdir, results, summ)
    log.info(f"{stem}: {nok}/{len(results)} frames photometered in {time.time()-tstart:.0f}s"
             + (f", summary {summ}" if t is not None else ''))
    return nok > 0


if __name__ == '__main__':
    sys.exit(main())
