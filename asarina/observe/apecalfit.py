#!/usr/bin/env python3
"""
Aperture growth curve parameter (APE) calibration fitter.

APE is the single free parameter in the exposure-time model.  It sets the
break magnitude — where the photon-noise regime meets the sky-noise regime:

    break_magnitude = −2.5·log10(APE · π/4 · FWHM² · (BGSIGMA·GAIN)²) + ZERO

A larger APE means fainter break (more sky-noise limited for a given FWHM).

APE and default_fwhm are co-calibrated
---------------------------------------
The runtime code evaluates the break magnitude using the FWHM that the user
passes on the command line (default: config default_fwhm).  Therefore APE
must be calibrated with the SAME FWHM convention:

  • The fitter reads the image-level FWHM from each catalog's FITS header
    (the FWHM keyword, measured by the pipeline for the whole frame).
  • This is the same value used when planning observations without explicit
    --fwhm on the command line.
  • After fitting, set default_fwhm = median header FWHM of the calibration
    data (reported by the fitter).

If you calibrate with data that has typical FWHM = 3 px and set
default_fwhm = 3.0, the model is self-consistent.  Do not mix datasets
with very different FWHM unless you intend to account for the spread.

Input data
----------
Per-image ECSV catalogs produced by the photometric pipeline (*-df.ecsv or
*-dft.ecsv files).  The fitter reads per-image conditions from the FITS
header embedded in each ECSV's metadata section, and per-star measurements
from the table columns.

Required FITS header keywords (in the ECSV metadata):
    FWHM      -- typical PSF FWHM for this image [pixels]
    BGSIGMA   -- background noise per pixel for this image [ADU]
    GAIN      -- CCD gain [e-/ADU]
    MAGZERO   -- photometric zeropoint for this exposure in calculator-
                 internal units (= absolute_zp_for_exptime − ZERO, where ZERO = 10)

Optional header keywords:
    CCD_NAME  -- camera identifier; used by --camera to filter to one camera

Required table columns:
    MAG_CALIB    -- calibrated star magnitude
    MAGERR_CALIB -- measured magnitude error
    FLAGS        -- SExtractor extraction flags (0 = clean)

The formula uses  mag_rel = MAG_CALIB − MAGZERO + ZERO  (ZERO = 10).

Usage (CLI)
-----------
    asarina-observe-apecalfit /data/2501/*-df.ecsv --camera C2
    asarina-observe-apecalfit /data/2501/*-df.ecsv --camera C2 --output residuals.png

Usage (library)
---------------
    from asarina.observe.apecalfit import fit_ape
    result = fit_ape(['/path/to/a.ecsv', '/path/to/b.ecsv'])
    print(result['ape'])            # → 6.146
    print(result['median_fwhm'])    # → 3.1  — set this as default_fwhm
    print(result['rms_log'])        # → 0.047

Workflow for a new telescope
-----------------------------
1. Reduce a set of calibration frames through the photometric pipeline.
2. Collect the resulting per-image *-df.ecsv / *-dft.ecsv output files.
3. Run this script:
       asarina-observe-apecalfit /data/*.ecsv --camera C2
4. Also update default_fwhm in the config to match the reported median_fwhm.
5. Re-train the background model:
       asarina-observe-train ~/phdb/stat/*.ecsv --camera C2
"""

import argparse
import glob
import sys
import warnings
from typing import List, Optional, Union

import numpy as np
from astropy.table import Table, vstack
from scipy.optimize import minimize_scalar

# ---------------------------------------------------------------------------
# Conventional offset in the zeropoint definition  (must match expcalc.py)
# ---------------------------------------------------------------------------
ZERO = 10

# ---------------------------------------------------------------------------
# Quality cuts
# ---------------------------------------------------------------------------
MIN_MAGERR = 0.003   # below → saturation or systematic-dominated
MAX_MAGERR = 0.300   # above → too noisy, model scatter exceeds fit signal
MAX_FLAGS  = 0       # SExtractor FLAGS: 0 = clean
MIN_FWHM   = 1.0     # image-level header FWHM lower bound [pixels]
MAX_FWHM   = 15.0    # image-level header FWHM upper bound [pixels]


# ---------------------------------------------------------------------------
# Model (mirrors expcalc.py ExposureCalculator, parameterised by APE)
# ---------------------------------------------------------------------------

def _sbl(A, B, N, x):
    return A * x + (B - A) * (abs(N) * np.sqrt(1.0 + x * x / (N * N)) + x) / 2.0


def _log_magerror(mag_rel, bgsigma, fwhm, ape, gain):
    bm = -2.5 * np.log10(ape * np.pi / 4 * fwhm**2 * (bgsigma * gain)**2) + ZERO
    return _sbl(0.2, 0.4, 2.5, mag_rel - bm) + 0.2 * bm - 2


# ---------------------------------------------------------------------------
# Data loader — reads individual per-image ECSV files
# ---------------------------------------------------------------------------

def _load_one(path: str) -> Optional[Table]:
    """Load one ECSV catalog; attach per-image header values as columns."""
    try:
        t = Table.read(path, format='ascii.ecsv')
    except Exception as e:
        print(f"  Warning: could not read {path}: {e}", file=sys.stderr)
        return None

    required_cols = ('MAG_CALIB', 'MAGERR_CALIB')
    if not all(c in t.colnames for c in required_cols):
        print(f"  Warning: {path} missing required columns, skipping.", file=sys.stderr)
        return None

    required_meta = ('FWHM', 'BGSIGMA', 'GAIN', 'MAGZERO')
    missing = [k for k in required_meta if k not in t.meta]
    if missing:
        print(f"  Warning: {path} missing header keys {missing}, skipping.", file=sys.stderr)
        return None

    for k in required_meta:
        t[k] = float(t.meta[k])

    t['CCD_NAME'] = str(t.meta.get('CCD_NAME', ''))

    return t


def load_catalogs(
    paths: Union[str, List[str]],
    camera: Optional[str] = None,
    max_flags: int = MAX_FLAGS,
    min_magerr: float = MIN_MAGERR,
    max_magerr: float = MAX_MAGERR,
    min_fwhm: float = MIN_FWHM,
    max_fwhm: float = MAX_FWHM,
    verbose: bool = True,
) -> dict:
    """
    Load and merge per-image ECSV catalogs; apply quality cuts.

    Parameters
    ----------
    paths  : path string, glob pattern, or list of paths
    camera : if given, keep only catalogs whose CCD_NAME header matches

    Returns a dict of equal-length numpy arrays:
        mag_rel, bgsigma, fwhm, gain, log_magerr_obs, n_total, n_good
    """
    if isinstance(paths, str):
        paths = sorted(glob.glob(paths)) or [paths]
    if not paths:
        raise RuntimeError("No catalog files found.")

    tables = []
    n_files = 0
    n_total = 0
    n_skipped_camera = 0
    for p in paths:
        t = _load_one(p)
        if t is None:
            continue
        if camera is not None:
            ccd = str(t.meta.get('CCD_NAME', '')).strip()
            if ccd and ccd != camera:
                n_skipped_camera += len(t)
                continue
        tables.append(t)
        n_files += 1
        n_total += len(t)

    if camera is not None and verbose and n_skipped_camera:
        print(f"Camera filter ({camera}): skipped {n_skipped_camera:,} stars from other cameras")

    if not tables:
        raise RuntimeError(
            f"No usable catalogs loaded"
            + (f" for camera {camera}" if camera else "") + "."
        )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = vstack(tables)

    if verbose:
        print(f"Loaded {n_total:,} stars from {n_files} file(s)"
              + (f"  [camera={camera}]" if camera else ""))

    magerr  = np.asarray(data['MAGERR_CALIB'], dtype=float)
    # MAGZERO in the catalog header is the standard photometric zeropoint.
    # The formula uses ZERO=10 internally, so mag_rel = MAG_CALIB - (MAGZERO - ZERO).
    mag_rel = np.asarray(data['MAG_CALIB'],    dtype=float) - np.asarray(data['MAGZERO'], dtype=float) + ZERO
    bgsigma = np.asarray(data['BGSIGMA'],      dtype=float)
    fwhm    = np.asarray(data['FWHM'],         dtype=float)
    gain    = np.asarray(data['GAIN'],          dtype=float)

    mask = (np.isfinite(mag_rel) & np.isfinite(magerr) &
            np.isfinite(bgsigma) & np.isfinite(fwhm) &
            (magerr  >= min_magerr) & (magerr  <= max_magerr) &
            (bgsigma >  0) &
            (fwhm    >= min_fwhm)   & (fwhm    <= max_fwhm))

    if 'FLAGS' in data.colnames:
        mask &= np.asarray(data['FLAGS'], dtype=int) <= max_flags

    n_good = mask.sum()
    if verbose:
        print(f"After quality cuts: {n_good:,}/{n_total:,} stars ({100*n_good/n_total:.1f}%)")
        print(f"  MAGERR range accepted : [{min_magerr:.3f}, {max_magerr:.3f}]")
        print(f"  Header FWHM accepted  : [{min_fwhm:.1f}, {max_fwhm:.1f}] px")
        print(f"  Median header FWHM    : {np.median(fwhm[mask]):.2f} px")

    if n_good < 50:
        raise RuntimeError(
            f"Too few good stars ({n_good}).  Adjust quality cuts or use more images."
        )

    return {
        'mag_rel':        mag_rel[mask],
        'bgsigma':        bgsigma[mask],
        'fwhm':           fwhm[mask],
        'gain':           gain[mask],
        'log_magerr_obs': np.log10(magerr[mask]),
        'n_total':        n_total,
        'n_good':         n_good,
        'median_fwhm':    float(np.median(fwhm[mask])),
    }


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------

def fit_ape(
    paths: Union[str, List[str]],
    camera: Optional[str] = None,
    ape_bounds: tuple = (0.5, 100.0),
    max_flags: int = MAX_FLAGS,
    min_magerr: float = MIN_MAGERR,
    max_magerr: float = MAX_MAGERR,
    min_fwhm: float = MIN_FWHM,
    max_fwhm: float = MAX_FWHM,
    verbose: bool = True,
) -> dict:
    """
    Fit the APE parameter to per-image ECSV calibration catalogs.

    Parameters
    ----------
    paths         : per-image ECSV file path(s) or glob pattern
    ape_bounds    : (min, max) search range for APE
    min/max_magerr: quality cut bounds on MAGERR_CALIB
    min/max_fwhm  : quality cut bounds on header FWHM [pixels]
    max_flags     : maximum SExtractor FLAGS value to accept

    Returns
    -------
    dict with keys:
        'ape'          : fitted APE value
        'rms_log'      : RMS of log10(magerr) residuals [dex]
        'bias_log'     : mean bias of residuals [dex]
        'p90_log'      : 90th-percentile |residual| [dex]
        'n_stars'      : number of stars used
        'n_total'      : total stars before quality cuts
        'median_fwhm'  : median header FWHM of calibration data [px]
                         → recommended value for config default_fwhm
    """
    if verbose:
        print("=" * 70)
        print("APE CALIBRATION FIT")
        print("=" * 70)
        print()

    data = load_catalogs(
        paths, camera=camera, max_flags=max_flags,
        min_magerr=min_magerr, max_magerr=max_magerr,
        min_fwhm=min_fwhm, max_fwhm=max_fwhm, verbose=verbose,
    )

    mag_rel = data['mag_rel']
    bgsigma = data['bgsigma']
    fwhm    = data['fwhm']
    gain    = data['gain']
    log_obs = data['log_magerr_obs']

    def objective(log_ape):
        ape = 10 ** log_ape
        log_pred = _log_magerror(mag_rel, bgsigma, fwhm, ape, gain)
        finite = np.isfinite(log_pred)
        if finite.sum() < 10:
            return 1e9
        return float(np.mean((log_pred[finite] - log_obs[finite]) ** 2))

    if verbose:
        print(f"\nFitting APE in [{ape_bounds[0]}, {ape_bounds[1]}]...")

    result = minimize_scalar(
        objective,
        bounds=(np.log10(ape_bounds[0]), np.log10(ape_bounds[1])),
        method='bounded',
    )
    ape_fit = float(10 ** result.x)

    log_pred = _log_magerror(mag_rel, bgsigma, fwhm, ape_fit, gain)
    finite   = np.isfinite(log_pred)
    resid    = log_pred[finite] - log_obs[finite]
    rms  = float(np.sqrt(np.mean(resid ** 2)))
    bias = float(np.mean(resid))
    p90  = float(np.percentile(np.abs(resid), 90))

    if verbose:
        print(f"\nFITTED  APE = {ape_fit:.4f}")
        print(f"  RMS residual (log magerr)  {rms:.4f} dex  (~{100*(10**rms-1):.0f}% in linear)")
        print(f"  Bias                       {bias:+.4f} dex")
        print(f"  90th-pct |residual|        {p90:.4f} dex")
        print(f"  Stars used                 {data['n_good']:,}")
        print(f"  Median header FWHM         {data['median_fwhm']:.2f} px")
        print()
        print(f"  → Set  default_fwhm = {data['median_fwhm']:.1f}  in your telescope config.")

    return {
        'ape':          ape_fit,
        'ape_bounds':   ape_bounds,
        'rms_log':      rms,
        'bias_log':     bias,
        'p90_log':      p90,
        'n_stars':      data['n_good'],
        'n_total':      data['n_total'],
        'median_fwhm':  data['median_fwhm'],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Fit APE (aperture growth curve parameter) from per-image ECSV catalogs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('catalog_files', nargs='+', metavar='FILE',
                        help='Per-image ECSV catalog files (e.g. /data/2501/*-df.ecsv)')
    parser.add_argument('--camera', default=None, metavar='NAME',
                        help='Keep only catalogs whose CCD_NAME header matches NAME '
                             '(e.g. C1, C2); default: use all')
    parser.add_argument('--min-magerr', type=float, default=MIN_MAGERR,
                        help='Minimum MAGERR_CALIB (exclude saturated stars)')
    parser.add_argument('--max-magerr', type=float, default=MAX_MAGERR,
                        help='Maximum MAGERR_CALIB (exclude very faint stars)')
    parser.add_argument('--min-fwhm', type=float, default=MIN_FWHM,
                        help='Minimum header FWHM to include [px]')
    parser.add_argument('--max-fwhm', type=float, default=MAX_FWHM,
                        help='Maximum header FWHM to include [px]')
    parser.add_argument('--max-flags', type=int, default=MAX_FLAGS,
                        help='Maximum SExtractor FLAGS value')
    parser.add_argument('--ape-min', type=float, default=0.5,
                        help='Lower bound for APE search')
    parser.add_argument('--ape-max', type=float, default=100.0,
                        help='Upper bound for APE search')
    args = parser.parse_args()

    camera = args.camera
    result = fit_ape(
        args.catalog_files,
        camera=camera,
        ape_bounds=(args.ape_min, args.ape_max),
        max_flags=args.max_flags,
        min_magerr=args.min_magerr,
        max_magerr=args.max_magerr,
        min_fwhm=args.min_fwhm,
        max_fwhm=args.max_fwhm,
        verbose=True,
    )

    cam_label = camera or 'C0'
    print(f"\nUpdate /etc/asarina/config [{cam_label}]:")
    print(f"  ape = {result['ape']:.4f}")
    print(f"  default_fwhm = {round(result['median_fwhm'], 1)}")
    print("Then retrain the background model:")
    print(f"  asarina-observe-train ~/phdb/stat/*.ecsv --camera {cam_label}")


if __name__ == '__main__':
    main()
