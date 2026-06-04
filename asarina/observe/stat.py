#!/usr/bin/env python3
"""
Per-image photometric statistics log.

Each record summarises one successfully calibrated image produced by the
asarina photometric pipeline.  Records are stored in per-night ECSV files
inside a configurable directory (default: /home/mates/phdb/stat/).

Night convention: noon-to-noon UTC — the same as asarina's daily summary
files.  A night starting at 20:00 UTC on 2025-05-28 and ending at 04:00
UTC on 2025-05-29 is stored in 20250528.ecsv.

Schema
------
jd         float64  d    Observation midpoint JD
exptime    float64  s    Exposure duration
filter     str           Filter name (e.g. Sloan_r)
airmass    float64       Observing airmass
sun_alt    float64  deg  Solar altitude
moon_alt   float64  deg  Lunar altitude
moon_dist  float64  deg  Angular distance moon→target
sun_dist   float64  deg  Angular distance sun→target
fwhm       float64  pix  Image-level PSF FWHM (from pipeline header)
ellip      float64       PSF ellipticity (1 - B/A)
zeropoint  float64  mag  Measured ZP for actual exptime (= MAGZERO from pipeline)
bgnoise    float64  ADU  Background RMS for actual exptime (= BGSIGMA from pipeline)
maglim     float64  mag  Limiting magnitude
image      str           FITS filename — deduplication key
ra         float64  deg  Target RA J2000
dec        float64  deg  Target Dec J2000

Derived columns added by read_stat() (not stored):
zp_1s      float64  mag  1s-normalised ZP  = zeropoint - 2.5*log10(exptime)
bgnoise_1s float64  ADU  1s-normalised BGN = bgnoise / sqrt(exptime)

Environment
-----------
RTS2_STAT_DIR — directory containing per-night ECSV files
               (default: /home/mates/phdb/stat)
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_STAT_DIR = os.environ.get('RTS2_STAT_DIR', '/home/mates/phdb/stat')

# ---------------------------------------------------------------------------
# Column schema: (name, dtype, unit, description)
# ---------------------------------------------------------------------------
_SCHEMA = [
    ('jd',        'f8',  'd',   'Observation midpoint JD'),
    ('exptime',   'f4',  's',   'Exposure duration'),
    ('filter',    'U16', '',    'Filter name'),
    ('airmass',   'f4',  '',    'Observing airmass'),
    ('sun_alt',   'f4',  'deg', 'Solar altitude'),
    ('moon_alt',  'f4',  'deg', 'Lunar altitude'),
    ('moon_dist', 'f4',  'deg', 'Angular distance moon to target'),
    ('sun_dist',  'f4',  'deg', 'Angular distance sun to target'),
    ('fwhm',      'f4',  'pix', 'Image-level PSF FWHM'),
    ('ellip',     'f4',  '',    'PSF ellipticity 1 - B/A'),
    ('zeropoint', 'f8',  'mag', 'Measured ZP for actual exptime = MAGZERO'),
    ('bgnoise',   'f4',  'ADU', 'Background RMS for actual exptime = BGSIGMA'),
    ('maglim',    'f4',  'mag', 'Limiting magnitude'),
    ('image',     'U96', '',    'FITS filename, deduplication key'),
    ('camera',    'U16', '',    'Camera name (CCD_NAME from pipeline header)'),
    ('ra',        'f8',  'deg', 'Target RA J2000'),
    ('dec',       'f8',  'deg', 'Target Dec J2000'),
]

COLUMN_NAMES = [s[0] for s in _SCHEMA]
_NUMERIC_COLS = {s[0] for s in _SCHEMA if s[1] != 'U16' and not s[1].startswith('U')}


# ---------------------------------------------------------------------------
# Night helpers
# ---------------------------------------------------------------------------

def night_from_jd(jd: float) -> str:
    """Return 'YYYYMMDD' for the astronomical night containing jd.

    Uses noon-to-noon UTC convention: subtract 12 h before computing date,
    so all observations from one sunset to the next sunrise share a date.
    Consistent with asarina's daily_summary_dir naming.
    """
    unix = (jd - 2440587.5) * 86400.0 - 43200.0
    return datetime.utcfromtimestamp(unix).strftime('%Y%m%d')


# ---------------------------------------------------------------------------
# Record extraction from per-image asarina ECSV
# ---------------------------------------------------------------------------

def record_from_ecsv(ecsv_path: str) -> Optional[dict]:
    """
    Extract a stat record from a per-image asarina photometric ECSV catalog.

    Reads the image-level metadata (FITS header keywords stored as ECSV
    table meta) and returns a dict ready for write_stat_record().

    Returns None if required keys are missing or values are invalid.
    """
    from astropy.table import Table

    try:
        tbl = Table.read(ecsv_path, format='ascii.ecsv')
    except Exception as e:
        logger.error(f"Cannot read {ecsv_path}: {e}")
        return None

    m = tbl.meta

    # ------------------------------------------------------------------
    # Required keys — must be present for a useful stat record
    # ------------------------------------------------------------------
    required = ['JD', 'EXPTIME', 'FILTER', 'AIRMASS', 'SUN_ALT', 'MOONALT',
                'MOONDIST', 'FWHM', 'BGSIGMA', 'MAGZERO', 'FITSFILE',
                'OBJRA', 'OBJDEC']
    missing = [k for k in required if k not in m]
    if missing:
        logger.warning(f"Missing header keys in {ecsv_path}: {missing}")
        return None

    # LIMMAG was renamed MAGLIM in a pipeline update; accept both
    maglim = m.get('LIMMAG', m.get('MAGLIM', float('nan')))

    # ------------------------------------------------------------------
    # Sun distance: use header SUNDIST if present, else compute from
    # OBJRA/OBJDEC + astropy (one ephemeris call, ~10 ms).
    # ------------------------------------------------------------------
    sun_dist = m.get('SUNDIST')
    if sun_dist is None:
        try:
            from astropy.coordinates import SkyCoord, get_body, EarthLocation
            from astropy.time import Time
            import astropy.units as u

            lat = float(m.get('LATITUDE', 49.9093889))
            lon = float(m.get('LONGITUD', 14.7813631))
            alt = float(m.get('ALTITUDE', 530.0))
            site = EarthLocation(lat=lat * u.deg, lon=lon * u.deg,
                                 height=alt * u.m)
            t       = Time(float(m['JD']), format='jd')
            sun     = get_body('sun', t, site)
            target  = SkyCoord(float(m['OBJRA']) * u.deg,
                               float(m['OBJDEC']) * u.deg)
            sun_dist = float(target.separation(sun).deg)
        except Exception as e:
            logger.warning(f"Could not compute sun_dist for {ecsv_path}: {e}")
            sun_dist = float('nan')

    return {
        'jd':        float(m['JD']),
        'exptime':   float(m['EXPTIME']),
        'filter':    str(m['FILTER']),
        'airmass':   float(m['AIRMASS']),
        'sun_alt':   float(m['SUN_ALT']),
        'moon_alt':  float(m['MOONALT']),
        'moon_dist': float(m['MOONDIST']),
        'sun_dist':  float(sun_dist),
        'fwhm':      float(m['FWHM']),
        'ellip':     float(m['ELLIP']) if 'ELLIP' in m else float('nan'),
        'zeropoint': float(m['MAGZERO']),
        'bgnoise':   float(m['BGSIGMA']),
        'maglim':    float(maglim),
        'image':     str(Path(m['FITSFILE']).name),
        'camera':    str(m.get('CCD_NAME', '')),
        'ra':        float(m['OBJRA']),
        'dec':       float(m['OBJDEC']),
    }


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_stat_record(record: dict, stat_dir: str = _STAT_DIR) -> None:
    """
    Append one image's record to the per-night stat ECSV.

    If a record for the same image already exists in the file (identified by
    the 'image' column), the old record is replaced.  This handles pipeline
    reprocessing cleanly without duplicates.

    The write is atomic: the new content is written to a temp file in the same
    directory, then renamed over the target — safe against partial writes and
    concurrent readers.

    Parameters
    ----------
    record  : dict with keys matching COLUMN_NAMES (see module docstring)
    stat_dir: directory containing per-night ECSV files
    """
    from astropy.table import Table

    stat_dir = Path(stat_dir)
    stat_dir.mkdir(parents=True, exist_ok=True)

    night = night_from_jd(record['jd'])
    path  = stat_dir / f'{night}.ecsv'
    tmp   = stat_dir / f'{night}.ecsv.tmp'

    if path.exists():
        try:
            tbl = Table.read(str(path), format='ascii.ecsv')
            # Remove stale record for this image if reprocessing
            if 'image' in tbl.colnames:
                mask = np.array([row['image'] != record['image'] for row in tbl])
                tbl  = tbl[mask]
                if len(tbl) < mask.size:
                    logger.debug(f"Replaced existing record for {record['image']}")
        except Exception as e:
            logger.warning(f"Could not read {path}, starting fresh: {e}")
            tbl = _empty_table()
    else:
        tbl = _empty_table()

    tbl.add_row({k: record.get(k, float('nan') if k in _NUMERIC_COLS else '')
                 for k in COLUMN_NAMES})

    tbl.write(str(tmp), format='ascii.ecsv', overwrite=True)
    os.rename(str(tmp), str(path))


def _empty_table():
    """Return an empty astropy Table with the full stat schema."""
    from astropy.table import Table, Column
    import astropy.units as u

    cols = []
    for name, dtype, unit, desc in _SCHEMA:
        arr = np.array([], dtype=dtype)
        col = Column(arr, name=name, description=desc)
        if unit:
            try:
                col.unit = unit
            except Exception:
                pass
        cols.append(col)
    return Table(cols)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def _read_one(path: Path) -> Optional[pd.DataFrame]:
    """Read a single stat ECSV file into a DataFrame."""
    try:
        # ECSV has a block of # comments then a plain column-name line.
        # pandas with comment='#' skips the metadata block and picks up
        # the column names from the first non-commented line.
        df = pd.read_csv(str(path), comment='#', sep=r'\s+', dtype={'image': str, 'filter': str})
        if df.empty:
            return None
        return df
    except Exception as e:
        logger.warning(f"Could not read {path}: {e}")
        return None


def _add_derived(df: pd.DataFrame) -> None:
    """Add zp_1s and bgnoise_1s columns in-place."""
    if 'exptime' in df.columns and 'zeropoint' in df.columns:
        df['zp_1s'] = df['zeropoint'] - 2.5 * np.log10(df['exptime'].clip(lower=1e-3))
    if 'exptime' in df.columns and 'bgnoise' in df.columns:
        df['bgnoise_1s'] = df['bgnoise'] / np.sqrt(df['exptime'].clip(lower=1e-3))


def read_stat(
    source: Union[str, Path, None] = None,
    since_jd: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load stat records from a file or directory of per-night ECSVs.

    Parameters
    ----------
    source : str, Path, or None
        A path to a single stat ECSV file, or a directory containing
        per-night ECSV files named YYYYMMDD.ecsv.
        None → uses RTS2_STAT_DIR environment variable.
    since_jd : float, optional
        If given, return only records with jd >= since_jd.

    Returns
    -------
    pd.DataFrame sorted by jd, with derived columns zp_1s and bgnoise_1s.
    """
    src = Path(source) if source is not None else Path(_STAT_DIR)

    if src.is_dir():
        files = sorted(src.glob('*.ecsv'))
        if not files:
            return pd.DataFrame(columns=COLUMN_NAMES)
        frames = [_read_one(f) for f in files]
        frames = [f for f in frames if f is not None and len(f) > 0]
        if not frames:
            return pd.DataFrame(columns=COLUMN_NAMES)
        data = pd.concat(frames, ignore_index=True)
    else:
        data = _read_one(src)
        if data is None:
            return pd.DataFrame(columns=COLUMN_NAMES)

    if since_jd is not None and 'jd' in data.columns:
        data = data[data['jd'] >= since_jd]

    _add_derived(data)
    data = data.sort_values('jd').reset_index(drop=True)
    return data


def main():
    """
    CLI: ingest one or more asarina per-image ECSV files into the stat directory.

    Designed to be called via find … | xargs:

        find ~/phdb -name '*.ecsv' | xargs rtspy-observe-stat-ingest
        find ~/phdb -name '*.ecsv' | xargs rtspy-observe-stat-ingest --stat-dir /tmp/stat

    Prints one line per file: OK, SKIP (already present and unchanged), or FAIL.
    Exit code is the number of failures (0 = all succeeded).
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Ingest asarina per-image ECSVs into the rtspy stat directory',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('ecsv_files', nargs='+', metavar='FILE',
                        help='Per-image ECSV catalog(s) produced by asarina')
    parser.add_argument('--stat-dir', default=_STAT_DIR, metavar='DIR',
                        help='Destination stat directory')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Only print failures')
    args = parser.parse_args()

    n_ok = n_fail = 0
    for path in args.ecsv_files:
        record = record_from_ecsv(path)
        if record is None:
            print(f'FAIL  {path}  (could not extract record)', file=sys.stderr)
            n_fail += 1
            continue
        try:
            write_stat_record(record, stat_dir=args.stat_dir)
            if not args.quiet:
                print(f'OK    {record["image"]}')
            n_ok += 1
        except Exception as e:
            print(f'FAIL  {path}  ({e})', file=sys.stderr)
            n_fail += 1

    if not args.quiet:
        print(f'\n{n_ok} ingested, {n_fail} failed', file=sys.stderr)
    sys.exit(n_fail)


def load_recent(
    source: Union[str, Path, None] = None,
    window_min: float = 15.0,
    reference_jd: Optional[float] = None,
) -> tuple:
    """
    Return (recent_df, last_row) for the given look-back window.

    Convenience wrapper used by observe.py.  Loads today's and yesterday's
    files to handle observations near midnight.

    Parameters
    ----------
    source      : stat file or directory (None → RTS2_STAT_DIR)
    window_min  : look-back window in minutes
    reference_jd: reference epoch (default: now)

    Returns
    -------
    (recent: pd.DataFrame, last: pd.Series | None)
    """
    import time as _time

    jd_now   = reference_jd if reference_jd is not None else (
        2440587.5 + _time.time() / 86400.0
    )
    since_jd = jd_now - max(window_min, 60.0 * 24) / 1440.0  # at least 1 day back

    data = read_stat(source, since_jd=since_jd)
    data = data[data['exptime'] > 0]

    t_start = jd_now - window_min / 1440.0
    recent  = data[data['jd'] >= t_start].sort_values('jd').reset_index(drop=True)
    last    = data.sort_values('jd').iloc[-1] if len(data) > 0 else None

    return recent, last
