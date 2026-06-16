#!/usr/bin/env python3
"""
Automatic calibration frame processing.

This script creates master darks and master flats from raw calibration frames
in the image archive. It handles:
- Discovery of available years and cameras in the archive
- Quality filtering of input frames based on statistics
- Camera-specific noise model validation for darks
- Proper pairing of flats with matching darks
- Physical camera identification via chip_id registry

Usage:
    make_calib.py --year 2024
    make_calib.py --year 2025 --archive /storage/archive
    make_calib.py --year 2024 --dry-run

Output structure:
    {camera_id}/{year}/ (e.g., andor46/2024/, fli534/2010/)
"""

MAX_WORKERS=16

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
import random
import re
import sqlite3
import hashlib
import struct
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from astropy.io import fits
    import numpy as np
except ImportError:
    print("Error: astropy and numpy are required. Install with: pip install astropy numpy")
    sys.exit(1)

from asarina.chip_id import get_camera_id


def image_hdu(hdul):
    """Return the HDU that actually holds image data and keywords.

    Rice/tile-compressed FITS (the ``.fitz`` files in older archives) keep an
    empty PrimaryHDU at index 0 and store the image plus its header keywords
    (CCD_SER, IMAGETYP, ...) in a CompImageHDU at index 1. Plain ``.fits`` files
    carry everything in the PrimaryHDU. This picks the right one transparently.
    """
    hdu = hdul[0]
    if hdu.data is None and len(hdul) > 1:
        hdu = hdul[1]
    return hdu


def ccd_ser_to_camera_id(ccd_ser: str) -> str:
    """Convert raw ccd_ser string to short camera_id using registry.

    When loading from cache, we have ccd_ser but may not have camera_id.
    This creates a fake header and runs it through the registry.
    """
    if not ccd_ser or ccd_ser == 'unknown':
        return 'unknown'
    # Create minimal header with ccd_ser in both fields the registry checks
    fake_header = {'CCD_SER': ccd_ser, 'CCD_TYPE': ccd_ser}
    return get_camera_id(fake_header)


# ============================================================================
# Statistics Cache - persistent storage to avoid recomputation
# ============================================================================

class StatsCache:
    """
    SQLite-based cache for image statistics.

    Key: file path + mtime (so we recompute if file changes)
    Value: all computed statistics
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default: ~/.cache/pyrt/stats_cache.db
            cache_dir = Path.home() / ".cache" / "pyrt"
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = cache_dir / "stats_cache.db"

        self.db_path = Path(db_path)
        self._conn = None  # Persistent connection
        self._init_db()

    def _get_conn(self):
        """Get persistent connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        """Close persistent connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stats (
                    file_path TEXT PRIMARY KEY,
                    mtime REAL,
                    exptime REAL,
                    sigma REAL,
                    median REAL,
                    average REAL,
                    ma_diff REAL,
                    ccd_temp REAL,
                    ccd_set REAL,
                    binning TEXT,
                    filter TEXT,
                    imagetyp TEXT,
                    ccd_ser TEXT,
                    naxis1 INTEGER,
                    naxis2 INTEGER
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mtime ON stats(file_path, mtime)")
            # Add columns if they don't exist (for existing DBs)
            try:
                conn.execute("ALTER TABLE stats ADD COLUMN naxis1 INTEGER")
            except sqlite3.OperationalError:
                pass  # Column already exists
            try:
                conn.execute("ALTER TABLE stats ADD COLUMN naxis2 INTEGER")
            except sqlite3.OperationalError:
                pass  # Column already exists
            try:
                conn.execute("ALTER TABLE stats ADD COLUMN camera_id TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists
            conn.commit()

    def get(self, file_path: str) -> Optional[dict]:
        """
        Get cached stats for a file, or None if not cached/stale/incomplete.

        This is a pure cache lookup - NO file I/O except stat() for mtime.
        If entry is missing, stale, or incomplete (missing naxis), returns None
        to trigger recomputation which will read the file and cache everything.
        """
        try:
            mtime = os.path.getmtime(file_path)
        except OSError:
            return None

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM stats WHERE file_path = ? AND mtime = ?",
                (file_path, mtime)
            ).fetchone()

            if not row:
                return None

            # Check for required fields - if missing, recompute
            try:
                naxis1 = row['naxis1'] if 'naxis1' in row.keys() else None
                naxis2 = row['naxis2'] if 'naxis2' in row.keys() else None
            except (KeyError, IndexError):
                naxis1, naxis2 = None, None

            # If dimensions missing, entry is incomplete - trigger recomputation
            if naxis1 is None or naxis2 is None:
                conn.execute("DELETE FROM stats WHERE file_path = ?", (file_path,))
                conn.commit()
                return None

            # Type conversion helpers
            def to_float(v):
                if v is None:
                    return None
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, bytes):
                    raise ValueError("Corrupted binary data")
                return float(v)

            def to_int(v):
                if v is None:
                    return 0
                if isinstance(v, (int, float)):
                    return int(v)
                if isinstance(v, bytes):
                    raise ValueError("Corrupted binary data")
                return int(v)

            def to_str(v):
                if v is None:
                    return None
                if isinstance(v, bytes):
                    return v.decode('utf-8', errors='replace')
                return str(v)

            try:
                # Get camera_id, falling back to ccd_ser if not in cache
                try:
                    camera_id = to_str(row['camera_id']) if row['camera_id'] else None
                except (KeyError, IndexError):
                    camera_id = None
                ccd_ser = to_str(row['ccd_ser']) if row['ccd_ser'] != '' else 'unknown'
                if not camera_id:
                    camera_id = ccd_ser_to_camera_id(ccd_ser)  # convert via registry

                return {
                    'file': to_str(row['file_path']),
                    'exptime': to_float(row['exptime']),
                    'sigma': to_float(row['sigma']),
                    'median': to_float(row['median']),
                    'average': to_float(row['average']),
                    'ma_diff': to_float(row['ma_diff']),
                    'ccd_temp': to_float(row['ccd_temp']),
                    'ccd_set': to_float(row['ccd_set']),
                    'binning': to_str(row['binning']),
                    'filter': to_str(row['filter']) if row['filter'] != '' else None,
                    'imagetyp': to_str(row['imagetyp']),
                    'ccd_ser': ccd_ser,
                    'camera_id': camera_id,
                    'naxis1': to_int(naxis1),
                    'naxis2': to_int(naxis2),
                }
            except (ValueError, TypeError):
                # Corrupted entry - delete and recompute
                conn.execute("DELETE FROM stats WHERE file_path = ?", (file_path,))
                conn.commit()
                return None
        return None

    def get_many(self, file_paths: list) -> dict:
        """
        Batch lookup - fetch all entries in one query, filter by mtime in Python.
        Returns dict mapping file_path -> cached stats (or None if stale/missing).
        Much faster than calling get() 54k times.
        """
        if not file_paths:
            return {}

        # Get mtimes for all files (fast syscalls)
        mtimes = {}
        for f in file_paths:
            try:
                mtimes[f] = os.path.getmtime(f)
            except OSError:
                pass

        # Fetch rows in batches (SQLite has placeholder limits)
        conn = self._get_conn()
        rows = []
        batch_size = 500  # Safe for all SQLite versions

        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            placeholders = ",".join("?" * len(batch))
            batch_rows = conn.execute(
                f"SELECT * FROM stats WHERE file_path IN ({placeholders})",
                batch
            ).fetchall()
            rows.extend(batch_rows)

        log(f"Cache query: requested {len(file_paths)}, found {len(rows)} rows", "debug")

        # Debug counters
        skip_mtime = 0
        skip_naxis = 0
        skip_stats = 0
        skip_error = 0

        # Type conversion helpers
        def to_float(v):
            if v is None:
                return None
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, bytes):
                # Legacy: some values stored as packed float32
                if len(v) == 4:
                    return struct.unpack('<f', v)[0]
                return None
            return float(v)

        def to_int(v):
            if v is None:
                return 0
            if isinstance(v, (int, float)):
                return int(v)
            if isinstance(v, bytes):
                return 0
            return int(v)

        def to_str(v):
            if v is None:
                return ''
            if isinstance(v, bytes):
                return ''
            return str(v)

        # Build result dict
        result = {}
        for row in rows:
            path = row['file_path']
            expected_mtime = mtimes.get(path)

            # Skip if file not found or mtime doesn't match
            if expected_mtime is None or row['mtime'] != expected_mtime:
                skip_mtime += 1
                continue

            # Check for required dimensions
            try:
                naxis1 = row['naxis1'] if 'naxis1' in row.keys() else None
                naxis2 = row['naxis2'] if 'naxis2' in row.keys() else None
            except (KeyError, IndexError):
                skip_naxis += 1
                continue

            if naxis1 is None or naxis2 is None:
                skip_naxis += 1
                continue

            # Check for required stats fields
            sigma = to_float(row['sigma'])
            median = to_float(row['median'])
            if sigma is None or median is None:
                skip_stats += 1
                continue  # Incomplete entry, will be recomputed

            try:
                # Get camera_id, falling back to ccd_ser if not in cache
                try:
                    camera_id = to_str(row['camera_id']) if row['camera_id'] else None
                except (KeyError, IndexError):
                    camera_id = None
                ccd_ser = to_str(row['ccd_ser']) if row['ccd_ser'] != '' else 'unknown'
                if not camera_id:
                    camera_id = ccd_ser_to_camera_id(ccd_ser)  # convert via registry

                result[path] = {
                    'file': path,
                    'exptime': to_float(row['exptime']),
                    'sigma': sigma,
                    'median': median,
                    'average': to_float(row['average']),
                    'ma_diff': to_float(row['ma_diff']),
                    'ccd_temp': to_float(row['ccd_temp']),
                    'ccd_set': to_float(row['ccd_set']),
                    'binning': to_str(row['binning']),
                    'filter': to_str(row['filter']) if row['filter'] != '' else None,
                    'imagetyp': to_str(row['imagetyp']),
                    'ccd_ser': ccd_ser,
                    'camera_id': camera_id,
                    'naxis1': to_int(naxis1),
                    'naxis2': to_int(naxis2),
                }
            except (ValueError, TypeError):
                skip_error += 1
                continue  # Skip corrupted entries

        log(f"Cache result: {len(result)} valid, skipped: mtime={skip_mtime}, naxis={skip_naxis}, stats={skip_stats}, error={skip_error}", "debug")
        return result

    def put(self, stats: dict):
        """Store stats in cache."""
        if stats is None:
            return

        try:
            mtime = os.path.getmtime(stats['file'])
        except OSError:
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO stats
                (file_path, mtime, exptime, sigma, median, average, ma_diff,
                 ccd_temp, ccd_set, binning, filter, imagetyp, ccd_ser, camera_id, naxis1, naxis2)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stats['file'],
                mtime,
                stats['exptime'],
                stats['sigma'],
                stats['median'],
                stats['average'],
                stats['ma_diff'],
                stats['ccd_temp'],
                stats['ccd_set'],
                stats['binning'],
                stats['filter'] or '',
                stats['imagetyp'],
                stats['ccd_ser'] or '',
                stats.get('camera_id') or '',
                stats.get('naxis1'),
                stats.get('naxis2'),
            ))
            conn.commit()

    def put_batch(self, stats_list: list):
        """Store multiple stats in cache efficiently."""
        if not stats_list:
            return

        rows = []
        for stats in stats_list:
            if stats is None:
                continue
            try:
                mtime = os.path.getmtime(stats['file'])
                rows.append((
                    stats['file'],
                    mtime,
                    stats['exptime'],
                    stats['sigma'],
                    stats['median'],
                    stats['average'],
                    stats['ma_diff'],
                    stats['ccd_temp'],
                    stats['ccd_set'],
                    stats['binning'],
                    stats['filter'] or '',
                    stats['imagetyp'],
                    stats['ccd_ser'] or '',
                    stats.get('camera_id') or '',
                    stats.get('naxis1'),
                    stats.get('naxis2'),
                ))
            except OSError:
                continue

        if rows:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany("""
                    INSERT OR REPLACE INTO stats
                    (file_path, mtime, exptime, sigma, median, average, ma_diff,
                     ccd_temp, ccd_set, binning, filter, imagetyp, ccd_ser, camera_id, naxis1, naxis2)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, rows)
                conn.commit()

    def stats(self) -> dict:
        """Return cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM stats").fetchone()[0]
            return {'entries': count, 'path': str(self.db_path)}


# Global cache instance
_stats_cache = None

def get_stats_cache() -> StatsCache:
    """Get or create the global stats cache."""
    global _stats_cache
    if _stats_cache is None:
        _stats_cache = StatsCache()
    return _stats_cache


# Camera noise model parameters: (expected_sigma, tolerance)
# These are used to validate dark frames by comparing pairs
# Keys are camera_id from chip_id.py registry
CAMERA_NOISE_MODELS = {
    # D50 cameras
    "andor46": {  # Andor iXon-Ultra DU888, serial 10046
        # sigma = sqrt(8.09² + 0.052 * exptime), from fit to noise-andor46.out
        "1x1": lambda exp, temp: (math.sqrt(8.09**2 + 0.052 * exp), 0.10),
        "2x2": lambda exp, temp: (8.36174, 0.16),
    },
    # BOOTES-2 / COLORES Andor iXon, serial 3567
    "andor3567": {
        # Pair-difference sigma measured flat at ~3.0 ADU across 1-300 s at
        # -58.5 C (read-noise dominated, negligible dark current). Wide-ish
        # tolerance covers the observed 1.9-3.65 spread; refine from
        # noise-andor3567.out once a validated run accumulates data.
        "1x1": lambda exp, temp: (3.0, 0.5),
        "2x2": lambda exp, temp: (3.0, 0.5),  # TODO: verify if 2x2 darks appear
    },
    "fli534": {  # FLI IMG4710 EEV CCD47-10, serial suffix 258.534
        # sigma = sqrt(5.72² + 0.0595 * exptime), from fit to noise-fli534.out
        "1x1": lambda exp, temp: (math.sqrt(5.72**2 + 0.0595 * exp), 0.10),
        # 2x2: same readnoise, 4x dark current (assumes same gain) - verify when data available
        "2x2": lambda exp, temp: (math.sqrt(5.72**2 + 0.238 * exp), 0.15),
    },
    # BART/CNF cameras
    "fli785": {  # FLI MAXcam EEV CCD47-10, serial suffix 259.785
        "1x1": lambda exp, temp: (
            (6.23999194857853**2 + 2.66690536046723 * (exp + 4.04663672443005) * (2.0 ** (temp / 4.82527675675717))) ** 0.5,
            0.05
        ),
        "2x2": lambda exp, temp: (
            (7.07675537020698**2 + 11.8171601133346 * (exp - 0.109076116406907) * (2.0 ** (temp / 4.48092070386459))) ** 0.5,
            0.05
        ),
    },
    # SBT cameras
    "mi6166": {  # Moravian G4-16000 Kodak KAF-16803 (SBT C1)
        # sigma = sqrt(10.02² + 0.045 * exptime), from fit to noise-mi6166.out
        "1x1": lambda exp, temp: (math.sqrt(10.02**2 + 0.045 * exp), 0.15),
        "2x2": lambda exp, temp: (math.sqrt(10.02**2 + 0.045 * exp), 0.15),  # TODO: verify 2x2
    },
    # Mobile backup cameras
    "mi2066": {  # Moravian G2-1600 Kodak KAF-1603
        "1x1": lambda exp, temp: (9.8 - 0.4, 0.2),
        "2x2": lambda exp, temp: (7.74, 0.2),
    },
    "mi2065": {  # Moravian G2-1600, new electronics
        "1x1": lambda exp, temp: (9.65, 0.2),
        "2x2": lambda exp, temp: (8.20603, 0.2),
    },
    "mi2065v1": {  # Moravian G2-1600, old electronics
        "1x1": lambda exp, temp: (10.6275, 0.2),
        "2x2": lambda exp, temp: (10.6275, 0.2),
    },
}

# Default filtering criteria when camera is not in the noise model
DEFAULT_NOISE_MODEL = {
    "1x1": lambda exp, temp: (8.0, 0.5),
    "2x2": lambda exp, temp: (8.0, 0.5),
}


@dataclass
class CalibConfig:
    """Configuration for calibration processing."""
    archive_base: str = "/archive"
    # Cameras to scan (empty = auto-detect from directory tree)
    cameras: list = field(default_factory=list)
    # Dark filtering keyed by camera_id
    dark_filter: dict = field(default_factory=dict)
    # Flat filtering keyed by camera_id
    flat_filter: dict = field(default_factory=dict)
    # Number of parallel workers
    parallel_workers: int = MAX_WORKERS
    # Max allowed |CCD_TEMP - CCD_SET| (deg C) for a frame to count as
    # temperature-stabilized. TEC cameras near their cooling floor sit a
    # couple degrees off setpoint at steady state (e.g. BOOTES-2 iXon stabilizes
    # at -58.5 vs a -60 setpoint), so 1.0 was too tight; 2.5 still rejects
    # genuinely warm/uncooled frames (which are tens of degrees off).
    temp_tolerance: float = 2.5


# Camera-specific filter settings (keyed by camera_id from registry)
CAMERA_FILTERS = {
    "dark": {
        "andor46": {"max_sigma": 15, "min_median": None, "max_median": 1300},
        # BOOTES-2 iXon: good darks ~99 ADU median, sigma ~2.1; warm/saturated
        # junk reaches median 32767 / sigma 216 - cut those out.
        "andor3567": {"max_sigma": 10, "min_median": None, "max_median": 1000},
        "fli534": {"max_sigma": 15, "min_median": None, "max_median": None},
        "fli785": {"max_sigma": 15, "min_median": None, "max_median": None},
        "mi6166": {"max_sigma": 15, "min_median": None, "max_median": None},
        "mi6167": {"max_sigma": 15, "min_median": None, "max_median": None},
        "mi2066": {"max_sigma": 15, "min_median": None, "max_median": None},
        "mi2065": {"max_sigma": 15, "min_median": None, "max_median": None},
    },
    "flat": {
        "andor46": {"min_median": 10000, "max_median": 50000},
        "andor3567": {"min_median": 5000, "max_median": 50000},
        "fli534": {"min_median": 5000, "max_median": 50000},
        "fli785": {"min_median": 5000, "max_median": 50000},
        "mi6166": {"min_median": 5000, "max_median": 50000},
        "mi6167": {"min_median": 5000, "max_median": 50000},
        "mi2066": {"min_median": 5000, "max_median": 50000},
        "mi2065": {"min_median": 5000, "max_median": 50000},
    },
}


# Default config
DEFAULT_CONFIG = CalibConfig(
    archive_base="/archive",
    dark_filter=CAMERA_FILTERS["dark"],
    flat_filter=CAMERA_FILTERS["flat"],
)


VERBOSE = False

def log(msg: str, level: str = "info"):
    """Simple logging with level prefix."""
    if level == "debug" and not VERBOSE:
        return
    prefix = {"info": "[INFO]", "warn": "[WARN]", "error": "[ERROR]", "debug": "[DEBUG]"}
    print(f"{prefix.get(level, '[INFO]')} {msg}")


def progress_bar(current: int, total: int, width: int = 40, prefix: str = ""):
    """Display an in-place progress bar."""
    if total == 0:
        return
    pct = current / total
    filled = int(width * pct)
    bar = "=" * filled + ">" + "." * (width - filled - 1) if filled < width else "=" * width
    print(f"\r{prefix}[{bar}] {current}/{total} ({100*pct:.1f}%)", end="", flush=True)
    if current >= total:
        print()  # newline when done


def discover_years(archive_base: str) -> list:
    """Discover available years in the archive."""
    years = []
    base = Path(archive_base)
    if not base.exists():
        return years

    for entry in sorted(base.iterdir()):
        if entry.is_dir() and entry.name.isdigit() and len(entry.name) == 4:
            years.append(entry.name)

    return years


# Directory-name conventions for calibration frames. Camera-level directory
# names are NOT in here on purpose: they are arbitrary (C0/C1 on D50, NF/WF on
# FRAM, andor/alta on BOOTES-1, absent on single-camera archives like BOOTES-2),
# so layout is detected by content, never by matching camera names.
TARGET_ID_RE = re.compile(r"^\d{5}$")          # RTS2 target id, e.g. 00001
CALIB_DIR_NAMES = {                            # accepted calib dir spellings
    "dark": ("dark", "darks"),
    "flat": ("flat", "flats"),
}
CALIB_TARGET_IDS = {                           # RTS2 convention: tgt 1=dark, 2=flat
    "dark": "00001",
    "flat": "00002",
}
_NIGHT_CONTENT_DIRS = {"dark", "darks", "flat", "flats"}


def _is_night_contents(d: Path) -> bool:
    """True if *d* directly holds a night's observations.

    A night-contents directory contains RTS2 target-id subdirs (5 digits) and/or
    calibration subdirs (dark/darks/flat/flats). This is the marker we use to
    locate the right depth without trusting any camera directory name.
    """
    try:
        for child in d.iterdir():
            if child.is_dir() and (TARGET_ID_RE.match(child.name)
                                   or child.name in _NIGHT_CONTENT_DIRS):
                return True
    except OSError:
        pass
    return False


def detect_layout(archive_base: str, year: str, sample: int = 30) -> str:
    """Detect archive layout for a year: 'single' or 'multi' camera.

    'single' = calibration sits directly under the night ({year}/{night}/dark/).
    'multi'  = a camera directory (any name) sits between night and calib
               ({year}/{night}/<camera>/dark/).
    Decided by inspecting directory *content*, not names.
    """
    year_path = Path(archive_base) / year
    if not year_path.exists():
        return "single"

    single = multi = 0
    for night in sorted(p for p in year_path.iterdir() if p.is_dir()):
        if _is_night_contents(night):
            single += 1
        else:
            try:
                if any(_is_night_contents(c) for c in night.iterdir() if c.is_dir()):
                    multi += 1
            except OSError:
                pass
        if single + multi >= sample:
            break

    return "multi" if multi > single else "single"


def discover_cameras(archive_base: str, year: str, layout: str = "auto") -> list:
    """Discover available cameras for a given year.

    Multi-camera layout: returns the camera directory names (whatever they are).
    Single-camera layout: there is no camera directory, so identify the
    camera(s) from FITS headers of a few calibration frames (e.g. 'andor3567').
    """
    if layout == "auto":
        layout = detect_layout(archive_base, year)

    year_path = Path(archive_base) / year
    if not year_path.exists():
        return []

    if layout == "multi":
        cameras = set()
        for night in year_path.iterdir():
            if not night.is_dir() or _is_night_contents(night):
                continue
            try:
                for cam in night.iterdir():
                    if cam.is_dir() and _is_night_contents(cam):
                        cameras.add(cam.name)
            except OSError:
                pass
        return sorted(cameras)

    # single-camera: identify by header from a sample of calib frames
    sample = (find_calibration_files(archive_base, year, None, "dark", layout="single")
              or find_calibration_files(archive_base, year, None, "flat", layout="single"))
    camera_ids = set()
    for f in sample[:25]:
        try:
            with fits.open(f) as hdul:
                cid = get_camera_id(image_hdu(hdul).header)
                if cid and cid != "unknown":
                    camera_ids.add(cid)
        except Exception:
            continue
    return sorted(camera_ids)


def find_calibration_files(archive_base: str, year: str, camera: str,
                           file_type: str, layout: str = "auto") -> list:
    """Find all calibration files (darks or flats) for a given year.

    Handles both the multi-camera layout ({year}/{night}/<camera>/dark[s]/) and
    the single-camera layout ({year}/{night}/dark[s]/), accepts both 'dark' and
    'darks' spellings, the RTS2 target-id convention (00001=dark, 00002=flat),
    and both '.fits' and Rice-compressed '.fitz' extensions. The path is only a
    hint for where to look; IMAGETYP from the header is the final arbiter
    (compute_stats_parallel filters on it).
    """
    import glob

    if layout == "auto":
        layout = detect_layout(archive_base, year)

    calib_dirs = CALIB_DIR_NAMES[file_type]
    target_id = CALIB_TARGET_IDS[file_type]
    exts = ("fits", "fitz")
    calib_subdirs = list(calib_dirs) + [target_id]

    if layout == "multi":
        cam = camera if camera else "*"
        prefixes = [f"{archive_base}/{year}/*/{cam}"]
    else:
        prefixes = [f"{archive_base}/{year}/*"]

    files = set()
    for prefix in prefixes:
        for sub in calib_subdirs:
            for ext in exts:
                files.update(glob.glob(f"{prefix}/{sub}/20*.{ext}"))

    return sorted(files)


def compute_image_stats(fits_path: str) -> dict:
    """Compute statistics for a FITS image (equivalent to do_sigma.py)."""
    try:
        with fits.open(fits_path) as hdul:
            hdu = image_hdu(hdul)
            data = hdu.data.astype(np.float32)
            header = hdu.header

            # Compute row-by-row differences for noise estimation
            scale_factor = 1.0489  # median to sigma conversion
            diffs = np.abs(data[:-1] - data[1:])
            row_medians = np.nanmedian(diffs, axis=1)
            sigma = np.nanmedian(row_medians) * scale_factor

            median_val = np.nanmedian(data)
            avg_val = np.nanmean(data)
            exp_time = header.get('EXPTIME', 0)

            # Get raw CCD_SER for noise model lookup
            ccd_ser = header.get('CCD_SER', header.get('CCD_TYPE', 'unknown'))
            if ccd_ser:
                ccd_ser = str(ccd_ser).strip()

            # Get short camera ID for grouping/naming
            camera_id = get_camera_id(header)

            return {
                'file': fits_path,
                'exptime': exp_time,
                'sigma': sigma,
                'median': median_val,
                'average': avg_val,
                'ma_diff': avg_val - median_val,
                'ccd_temp': header.get('CCD_TEMP', None),
                'ccd_set': header.get('CCD_SET', None),
                'binning': header.get('BINNING', '1x1'),
                'filter': header.get('FILTER', None),
                'imagetyp': header.get('IMAGETYP', None),
                'ccd_ser': ccd_ser,       # raw value for noise model
                'camera_id': camera_id,   # short ID for grouping/naming
                'naxis1': data.shape[1],  # width
                'naxis2': data.shape[0],  # height
            }
    except Exception as e:
        # Can't call log() from subprocess, just return None
        return None


def compute_stats_parallel(files: list, image_type: str, n_workers: int = None,
                           progress_prefix: str = "  ") -> list:
    """
    Compute image statistics in parallel using multiple processes.

    Uses a persistent SQLite cache to avoid recomputing stats for files
    that have already been processed.
    """
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, MAX_WORKERS)  # cap at 16 to avoid overwhelming I/O

    n_files = len(files)
    cache = get_stats_cache()

    # Phase 1: Batch cache lookup (one query instead of N)
    log(f"Checking cache for {n_files} files...", "debug")
    cached_map = cache.get_many(files)

    results = []
    to_compute = []

    wrong_type_count = 0
    for f in files:
        cached = cached_map.get(f)
        if cached and cached.get('imagetyp') == image_type:
            results.append(cached)
        elif cached:
            # Cached but wrong imagetyp - don't recompute, just skip
            wrong_type_count += 1
        else:
            to_compute.append(f)

    if wrong_type_count > 0:
        log(f"Skipped {wrong_type_count} files with wrong IMAGETYP (not '{image_type}')", "debug")

    n_cached = len(results)
    n_to_compute = len(to_compute)

    if n_cached > 0:
        log(f"Found {n_cached}/{n_files} in cache, computing {n_to_compute}", "info")

    if n_to_compute == 0:
        progress_bar(n_files, n_files, prefix=f"{progress_prefix}{image_type}: ")
        return results

    # Phase 2: Compute missing stats in parallel
    log(f"Using {n_workers} parallel workers", "info")
    completed = 0
    newly_computed = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit jobs only for files not in cache
        future_to_file = {executor.submit(compute_image_stats, f): f for f in to_compute}

        # Collect results as they complete
        for future in as_completed(future_to_file):
            completed += 1
            # Progress shows total (cached + computed)
            if completed % 10 == 0 or completed == n_to_compute:
                progress_bar(n_cached + completed, n_files, prefix=f"{progress_prefix}{image_type}: ")

            try:
                stats = future.result()
                if stats:
                    newly_computed.append(stats)
                    if stats.get('imagetyp') == image_type:
                        results.append(stats)
            except Exception as e:
                log(f"Error processing file: {e}", "debug")

    # Phase 3: Save newly computed stats to cache
    if newly_computed:
        cache.put_batch(newly_computed)
        log(f"Cached {len(newly_computed)} new stats", "debug")

    return results


def filter_darks(stats_list: list, config: CalibConfig, camera: str) -> list:
    """Filter dark frames based on telescope/camera-specific criteria."""
    if camera not in config.dark_filter:
        log(f"No dark filter defined for camera {camera}, using all frames", "warn")
        return stats_list

    n_input = len(stats_list)
    n_incomplete = sum(1 for s in stats_list
                       if s.get('sigma') is None or s.get('median') is None)
    if n_incomplete > 0:
        log(f"Skipping {n_incomplete}/{n_input} frames with incomplete stats", "debug")

    filt = config.dark_filter[camera]
    filtered = []

    # Debug: show actual value ranges to help tune filters
    valid_stats = [s for s in stats_list if s and s.get('median') is not None and s.get('sigma') is not None]
    if valid_stats:
        medians = [s['median'] for s in valid_stats]
        sigmas = [s['sigma'] for s in valid_stats]
        log(f"Input stats: median={min(medians):.0f}-{max(medians):.0f}, "
            f"sigma={min(sigmas):.1f}-{max(sigmas):.1f}", "debug")
        log(f"Filter: {filt}", "debug")

    for s in stats_list:
        if s is None:
            continue

        # Check temperature stabilization (configurable tolerance)
        if s['ccd_temp'] is not None and s['ccd_set'] is not None:
            temp_diff = abs(s['ccd_temp'] - s['ccd_set'])
            if temp_diff > config.temp_tolerance:
                log(f"Skipping {s['file']} - temperature not stabilized ({s['ccd_temp']:.1f} vs {s['ccd_set']:.1f})", "debug")
                continue

        # Skip entries with missing stats
        if s['sigma'] is None or s['median'] is None or s['average'] is None:
            continue

        # Apply sigma filter
        if 'max_sigma' in filt and s['sigma'] > filt['max_sigma']:
            continue

        # Apply median filters
        if 'min_median' in filt and filt['min_median'] is not None and s['median'] < filt['min_median']:
            continue
        if 'max_median' in filt and filt['max_median'] is not None and s['median'] > filt['max_median']:
            continue

        # D50-specific: median-average difference check
        if 'median_diff_max' in filt:
            diff = abs(s['median'] - s['average'])
            if diff > filt['median_diff_max']:
                continue

        filtered.append(s)

    return filtered


def filter_flats(stats_list: list, config: CalibConfig, camera: str) -> list:
    """Filter flat frames based on telescope/camera-specific criteria."""
    if camera not in config.flat_filter:
        log(f"No flat filter defined for camera {camera}, using all frames", "warn")
        return stats_list

    filt = config.flat_filter[camera]
    filtered = []

    for s in stats_list:
        if s is None:
            continue

        # Apply sigma filter (min for flats - need good signal)
        if 'min_sigma' in filt and s['sigma'] < filt['min_sigma']:
            continue

        # Apply median filters
        if 'min_median' in filt and s['median'] < filt['min_median']:
            continue
        if 'max_median' in filt and s['median'] > filt['max_median']:
            continue

        filtered.append(s)

    return filtered


def group_darks(stats_list: list) -> dict:
    """Group dark frames by temperature, exposure time, binning, and frame size."""
    groups = defaultdict(list)

    for s in stats_list:
        if s is None:
            continue

        temp = s['ccd_temp']
        if temp is not None:
            # Round to nearest 5 degrees (e.g., -31.2 -> -30, -33 -> -35)
            temp = 5 * round(temp / 5)

        # Include frame dimensions to avoid mixing different sensor modes
        size = (s.get('naxis1', 0), s.get('naxis2', 0))
        key = (temp, s['exptime'], s['binning'], size)
        groups[key].append(s)

    return dict(groups)


def group_flats(stats_list: list) -> dict:
    """Group flat frames by filter and binning."""
    groups = defaultdict(list)

    for s in stats_list:
        if s is None:
            continue

        key = (s['filter'], s['binning'])
        groups[key].append(s)

    return dict(groups)


def validate_dark_pair(file1: str, file2: str, expected_sigma: float, tolerance: float) -> tuple:
    """
    Validate a pair of darks by checking the sigma of their difference.

    Returns: (is_valid, measured_sigma)
    """
    try:
        with fits.open(file1) as h1, fits.open(file2) as h2:
            data1 = image_hdu(h1).data.astype(np.float32)
            data2 = image_hdu(h2).data.astype(np.float32)

            diff = data1 - data2
            row_stds = np.array([np.std(row) for row in diff])
            measured_sigma = np.median(row_stds)

            # Check if measured sigma is within expected range
            # Lower bound: reject near-zero (saturated/dead images). Floor is
            # model-relative (half the expected sigma), not a fixed constant:
            # low-noise cameras like the BOOTES-2 iXon have real darks at ~1.9
            # ADU, which a hardcoded 2.7 floor would wrongly reject.
            # Upper bound: reject images with light leaks / bad reset
            lower = max(expected_sigma - 10 * tolerance, 0.5 * expected_sigma)
            upper = expected_sigma + 2 * tolerance

            is_valid = lower < measured_sigma < upper
            return is_valid, measured_sigma
    except Exception as e:
        log(f"Error validating dark pair: {e}", "error")
        return False, -1.0


def get_camera_noise_params(camera_id: str, binning: str, exptime: float, temp: float) -> tuple:
    """Get expected noise parameters for a camera (by camera_id from registry)."""
    model = CAMERA_NOISE_MODELS.get(camera_id, DEFAULT_NOISE_MODEL)
    bin_model = model.get(binning, model.get("1x1", DEFAULT_NOISE_MODEL["1x1"]))

    return bin_model(exptime, temp)


def select_valid_darks(dark_group: list, validate: bool = True) -> list:
    """
    Select valid darks from a group using sequential consensus validation.

    Algorithm: O(N) instead of O(N²)
    1. Bootstrap: build initial "golden set" of 2+ mutually-validating darks
    2. Validate: each remaining dark is compared against 2 random golden members
    3. Re-check: suspects get a second chance against different golden members

    This works because:
    - good × good = pass (both have expected read noise)
    - good × bad = fail (light leak / bad reset raises sigma)
    - bad × bad = fail (both have excess signal)

    With ~99% good darks, bootstrap succeeds quickly and we catch the ~1% bad ones.
    """
    if not validate or len(dark_group) < 2:
        return [s['file'] for s in dark_group]

    # Get camera and exposure info from first frame
    first = dark_group[0]
    camera_id = first['camera_id']  # Use camera_id for noise model lookup
    binning = first['binning']
    exptime = first['exptime']
    temp = first['ccd_temp'] or -30

    expected_sigma, tolerance = get_camera_noise_params(camera_id, binning, exptime, temp)

    log(f"  Noise model: camera={camera_id}, exp={exptime}, expected_sigma={expected_sigma:.2f} ± {tolerance:.2f}", "debug")

    files = [s['file'] for s in dark_group]
    n_files = len(files)

    # Shuffle to avoid systematic bias (e.g., if bad darks are clustered in time)
    shuffled = files.copy()
    random.shuffle(shuffled)

    validated = []  # The "golden set" of known-good darks
    suspects = []   # Darks that failed initial validation
    comparisons = 0 # Counter for logging
    measured_sigmas = []  # Collect measured sigmas for diagnostics

    # Phase 1: Bootstrap - build initial golden set
    # We need at least 2 mutually-validating darks to start
    bootstrap_attempts = 0
    max_bootstrap_attempts = min(20, n_files)  # Don't try forever

    for dark in shuffled:
        if len(validated) >= 2:
            # Golden set established, move to phase 2
            break

        bootstrap_attempts += 1
        if bootstrap_attempts > max_bootstrap_attempts:
            break

        if len(validated) == 0:
            # First dark - tentatively accept
            validated.append(dark)
        else:
            # Must validate against all current golden members
            all_pass = True
            for golden in validated:
                comparisons += 1
                is_valid, sigma = validate_dark_pair(dark, golden, expected_sigma, tolerance)
                measured_sigmas.append(sigma)
                mark = "*" if is_valid else "x"
                log(f"    {mark} {Path(dark).name} vs {Path(golden).name}: "
                    f"sigma={sigma:.2f} (expect {expected_sigma:.2f}±{tolerance:.2f})", "debug")
                if not is_valid:
                    all_pass = False
                    break

            if all_pass:
                validated.append(dark)
            else:
                suspects.append(dark)

    # If bootstrap failed, show diagnostic info and SKIP this group
    if len(validated) < 2:
        valid_sigmas = [s for s in measured_sigmas if s > 0]
        if valid_sigmas:
            log(f"Bootstrap FAILED for group (camera={camera_id}, exp={exptime}). "
                f"Measured sigmas: min={min(valid_sigmas):.2f}, max={max(valid_sigmas):.2f}, "
                f"median={sorted(valid_sigmas)[len(valid_sigmas)//2]:.2f}. "
                f"Expected: {expected_sigma:.2f}±{tolerance:.2f}", "warn")
            log(f"  -> Add noise model for {camera_id} with sigma ~{min(valid_sigmas):.1f}, or use --no-validate", "warn")
        else:
            log(f"Bootstrap FAILED for group (camera={camera_id}, exp={exptime}) - "
                f"no valid comparisons. Check input files.", "warn")
        return []  # Return empty - don't create master from unvalidated darks

    # Phase 2: Validate remaining darks against golden set (parallel)
    # Fix golden set size - compare each remaining dark against 2 golden members
    remaining = [f for f in shuffled if f not in validated and f not in suspects]
    n_remaining = len(remaining)
    golden_set = validated.copy()  # Fixed size for parallel comparison
    sample_size = min(2, len(golden_set))
    n_workers = min(os.cpu_count() or 1, MAX_WORKERS)  # Cap workers for I/O bound task

    if n_remaining > 0 and sample_size > 0:
        # Generate all comparison pairs upfront
        comparison_pairs = []
        for dark in remaining:
            sample = random.sample(golden_set, sample_size)
            for golden in sample:
                comparison_pairs.append((dark, golden, expected_sigma, tolerance))

        n_comparisons = len(comparison_pairs)
        # Run comparisons in parallel
        results_map = {}  # dark -> list of (is_valid, sigma)

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(validate_dark_pair, *args): args[0]
                       for args in comparison_pairs}

            completed = 0
            for future in as_completed(futures):
                dark = futures[future]
                is_valid, sigma = future.result()
                if dark not in results_map:
                    results_map[dark] = []
                results_map[dark].append((is_valid, sigma))
                measured_sigmas.append(sigma)
                completed += 1
                progress_bar(completed, n_comparisons, prefix="    validating: ")

        comparisons += n_comparisons

        # Evaluate results
        for dark in remaining:
            results = results_map.get(dark, [])
            passes = sum(1 for is_valid, _ in results if is_valid)
            if passes == sample_size:
                validated.append(dark)
            else:
                suspects.append(dark)

    # Phase 3: Re-check suspects (parallel)
    # Give suspects another chance against different golden members
    still_suspect = []

    if len(validated) < 3:
        # Not enough golden darks for meaningful re-check
        still_suspect = suspects
    elif suspects:
        # Generate comparison pairs for all suspects
        recheck_pairs = []
        for dark in suspects:
            sample = random.sample(validated, min(3, len(validated)))
            for golden in sample:
                recheck_pairs.append((dark, golden, expected_sigma, tolerance))

        n_recheck = len(recheck_pairs)
        # Run in parallel
        recheck_results = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(validate_dark_pair, *args): args[0]
                       for args in recheck_pairs}
            completed = 0
            for future in as_completed(futures):
                dark = futures[future]
                is_valid, sigma = future.result()
                if dark not in recheck_results:
                    recheck_results[dark] = []
                recheck_results[dark].append((is_valid, sigma))
                measured_sigmas.append(sigma)
                completed += 1
                progress_bar(completed, n_recheck, prefix="    rechecking: ")

        comparisons += n_recheck

        # Evaluate: accept if majority pass (2 out of 3)
        for dark in suspects:
            results = recheck_results.get(dark, [])
            passes = sum(1 for is_valid, _ in results if is_valid)
            if passes >= 2:
                validated.append(dark)
                log(f"Recovered suspect dark on re-check: {Path(dark).name}", "debug")
            else:
                still_suspect.append(dark)

    # Log results - show summary at info level, sigma stats at debug
    valid_sigmas = [s for s in measured_sigmas if s > 0]
    if valid_sigmas:
        sigma_stats = f"sigma: min={min(valid_sigmas):.2f}, max={max(valid_sigmas):.2f}"
        median_sigma = sorted(valid_sigmas)[len(valid_sigmas) // 2]

        # Log noise data to file for future model refinement
        noise_file = Path(f"noise-{camera_id}.out")
        write_header = not noise_file.exists()
        with open(noise_file, "a") as f:
            if write_header:
                f.write("# Noise measurements for camera model fitting\n")
                f.write("# sigma = sqrt(readnoise^2 + dark_current * exptime * 2^(temp/T0))\n")
                f.write("# exptime\ttemp\tbinning\tmedian_sigma\tmin_sigma\tmax_sigma\tn_samples\n")
            f.write(f"{exptime:.1f}\t{temp:.1f}\t{binning}\t{median_sigma:.3f}\t"
                    f"{min(valid_sigmas):.3f}\t{max(valid_sigmas):.3f}\t{len(valid_sigmas)}\n")
    else:
        sigma_stats = "no valid measurements"

    log(f"  Validated {len(validated)}/{n_files} darks ({comparisons} comparisons), {sigma_stats}", "info")

    if still_suspect:
        log(f"  Rejected: {', '.join(Path(f).name for f in still_suspect[:5])}"
            f"{'...' if len(still_suspect) > 5 else ''}", "debug")

    if len(validated) < 2:
        log(f"Not enough valid darks in group (camera={camera_id}, exp={exptime}), skipping", "warn")
        return []  # Return empty - don't create master from unvalidated darks

    return validated


def find_script(name: str) -> Path:
    """Find a helper script, checking multiple locations."""
    # First try: same directory as this script (resolving symlinks)
    script_dir = Path(__file__).resolve().parent
    path = script_dir / name
    if path.exists():
        return path

    # Second try: /home/mates/pyrt/calib (hardcoded fallback)
    path = Path("/home/mates/pyrt/calib") / name
    if path.exists():
        return path

    # Third try: search in PATH
    import shutil
    found = shutil.which(name)
    if found:
        return Path(found)

    raise FileNotFoundError(f"Cannot find {name}. Looked in: {script_dir}, /home/mates/pyrt/calib, PATH")


def run_mixdark(output: str, input_files: list, maxparam: int = 150) -> bool:
    """Run mixdark.py to combine dark frames."""
    mixdark_path = find_script("mixdark.py")

    # mixdark.py handles chunking and parallelism internally
    cmd = ["python3", str(mixdark_path), output] + input_files
    log(f"Running: {cmd[0]} {cmd[1]} {cmd[2]} [{len(input_files)} files]", "debug")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown error"
        log(f"mixdark.py failed (exit {result.returncode}): {error_msg}", "error")
        return False

    if not os.path.exists(output):
        log(f"mixdark.py completed but output file not created: {output}", "error")
        if result.stdout:
            log(f"stdout: {result.stdout[:500]}", "debug")
        return False

    return True


def run_mixflat(output: str, input_files: list) -> bool:
    """Run mixflat.py to combine flat frames."""
    mixflat_path = find_script("mixflat.py")

    # mixflat.py handles chunking and parallelism internally
    cmd = ["python3", str(mixflat_path), output] + input_files
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown error"
        log(f"mixflat.py failed: {error_msg}", "error")
        return False

    return os.path.exists(output)


def create_master_darks(dark_groups: dict, output_dir: Path, validate: bool = True,
                        dry_run: bool = False) -> dict:
    """Create master dark frames from grouped darks."""
    master_darks = {}

    for key, group in dark_groups.items():
        # Unpack key - may be (temp, exp, bin) or (temp, exp, bin, size)
        if len(key) == 4:
            temp, exptime, binning, size = key
        else:
            temp, exptime, binning = key
            size = None

        if len(group) < 2:
            log(f"Skipping dark group (temp={temp}, exp={exptime}, bin={binning}) - only {len(group)} frames", "warn")
            continue

        # Include frame size in filename if not standard (to distinguish overscan modes)
        if size and size != (0, 0):
            size_suffix = f"-{size[0]}x{size[1]}"
        else:
            size_suffix = ""

        # Simpler filename - camera_id is already in the directory path
        output_name = f"dark{temp:+04d}-{exptime:05.1f}-{binning}{size_suffix}.fits"
        output_path = output_dir / output_name

        if output_path.exists():
            log(f"{output_name} already exists, skipping", "info")
            master_darks[(temp, exptime, binning, size)] = str(output_path)
            continue

        # Select valid darks
        valid_files = select_valid_darks(group, validate=validate)

        if len(valid_files) < 2:
            log(f"Not enough valid darks for {output_name}", "warn")
            continue

        log(f"Creating {output_name} from {len(valid_files)} frames", "info")

        if dry_run:
            log(f"  [DRY-RUN] Would combine: {', '.join(Path(f).name for f in valid_files[:3])}...", "info")
        else:
            if run_mixdark(str(output_path), valid_files):
                master_darks[(temp, exptime, binning, size)] = str(output_path)
                log(f"  Created {output_name}", "info")
            else:
                log(f"  Failed to create {output_name}", "error")

    return master_darks


def find_matching_dark(flat_stats: dict, master_darks: dict) -> Optional[str]:
    """Find the best matching master dark for a flat frame."""
    flat_temp = flat_stats['ccd_temp']
    if flat_temp is not None:
        flat_temp = 2 * int(flat_temp / 2 + 0.5) if flat_temp > 0 else 2 * int(flat_temp / 2 - 0.5)
    flat_exptime = flat_stats['exptime']
    flat_binning = flat_stats['binning']
    flat_size = (flat_stats.get('naxis1', 0), flat_stats.get('naxis2', 0))

    best_match = None
    best_dist = float('inf')

    for key, dark_path in master_darks.items():
        # Unpack key - may be (temp, exp, bin) or (temp, exp, bin, size)
        if len(key) == 4:
            temp, exptime, binning, size = key
        else:
            temp, exptime, binning = key
            size = None

        if binning != flat_binning:
            continue
        if temp != flat_temp:
            continue
        # Must match frame size
        if size is not None and size != flat_size:
            continue

        dist = abs(exptime - flat_exptime)
        if dist < best_dist:
            best_dist = dist
            best_match = dark_path

    return best_match


def subtract_dark(flat_file: str, dark_file: str, output_file: str) -> bool:
    """Subtract dark from flat (normalization is done by mixflat.sh)."""
    try:
        with fits.open(flat_file) as flat_hdu, fits.open(dark_file) as dark_hdu:
            flat_image = image_hdu(flat_hdu)
            flat_data = flat_image.data.astype(np.float32)
            dark_data = image_hdu(dark_hdu).data.astype(np.float32)
            header = flat_image.header.copy()

            # Subtract dark only - mixflat.sh handles normalization
            corrected = flat_data - dark_data

            # Save
            hdu = fits.PrimaryHDU(data=corrected, header=header)
            hdu.writeto(output_file, overwrite=True)

            return True
    except Exception as e:
        log(f"Error processing flat {flat_file}: {e}", "error")
        return False


def create_master_flats(flat_groups: dict, master_darks: dict, output_dir: Path,
                        dry_run: bool = False) -> dict:
    """Create master flat frames from grouped flats."""
    master_flats = {}

    for (filter_name, binning), group in flat_groups.items():
        if len(group) < 1:
            continue

        output_name = f"flat-{filter_name}-{binning}.fits"
        output_path = output_dir / output_name

        if output_path.exists():
            log(f"{output_name} already exists, skipping", "info")
            master_flats[(filter_name, binning)] = str(output_path)
            continue

        # For each flat, find matching dark and prepare dark-subtracted version
        flats_with_darks = []
        for s in group:
            dark = find_matching_dark(s, master_darks)
            if dark:
                flats_with_darks.append((s['file'], dark))
            else:
                log(f"No matching dark for {s['file']}", "warn")

        if not flats_with_darks:
            log(f"No flats with matching darks for {output_name}", "warn")
            continue

        log(f"Creating {output_name} from {len(flats_with_darks)} frames", "info")

        if dry_run:
            log(f"  [DRY-RUN] Would combine flats with their darks", "info")
            continue

        # Create temporary dark-subtracted flats (use ~/tmp for space)
        user_tmp = Path.home() / "tmp"
        user_tmp.mkdir(exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="calib_", dir=user_tmp))
        temp_files = []

        try:
            for i, (flat, dark) in enumerate(flats_with_darks):
                temp_file = temp_dir / f"temp_flat_{i:03d}.fits"
                if subtract_dark(flat, dark, str(temp_file)):
                    temp_files.append(str(temp_file))

            if len(temp_files) >= 1:
                if run_mixflat(str(output_path), temp_files):
                    master_flats[(filter_name, binning)] = str(output_path)
                    log(f"  Created {output_name}", "info")
                else:
                    log(f"  Failed to create {output_name}", "error")
        finally:
            # Clean up temp files
            shutil.rmtree(temp_dir, ignore_errors=True)

    return master_flats


def group_by_camera_id(stats_list: list) -> dict:
    """Group stats by camera_id."""
    groups = defaultdict(list)
    for s in stats_list:
        if s is None:
            continue
        camera_id = s.get('camera_id', 'unknown')
        groups[camera_id].append(s)
    return dict(groups)


def filter_darks_generic(stats_list: list, config: CalibConfig) -> list:
    """Filter dark frames using camera_id-specific criteria.

    Logs the input value distribution and a per-reason breakdown of rejections,
    so the filter/noise model for a new camera can be tuned from real numbers
    instead of guesswork.
    """
    # Permissive defaults for unknown cameras - let validation catch bad frames
    default_filter = {"max_sigma": 50, "min_median": None, "max_median": None}

    # Per-reason rejection counters for diagnostics
    drop = defaultdict(int)
    filtered = []

    # Report the input distribution up front (per camera_id), so an operator can
    # see where to put thresholds without querying the stats cache by hand.
    valid = [s for s in stats_list
             if s and s.get('sigma') is not None and s.get('median') is not None]
    for cam in sorted(set(s.get('camera_id', 'unknown') for s in valid)):
        cv = [s for s in valid if s.get('camera_id', 'unknown') == cam]
        def pct(key, q):
            xs = sorted(s[key] for s in cv if s.get(key) is not None)
            return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float('nan')
        td = [abs(s['ccd_temp'] - s['ccd_set']) for s in cv
              if s.get('ccd_temp') is not None and s.get('ccd_set') is not None]
        td_med = sorted(td)[len(td) // 2] if td else float('nan')
        log(f"  {cam}: n={len(cv)} "
            f"median[p10/50/90]={pct('median',.1):.0f}/{pct('median',.5):.0f}/{pct('median',.9):.0f} "
            f"sigma[p10/50/90]={pct('sigma',.1):.2f}/{pct('sigma',.5):.2f}/{pct('sigma',.9):.2f} "
            f"|temp-set|[med]={td_med:.2f} (tol={config.temp_tolerance})", "info")
        log(f"  {cam}: dark filter = {config.dark_filter.get(cam, default_filter)}", "info")

    for s in stats_list:
        if s is None:
            continue

        # Get filter for this camera_id, fall back to default
        camera_id = s.get('camera_id', 'unknown')
        filt = config.dark_filter.get(camera_id, default_filter)

        # Check temperature stabilization (configurable tolerance)
        if s['ccd_temp'] is not None and s['ccd_set'] is not None:
            if abs(s['ccd_temp'] - s['ccd_set']) > config.temp_tolerance:
                drop['temp_unstable'] += 1
                continue

        if s['sigma'] is None or s['median'] is None or s['average'] is None:
            drop['missing_stats'] += 1
            continue

        if 'max_sigma' in filt and s['sigma'] > filt['max_sigma']:
            drop['sigma_high'] += 1
            continue
        if 'min_median' in filt and filt['min_median'] is not None:
            if s['median'] < filt['min_median']:
                drop['median_low'] += 1
                continue
        if 'max_median' in filt and filt['max_median'] is not None:
            if s['median'] > filt['max_median']:
                drop['median_high'] += 1
                continue

        filtered.append(s)

    if drop:
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(drop.items()))
        log(f"  Dark rejections: {breakdown}", "info")

    return filtered


def filter_flats_generic(stats_list: list, config: CalibConfig) -> list:
    """Filter flat frames using camera_id-specific criteria."""
    # Permissive defaults for unknown cameras
    default_filter = {"min_median": 5000, "max_median": 50000}

    filtered = []
    for s in stats_list:
        if s is None:
            continue

        # Get filter for this camera_id, fall back to default
        camera_id = s.get('camera_id', 'unknown')
        filt = config.flat_filter.get(camera_id, default_filter)

        if 'min_sigma' in filt and s['sigma'] < filt['min_sigma']:
            continue
        if 'min_median' in filt and s['median'] < filt['min_median']:
            continue
        if 'max_median' in filt and s['median'] > filt['max_median']:
            continue

        filtered.append(s)

    return filtered


def process_calibrations(config: CalibConfig, year: str, camera: Optional[str] = None,
                         output_base: Optional[str] = None, validate_darks: bool = True,
                         dry_run: bool = False, skip_flats: bool = False,
                         n_workers: int = None, stats_only: bool = False,
                         layout: str = "auto"):
    """Process calibration frames.

    Find ALL files in archive, group by physical camera_id,
    output to {camera_id}/{year}/.
    """

    log(f"Processing calibrations for {year} from {config.archive_base}", "info")

    if layout == "auto":
        layout = detect_layout(config.archive_base, year)
    log(f"Archive layout: {layout}-camera", "info")

    # Show cache info
    cache = get_stats_cache()
    cache_stats = cache.stats()
    log(f"Stats cache: {cache_stats['entries']} entries at {cache_stats['path']}", "debug")

    # Setup base output directory
    if output_base:
        output_dir = Path(output_base)
    else:
        output_dir = Path.cwd()

    # --- FIND ALL DARK FILES ---
    log(f"\nFinding dark frames across all archive cameras...", "info")
    dark_files = find_calibration_files(config.archive_base, year, camera, "dark", layout=layout)
    log(f"Found {len(dark_files)} dark files", "info")

    all_dark_stats = []
    if dark_files:
        log(f"Computing dark statistics (cached results reused)...", "info")
        all_dark_stats = compute_stats_parallel(dark_files, "dark", n_workers=n_workers)
        log(f"Valid dark frames: {len(all_dark_stats)}", "info")

    # --- FIND ALL FLAT FILES ---
    flat_files = []
    all_flat_stats = []
    if not skip_flats:
        log(f"\nFinding flat frames across all archive cameras...", "info")
        flat_files = find_calibration_files(config.archive_base, year, camera, "flat", layout=layout)
        log(f"Found {len(flat_files)} flat files", "info")

        if flat_files:
            log(f"Computing flat statistics (cached results reused)...", "info")
            all_flat_stats = compute_stats_parallel(flat_files, "flat", n_workers=n_workers)
            log(f"Valid flat frames: {len(all_flat_stats)}", "info")

    if stats_only:
        log("Stats-only mode, done.", "info")
        return

    # --- GROUP BY CAMERA_ID ---
    dark_by_camera = group_by_camera_id(all_dark_stats)
    flat_by_camera = group_by_camera_id(all_flat_stats)

    # Get all unique camera IDs
    all_camera_ids = sorted(set(dark_by_camera.keys()) | set(flat_by_camera.keys()))
    log(f"\nDiscovered cameras: {', '.join(all_camera_ids)}", "info")

    # --- PROCESS EACH CAMERA ---
    for camera_id in all_camera_ids:
        log(f"\n{'='*60}", "info")
        log(f"Processing camera: {camera_id}", "info")
        log(f"{'='*60}", "info")

        # Output directory: {year}/{camera_id}/
        # Structure: {camera_id}/{year}/ - groups all years for a camera together
        final_output = output_dir / camera_id / year

        if not dry_run:
            final_output.mkdir(parents=True, exist_ok=True)

        log(f"Output directory: {final_output}", "info")

        # --- DARK FRAMES ---
        dark_stats = dark_by_camera.get(camera_id, [])
        log(f"Dark frames for {camera_id}: {len(dark_stats)}", "info")

        if dark_stats:
            # Filter darks (generic filter for all cameras)
            filtered_darks = filter_darks_generic(dark_stats, config)
            log(f"After filtering: {len(filtered_darks)} darks", "info")

            # Group darks by temp/exp/bin
            dark_groups = group_darks(filtered_darks)
            log(f"Dark groups (by temp/exp/bin): {len(dark_groups)}", "info")

            # Create master darks
            master_darks = create_master_darks(dark_groups, final_output,
                                               validate=validate_darks, dry_run=dry_run)
            log(f"Created {len(master_darks)} master darks", "info")
        else:
            master_darks = {}
            log("No dark files for this camera", "warn")

        if skip_flats:
            continue

        # --- FLAT FRAMES ---
        flat_stats = flat_by_camera.get(camera_id, [])
        log(f"Flat frames for {camera_id}: {len(flat_stats)}", "info")

        if flat_stats and master_darks:
            # Filter flats
            filtered_flats = filter_flats_generic(flat_stats, config)
            log(f"After filtering: {len(filtered_flats)} flats", "info")

            # Group flats by filter/bin
            flat_groups = group_flats(filtered_flats)
            log(f"Flat groups (by filter/bin): {len(flat_groups)}", "info")

            # Create master flats
            master_flats = create_master_flats(flat_groups, master_darks, final_output,
                                               dry_run=dry_run)
            log(f"Created {len(master_flats)} master flats", "info")
        elif not master_darks:
            log("Skipping flats - no master darks available", "warn")
        else:
            log("No flat files for this camera", "warn")


def main():
    parser = argparse.ArgumentParser(
        description="Automatic calibration frame processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full processing (stats are cached automatically in SQLite)
  %(prog)s -y 2024 -j 24                           # First run computes & caches
  %(prog)s -y 2024                                 # Reruns use cached stats
  %(prog)s -y 2024 --archive /storage/archive      # Custom archive path

  # Just populate the cache without creating master frames
  %(prog)s -y 2024 --stats-only -j 24

  # Discovery
  %(prog)s --list-years
  %(prog)s --year 2025 --list-cameras

  # Cache management (~/.cache/pyrt/stats_cache.db)
  %(prog)s --cache-stats                           # Show cache info
  %(prog)s --clear-cache                           # Clear the cache
        """
    )
    parser.add_argument("--year", "-y", type=str,
                        help="Year to process (e.g., 2024)")
    parser.add_argument("--camera", "-c", type=str,
                        help="Specific camera to process (e.g., C0, C2)")
    parser.add_argument("--output", "-o", type=str,
                        help="Base output directory (default: current directory)")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Show what would be done without actually doing it")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip dark validation (use all darks)")
    parser.add_argument("--skip-flats", action="store_true",
                        help="Only process darks, skip flats")
    parser.add_argument("--list-years", action="store_true",
                        help="List available years in archive and exit")
    parser.add_argument("--list-cameras", action="store_true",
                        help="List available cameras for the specified year and exit")
    parser.add_argument("--archive", "-a", type=str,
                        help="Archive base path (default: /archive)")
    parser.add_argument("--layout", type=str, default="auto",
                        choices=["auto", "single", "multi"],
                        help="Archive layout: 'multi' has a camera dir between "
                             "night and calib, 'single' does not. Default: auto-detect.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed debug output")
    parser.add_argument("--workers", "-j", type=int, default=None,
                        help="Number of parallel workers (default: auto, max 16)")
    parser.add_argument("--stats-only", "-s", action="store_true",
                        help="Only compute statistics (populate cache), don't create masters")
    parser.add_argument("--cache-stats", action="store_true",
                        help="Show cache statistics and exit")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear the statistics cache and exit")

    args = parser.parse_args()

    # Set global verbose flag
    global VERBOSE
    VERBOSE = args.verbose

    # Use default config, override archive if specified
    config = DEFAULT_CONFIG
    if args.archive:
        config.archive_base = args.archive

    # List years mode
    if args.list_years:
        years = discover_years(config.archive_base)
        if years:
            print(f"Available years in {config.archive_base}:")
            for y in years:
                print(f"  {y}")
        else:
            print(f"No years found in {config.archive_base}")
        return

    # List cameras mode
    if args.list_cameras:
        if not args.year:
            print("Error: --year is required with --list-cameras")
            sys.exit(1)
        cameras = discover_cameras(config.archive_base, args.year, layout=args.layout)
        if cameras:
            print(f"Available cameras in {args.year}:")
            for c in cameras:
                print(f"  {c}")
        else:
            print(f"No cameras found for {args.year}")
        return

    # Cache management
    cache = get_stats_cache()
    if args.cache_stats:
        stats = cache.stats()
        print(f"Cache location: {stats['path']}")
        print(f"Cached entries: {stats['entries']}")
        return

    if args.clear_cache:
        cache_path = cache.db_path
        if cache_path.exists():
            os.unlink(cache_path)
            print(f"Cleared cache: {cache_path}")
        else:
            print("Cache is already empty")
        return

    # Require year for processing
    if not args.year:
        print("Error: --year is required for processing")
        sys.exit(1)

    # Process calibrations
    process_calibrations(
        config=config,
        year=args.year,
        camera=args.camera,
        output_base=args.output,
        validate_darks=not args.no_validate,
        dry_run=args.dry_run,
        skip_flats=args.skip_flats,
        n_workers=args.workers,
        stats_only=args.stats_only,
        layout=args.layout,
    )

    log("\nDone!", "info")


if __name__ == "__main__":
    main()
