#!/usr/bin/env python3
"""
RTS2 image processor — real-time astrometric and photometric calibration.

Invoked by the RTS2 imgproc daemon with the raw image path as the sole argument.
Outputs a single corrwerr line to stdout for RTS2; everything else goes to stderr.

Sequence:
  0. (SBT) patch missing LTV1/2 windowing keywords
  1. dark/flat correct raw image
  2. generate web preview from calibrated image
  3. phcat + dophot solve pass on calibrated image
  4. copy WCS from calibrated back into raw image (then release raw)
  5. print corrwerr to stdout  ← RTS2 archives the raw image after this
  6. dophot photometry pass
  7. save ECSV/PNG results
  8. upload ECSV to database server
  9. notify transient daemon
  10. locate archived raw by night/camera/filename, update its header
"""

import os
import glob
import logging
import math
from typing import Optional
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

warnings.filterwarnings('ignore', category=FITSFixedWarning)

from asarina.pipeline.get_ecsv import PhotometryPipeline
from asarina.pipeline.pipeline_utils import TransientSearcher
from asarina.pipeline.c0_pipeline import DatabaseUploader
from asarina.pipeline.proc_images import CAMERA_CROPS
from asarina.pipeline.patch_window import compute_keywords, KNOWN_WINDOW_SIZES
from asarina.chip_id import get_camera_id

# Logging is configured in main() after -v is parsed.
# stdout is reserved for RTS2 (corrwerr).
logger = logging.getLogger(__name__)

# WCS keywords to propagate from calibrated image to raw image.
# CRPIX1/2 are handled separately (crop offset reversal).
# CDELT*/CROTA2 (old-style) and RADESYS/EQUINOX are intentionally excluded:
# the CD matrix is sufficient, and the others are either already correct in
# the raw header or could conflict with the CD matrix.
_WCS_KEYWORDS = [
    'CTYPE1', 'CTYPE2',
    'CRVAL1', 'CRVAL2',
    'CD1_1',  'CD1_2',
    'CD2_1',  'CD2_2',
]
# SIP distortion keywords (A_*, B_*, AP_*, BP_*, A_ORDER, B_ORDER, …)
_SIP_PREFIXES = ('A_', 'B_', 'AP_', 'BP_')


def _angular_separation_deg(ra1: float, dec1: float,
                             ra2: float, dec2: float) -> float:
    """Angular separation in degrees (all inputs in degrees). Haversine formula."""
    r1, d1, r2, d2 = (math.radians(x) for x in (ra1, dec1, ra2, dec2))
    a = (math.sin((d2 - d1) / 2) ** 2
         + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2) ** 2)
    return math.degrees(2 * math.asin(math.sqrt(a)))


def _night_id(unix_time: float):
    """Return (year_str, night_str) for the RTS2 %Y/%N path components.

    Night is the date of the last solar meridian passage before the
    observation: date(UT - 12 h), formatted YYYYMMDD.
    """
    dt = datetime.fromtimestamp(unix_time - 43200, tz=timezone.utc)
    return dt.strftime('%Y'), dt.strftime('%Y%m%d')


def _copy_wcs_to_raw(calibrated_path: Path, raw_path: Path, chip_id: str,
                     ecsv_path: Optional[Path] = None) -> bool:
    """Copy WCS from calibrated image back into the raw image.

    The calibrated image may have been cropped; CRPIX values are adjusted
    back to raw-image coordinates using the known crop offset.

    ecsv_path, if given, is read for ASTSIGMA/IDNUM quality checks.
    These keywords live in the ECSV metadata, not the FITS header.
    """
    from astropy.table import Table
    crop = CAMERA_CROPS.get(chip_id)
    col_offset = crop[1].start if crop is not None else 0
    row_offset = crop[0].start if crop is not None else 0

    try:
        with fits.open(str(calibrated_path)) as cal:
            cal_hdr = cal[0].header

        # Validate before touching anything: require a complete, sane solution.
        for required in ('CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
                         'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'CTYPE1', 'CTYPE2'):
            if required not in cal_hdr:
                logger.warning(f"WCS copy skipped: {required} missing from solution")
                return False
        ra = float(cal_hdr['CRVAL1'])
        dec = float(cal_hdr['CRVAL2'])
        if not (0 <= ra < 360 and -90 <= dec <= 90):
            logger.warning(f"WCS copy skipped: CRVAL out of range (ra={ra}, dec={dec})")
            return False

        # Quality check from ECSV metadata (authoritative source).
        # ASTSIGMA and IDNUM (matched-star count) live there, not in the FITS header.
        if ecsv_path is not None and ecsv_path.exists():
            meta = Table.read(str(ecsv_path), format='ascii.ecsv').meta
            astsigma = meta.get('ASTSIGMA')
            idnum    = meta.get('IDNUM')
            if astsigma is None:
                logger.warning("WCS copy skipped: ASTSIGMA missing from ECSV")
                return False
            if float(astsigma) >= 0.5:
                logger.warning(f"WCS copy skipped: ASTSIGMA={astsigma:.3f} >= 0.5")
                return False
            if idnum is None or int(idnum) <= 20:
                logger.warning(f"WCS copy skipped: IDNUM={idnum} <= 20")
                return False
            logger.debug(f"WCS quality ok: ASTSIGMA={astsigma:.3f} IDNUM={idnum}")

        with fits.open(str(raw_path), mode='update') as raw:
            hdr = raw[0].header

            for kw in _WCS_KEYWORDS:
                if kw in cal_hdr:
                    hdr[kw] = cal_hdr[kw]

            # CRPIX: reverse the crop offset back to raw-image coordinates
            hdr['CRPIX1'] = cal_hdr['CRPIX1'] + col_offset
            hdr['CRPIX2'] = cal_hdr['CRPIX2'] + row_offset

            # SIP distortion coefficients (order-independent copy)
            for key in cal_hdr:
                if any(key.startswith(p) for p in _SIP_PREFIXES):
                    hdr[key] = cal_hdr[key]

            raw.flush()

        logger.info(f"WCS written to raw image {raw_path.name}")
        return True

    except Exception as e:
        logger.error(f"Failed to copy WCS to raw image: {e}")
        return False


def _report_fwhm(cat_path: Path, ccd_name: str) -> None:
    """Read FWHM from the phcat catalog and report it via RTS2 scriptcomm.

    cat_path is the ECSV catalog produced by pyrt-phcat ({base}.cat).
    The FWHM value is stored in the table metadata under key 'FWHM'.
    """
    try:
        from astropy.table import Table
        import rts2
        meta = Table.read(str(cat_path), format='ascii.ecsv').meta
        fwhm = meta.get('FWHM')
        if fwhm is None:
            logger.warning("FWHM not found in phcat catalog metadata")
            return
        fwhm = float(fwhm)
        rts2.Rts2Comm().doubleValue(f'fwhm_{ccd_name}', 'calculated FWHM', fwhm)
        logger.info(f"FWHM reported: fwhm_{ccd_name}={fwhm:.2f} px")
    except Exception as e:
        logger.error(f"Failed to report FWHM: {e}")


def _corrwerr(calibrated_path: Path, raw_header: fits.Header,
              chip_id: str) -> None:
    """Compute and print the corrwerr line for RTS2.

    Format: corrwerr 1 ra dec ra_offset dec_offset angular_separation
    All values in degrees.  Printed to stdout and flushed immediately.
    """
    target_ra  = raw_header.get('OBJRA',  raw_header.get('RASC'))
    target_dec = raw_header.get('OBJDEC', raw_header.get('DECL'))
    if target_ra is None or target_dec is None:
        logger.warning("No target RA/Dec in header, skipping corrwerr")
        return

    naxis1 = raw_header.get('NAXIS1', 0)
    naxis2 = raw_header.get('NAXIS2', 0)
    slit_x = raw_header.get('slitposx', -1)
    slit_x = slit_x if slit_x >= 0 else naxis1 / 2
    slit_y = naxis2 / 2

    # Adjust slit position to calibrated (cropped) image coordinates
    crop = CAMERA_CROPS.get(chip_id)
    if crop is not None:
        slit_x -= crop[1].start
        slit_y -= crop[0].start

    try:
        with fits.open(str(calibrated_path)) as cal:
            wcs = WCS(cal[0].header)
        ra, dec = (float(v) for v in wcs.all_pix2world(slit_x, slit_y, 0))
    except Exception as e:
        logger.error(f"WCS evaluation failed: {e}")
        return

    sep = _angular_separation_deg(target_ra, target_dec, ra, dec)
    print(f"corrwerr 1 {ra:.10f} {dec:.10f} "
          f"{target_ra - ra:.10f} {target_dec - dec:.10f} {sep:.10f}",
          flush=True)
    logger.info(f"corrwerr: ra={ra:.6f} dec={dec:.6f} sep={sep*3600:.1f}\"")


def _make_web_image(calibrated_path: Path, ccd_name: str,
                    web_dir: str = "/var/www/info") -> None:
    """Generate web preview images from the calibrated FITS file.

    Replaces c0toweb.sh.  Uses the calibrated (dark/flat corrected) image
    for better quality.  Skips silently if another process holds the lock
    or if the cadence limit (3 s) has not elapsed.

    ccd_name is the RTS2 camera name (e.g. 'C0'), taken from CCD_NAME header.
    """
    lock_path = Path(f"/dev/shm/{ccd_name}.lock")
    temp_fits  = Path(f"/dev/shm/{ccd_name}.fits")
    web        = Path(web_dir)

    now = time.time()

    if lock_path.exists() and (now - lock_path.stat().st_mtime) < 60:
        logger.debug("web image: skipping, lock held by another process")
        return

    if temp_fits.exists() and (now - temp_fits.stat().st_mtime) < 3:
        logger.debug("web image: skipping, cadence limit")
        return

    lock_path.touch()
    try:
        shutil.copy2(str(calibrated_path), str(temp_fits))

        # Read dimensions from the calibrated image
        with fits.open(str(temp_fits)) as hdul:
            hdr = hdul[0].header
            naxis1 = hdr.get('NAXIS1', 1000)
            naxis2 = hdr.get('NAXIS2', 1000)

        full_jpg   = web / f"{ccd_name}_full.jpg"
        small_jpg  = web / f"{ccd_name}_small.jpg"
        center_jpg = web / f"{ccd_name}_center.jpg"
        info_txt   = web / f"{ccd_name}_info.txt"

        # FITS → JPEG  (%H:%M in the label is expanded by f2cj from DATE-OBS)
        ret = subprocess.run(
            ["f2cj", "--label", f"D50 {ccd_name} - %H:%M",
             "-o", str(full_jpg), "-i", str(temp_fits)],
            capture_output=True,
        )
        if ret.returncode != 0:
            logger.warning("f2cj failed, web image not updated")
            return

        # Info text (read keywords fresh from the calibrated header)
        with fits.open(str(temp_fits)) as hdul:
            h = hdul[0].header
            info_txt.write_text(
                f"{ccd_name}: {h.get('DATE-OBS', '')}, "
                f"ID {h.get('TARGET', '')}, {h.get('OBJECT', '')}, "
                f"{h.get('EXPOSURE', h.get('EXPTIME', ''))}s, "
                f"{h.get('FILTER', '')}\n"
                f"{calibrated_path}\n"
            )

        # Thumbnail (height 300, preserve aspect)
        subprocess.run(
            ["convert", "-resize", "x300", str(full_jpg), str(small_jpg)],
            capture_output=True,
        )

        # 300×300 centre crop
        cx = max(naxis1 // 2 - 150, 0)
        cy = max(naxis2 // 2 - 150, 0)
        subprocess.run(
            ["convert", "-crop", f"300x300+{cx}+{cy}",
             str(full_jpg), str(center_jpg)],
            capture_output=True,
        )

        logger.info(f"web images updated in {web_dir}")

    except Exception as e:
        logger.error(f"web image generation failed: {e}")
    finally:
        lock_path.unlink(missing_ok=True)


def _update_archive(raw_path: Path, calibrated_path: Path,
                    ctime: float, chip_id: str, ccd_name: str,
                    ecsv_path: Optional[Path] = None,
                    archive_root: str = "/images") -> None:
    """Find the archived raw image and update its header with the final WCS.

    ccd_name is the RTS2 camera name used in the archive path (e.g. 'C0').
    chip_id is the physical camera identifier used for CAMERA_CROPS.
    """
    year, night = _night_id(ctime)
    basename = raw_path.name
    pattern = f"{archive_root}/{year}/{night}/{ccd_name}/*/{basename}"
    candidates = glob.glob(pattern)

    if not candidates:
        logger.warning(f"Archived image not found: {pattern}")
        return

    archive_path = Path(candidates[0])
    if len(candidates) > 1:
        logger.warning(f"Multiple archive candidates, using {archive_path}")

    if _copy_wcs_to_raw(calibrated_path, archive_path, chip_id, ecsv_path):
        logger.info(f"Archive header updated: {archive_path}")


def _drop_privileges(username: str) -> None:
    """Drop from root to username for user-space operations.

    One-way — call only after all root-requiring work is done.
    No-op if already running as that user or as non-root.
    """
    import pwd
    if os.getuid() != 0:
        return
    pw = pwd.getpwnam(username)
    os.setgid(pw.pw_gid)
    os.setuid(pw.pw_uid)
    os.environ['HOME'] = pw.pw_dir
    logger.debug(f"Dropped privileges to {username} (uid={pw.pw_uid})")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RTS2 imgproc pipeline")
    parser.add_argument('fits_file', help='Raw FITS image')
    parser.add_argument('--ssh-key', default=None,
                        help='Path to SSH private key for database upload (if omitted, upload is skipped)')
    parser.add_argument('--sbt-window-patch', action='store_true',
                        help='Patch missing LTV1/2 windowing keywords before calibration (SBT windowed frames)')
    parser.add_argument('--sip', type=int, default=1, metavar='N',
                        help='SIP polynomial order for pyrt-dophot (default: 1, SBT uses 2)')
    parser.add_argument('--passes', type=int, default=2, metavar='N',
                        help='Number of pyrt-dophot passes (default: 2, SBT uses 3)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show subprocess output and debug logging')
    parser.add_argument('-r', '--realtime', action='store_true',
                        help='Enable real-time web preview generation (do not use during offline reprocessing)')
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stderr, format='%(levelname)s %(name)s: %(message)s')
    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.WARNING)

    raw_path = Path(args.fits_file)
    if not raw_path.exists():
        logger.error(f"File not found: {raw_path}")
        sys.exit(1)

    # Read what we need from the raw header before anyone touches it
    with fits.open(str(raw_path)) as hdul:
        raw_header = hdul[0].header.copy()
        ctime    = raw_header.get('CTIME')
        chip_id  = get_camera_id(raw_header)          # physical: andor46, mi6166, …
        ccd_name = raw_header.get('CCD_NAME', chip_id) # RTS2 name: C0, C1, …

    pipeline = PhotometryPipeline(
        phdb_root='/home/mates/phdb',
        png_root='/home/mates/png',
    )
    temp_dir = Path(tempfile.mkdtemp(prefix='imgproc.'))
    t_start = time.time()

    try:
        # 0. Patch windowing keywords if absent (SBT windowed frames only)
        if args.sbt_window_patch:
            with fits.open(str(raw_path)) as hdul:
                h = hdul[0].header
                already_patched = 'LTV1' in h and 'LTV2' in h
                naxis1 = h.get('NAXIS1')
                naxis2 = h.get('NAXIS2')
            if already_patched:
                logger.debug("sbt-window-patch: LTV1/2 already present, skipping")
            elif naxis1 == naxis2 and naxis1 in KNOWN_WINDOW_SIZES:
                kw = compute_keywords(naxis1, chip_w=4144, chip_h=4127, shift=0)
                with fits.open(str(raw_path), mode='update') as hdul:
                    for key, (val, comment) in kw.items():
                        hdul[0].header[key] = (val, comment)
                logger.info(f"sbt-window-patch: wrote windowing keywords (window={naxis1})")
            else:
                logger.debug(f"sbt-window-patch: NAXIS1={naxis1} NAXIS2={naxis2} not a windowed frame, skipping")

        # 1. Dark/flat correction
        fits_file = pipeline.calibrate(raw_path, temp_dir)
        if fits_file is None:
            sys.exit(1)
        calibrated = temp_dir / fits_file

        # 2. Web preview — runs in parallel with solve, does not delay corrwerr
        #    Skipped during offline reprocessing to avoid interfering with live observations.
        if args.realtime:
            threading.Thread(
                target=_make_web_image, args=(calibrated, ccd_name), daemon=True,
            ).start()

        # 3. Source detection + solve + initial photometry
        if not pipeline.solve(fits_file, temp_dir):
            sys.exit(1)

        # 3.5. Report FWHM from phcat catalog to RTS2 (real-time only)
        if args.realtime:
            cat_path = temp_dir / fits_file.replace('.fits', '.cat')
            _report_fwhm(cat_path, ccd_name)

        # 4. Copy WCS into raw image and release it
        #    No ECSV yet at this point (photometry runs after corrwerr).
        _copy_wcs_to_raw(calibrated, raw_path, chip_id)

        # 5. Report to RTS2
        _corrwerr(calibrated, raw_header, chip_id)
        logger.info(f"corrwerr produced in {time.time() - t_start:.1f}s")

        # --- RTS2 reads corrwerr and begins archiving the raw image ---

        # Chown temp_dir to mates and make files world-readable before dropping
        # root.  shutil.move needs write access to the source directory; the
        # transient daemon runs as a different user and needs read access.
        import pwd, stat
        pw = pwd.getpwnam('mates')
        for p in [temp_dir, *temp_dir.rglob('*')]:
            try:
                os.chown(p, pw.pw_uid, pw.pw_gid)
                current = stat.S_IMODE(os.stat(p).st_mode)
                if p.is_dir():
                    os.chmod(p, current | 0o755)
                else:
                    os.chmod(p, current | 0o644)
            except OSError as e:
                logger.warning(f"chown/chmod {p}: {e}")

        # Drop from root to mates — all remaining work is in user space.
        _drop_privileges('mates')

        # 6. Refined photometry (as mates)
        ecsv = pipeline.photometry(fits_file, temp_dir, sip=args.sip, passes=args.passes)
        if ecsv is None:
            sys.exit(1)

        # 7. Save ECSV and PNG to phdb/png (as mates)
        ecsv_path = None
        if ctime is not None:
            ecsv_path, _ = pipeline.save_results(ecsv, temp_dir, ctime)

        # 8. Upload ECSV to database server
        if ecsv_path is not None and args.ssh_key is not None:
            DatabaseUploader(ssh_key=args.ssh_key).upload_ecsv(ecsv_path)

        # 10. Notify transient daemon (as mates, so fnovotny can read the files)
        if ecsv_path is not None:
            dft = temp_dir / fits_file.replace('df.fits', 'dft.fits')
            fits_for_transients = str(dft) if dft.exists() else str(calibrated)
            TransientSearcher().search_transients(ecsv_path, fits_for_transients)

        logger.info(f"imgproc total run time {time.time() - t_start:.1f}s")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
