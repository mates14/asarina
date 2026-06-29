#!/usr/bin/env python3

import fcntl
import os
import subprocess
import tempfile
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple
import logging
from datetime import datetime

from astropy.io import fits
from asarina.pipeline.image import ImageProcessor

logger = logging.getLogger(__name__)


class PhotometryPipeline:
    """Complete photometry pipeline from raw images to ECSV catalogs."""

    def __init__(self,
                 phdb_root: str = "~/phdb",
                 png_root: str = "~/png",
                 calib_root: str = "/home/mates/calib",
                 calib_dir_template: str = "/home/mates/flat{year}/",
                 # Alternative dark calibration
                 smart_dark_calib: str = None,
                 # Astrometric solve
                 pixel_scale: float = None,
                 # Output layout
                 phdb_date_fmt: str = "%y%m",
                 daily_summary_dir: str = None,
                 stat_dir: str = None,
                 # Photometry overrides
                 dophot_model: str = None,
                 dophot_catalog: str = None,
                 dophot_maglim: float = None,
                 dophot_enlarge: float = None,
                 dophot_terms: str = None,
                 dophot_idlimit: int = None,
                 dophot_max_stars: int = 1000,
                 # Makak-mode bundle
                 makak_mode: bool = False):

        self.phdb_root = Path(phdb_root).expanduser()
        self.png_root = Path(png_root).expanduser()
        self.calib_root = calib_root
        self.calib_dir_template = calib_dir_template

        self.smart_dark_calib = smart_dark_calib
        self.pixel_scale = pixel_scale
        self.phdb_date_fmt = phdb_date_fmt
        self.daily_summary_dir = Path(daily_summary_dir) if daily_summary_dir else None
        # stat_dir: where per-night stat ECSVs are written.
        # None disables stat logging.  Can also be set via RTS2_STAT_DIR env var.
        self.stat_dir = (
            Path(stat_dir).expanduser() if stat_dir is not None
            else (Path(os.environ['RTS2_STAT_DIR']).expanduser()
                  if 'RTS2_STAT_DIR' in os.environ else None)
        )

        self.dophot_model = dophot_model
        self.dophot_catalog = dophot_catalog
        self.dophot_maglim = dophot_maglim
        self.dophot_enlarge = dophot_enlarge
        self.dophot_terms = dophot_terms
        self.dophot_idlimit = dophot_idlimit
        self.dophot_max_stars = dophot_max_stars

        # makak_mode enables:
        #   - dark frame detection via slitposx < 0.5
        #   - default pixel scale hint of 55 arcsec/px (overrideable by pixel_scale)
        #   - -k flag in pyrt-dophot
        #   - camera crop for mi0315 in smart_dark path
        self.makak_mode = makak_mode

        self.image_processor = ImageProcessor(calib_root=calib_root,
                                              calib_dir_template=calib_dir_template)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_header_value(self, filepath: str, keyword: str):
        try:
            with fits.open(filepath) as hdul:
                return hdul[0].header[keyword]
        except Exception as e:
            logger.error(f"Error reading {keyword} from {filepath}: {e}")
            return None

    def _get_year_month_code(self, ctime: float) -> str:
        return datetime.fromtimestamp(ctime).strftime(self.phdb_date_fmt)

    def _check_existing_ecsv(self, image_path: str) -> Optional[str]:
        ctime = self._get_header_value(image_path, 'CTIME')
        if ctime is None:
            return None
        ym = self._get_year_month_code(ctime)
        ecsv_dir = self.phdb_root / ym
        base_name = Path(image_path).name.replace('-RA.fits', '').replace('.fits', '')
        if ecsv_dir.exists():
            matching = list(ecsv_dir.glob(f"{base_name}*.ecsv"))
            if matching:
                return str(matching[0])
        return None

    # ------------------------------------------------------------------
    # Pipeline phases — callable individually for imgproc integration
    # ------------------------------------------------------------------

    def _calibrate_standard(self, image_path: Path, temp_dir: Path) -> Optional[str]:
        """Dark/flat correction via ImageProcessor (master frame database)."""
        processor = ImageProcessor(calib_root=self.calib_root,
                                   calib_dir_template=self.calib_dir_template)
        processor.load_calibration_frames([str(image_path)])
        processed = processor.process_all_objects(output_dir=str(temp_dir), overwrite=True)
        if not processed:
            logger.error(f"Dark/flat correction produced no output for {image_path.name}")
            return None
        return Path(processed[0]).name

    def _calibrate_smart_dark(self, image_path: Path, temp_dir: Path) -> Optional[str]:
        """Per-pixel analytical dark correction (no flat field).

        Uses the smart_dark model to correct for dark current with
        temperature optimisation.  Crops the image to the active area
        if a crop is defined for the detected camera chip.
        """
        import numpy as np
        from asarina.calib.smart_dark import smart_dark, image_bgsigma
        from asarina.chip_id import load_chip_id
        from asarina.pipeline.image import CAMERA_CROPS

        try:
            with fits.open(str(image_path)) as hdul:
                header = hdul[0].header.copy()
                data = hdul[0].data.astype(np.float64)

            c_time = header['CTIME']
            fltr = header.get('FILTER', 'UNK')
            usec = header.get('USEC', 0)
            expt = header.get('EXPTIME', 0)
            temp = header.get('CCD_TEMP', 20.0)

            corrected, corr_temp = smart_dark(data, self.smart_dark_calib,
                                              initial_temp=temp)

            # Camera crop (full-frame readouts only)
            chip = load_chip_id(header)
            crop = CAMERA_CROPS.get(chip)
            if crop is not None:
                corrected = corrected[crop]
                if 'CRPIX1' in header:
                    header['CRPIX1'] -= crop[1].start
                if 'CRPIX2' in header:
                    header['CRPIX2'] -= crop[0].start

            header['CCD_TEMP'] = float(corr_temp)
            header['BGSIGMA'] = image_bgsigma(corrected)

            datum = datetime.utcfromtimestamp(c_time)
            datestr = datum.strftime("%Y%m%d%H%M%S")
            output_name = f"{datestr}-{int(usec)//1000:03d}-{fltr}-{expt:03.0f}-df.fits"
            output_path = temp_dir / output_name

            fits.PrimaryHDU(data=corrected.astype(np.float32),
                            header=header).writeto(str(output_path))
            return output_name

        except Exception as e:
            logger.error(f"Smart dark calibration failed for {image_path.name}: {e}")
            return None

    def calibrate(self, image_path: Path, temp_dir: Path) -> Optional[str]:
        """Apply dark/flat correction.

        Uses smart_dark pixel model when --smart-dark is configured,
        otherwise falls back to the master frame database path.

        Returns the basename of the calibrated FITS file inside temp_dir,
        or None on failure.
        """
        t = time.time()
        if self.smart_dark_calib:
            result = self._calibrate_smart_dark(image_path, temp_dir)
        else:
            result = self._calibrate_standard(image_path, temp_dir)
        logger.info(f"calibrate took {time.time()-t:.3f}s")

        if result is not None and self.makak_mode:
            cal_path = temp_dir / result
            with fits.open(str(cal_path), mode='update') as hdul:
                if 'CCD_NAME' not in hdul[0].header:
                    hdul[0].header['CCD_NAME'] = 'makak'

        return result

    def solve(self, fits_file: str, temp_dir: Path) -> bool:
        """Source detection + astrometric solve.

        Runs pyrt-phcat then pyrt-field-solve.  On success the calibrated
        FITS file in temp_dir contains a valid WCS.

        A pixel scale hint (arcsec/px) can be supplied via the pixel_scale
        constructor argument or --makak (which defaults to 55 arcsec/px).

        Returns True on success.
        """
        # Source detection
        t = time.time()
        ret = subprocess.run(
            ["pyrt-phcat", fits_file],
            cwd=str(temp_dir), capture_output=True, text=True,
        )
        if ret.returncode != 0:
            out = (ret.stdout or '').rstrip()
            err = (ret.stderr or '').rstrip()
            logger.error(f"pyrt-phcat failed"
                         + (f"\nstdout:\n{out}" if out else '')
                         + (f"\nstderr:\n{err}" if err else ''))
            return False
        logger.info(f"pyrt-phcat took {time.time()-t:.3f}s")

        # Astrometric solve
        scale = self.pixel_scale
        if scale is None and self.makak_mode:
            scale = 55.0

        solve_cmd = ["pyrt-field-solve"]
        if scale is not None:
            solve_cmd += ["--scale", str(scale)]
        solve_cmd.append(fits_file)

        t = time.time()
        ret = subprocess.run(
            solve_cmd,
            cwd=str(temp_dir), capture_output=True, text=True,
        )
        elapsed = time.time() - t
        if ret.returncode != 0:
            out = (ret.stdout or '').rstrip()
            err = (ret.stderr or '').rstrip()
            logger.error(f"pyrt-field-solve failed after {elapsed:.3f}s"
                         + (f"\nstdout:\n{out}" if out else '')
                         + (f"\nstderr:\n{err}" if err else ''))
            return False
        if ret.stdout:
            logger.debug(f"pyrt-field-solve stdout:\n{ret.stdout.rstrip()}")
        if ret.stderr:
            logger.debug(f"pyrt-field-solve stderr:\n{ret.stderr.rstrip()}")
        logger.info(f"pyrt-field-solve took {elapsed:.3f}s")

        return True

    def photometry(self, fits_file: str, temp_dir: Path,
                   sip: int = 1, passes: int = 2) -> Optional[str]:
        """Photometry: catalog matching + dophot with astrometry refit.

        Runs pyrt-cat2det to match the catalog to detections, then pyrt-dophot
        on the resulting .det file (which also refits the astrometry).
        Returns the ECSV basename on success, None on failure.

        passes controls the total number of dophot iterations.
        sip is the SIP polynomial order passed to pyrt-dophot via -S.
        """
        det_file  = fits_file.replace('.fits', '.det')
        ecsv_file = fits_file.replace('.fits', '.ecsv')

        # Catalog → detection matching
        t = time.time()
        ret = subprocess.run(
            ["pyrt-cat2det", fits_file],
            cwd=str(temp_dir), capture_output=True, text=True,
        )
        if ret.returncode != 0:
            out = (ret.stdout or '').rstrip()
            err = (ret.stderr or '').rstrip()
            logger.error(f"pyrt-cat2det failed"
                         + (f"\nstdout:\n{out}" if out else '')
                         + (f"\nstderr:\n{err}" if err else ''))
            return None
        logger.info(f"pyrt-cat2det took {time.time()-t:.3f}s")

        # Build pyrt-dophot command
        terms = self.dophot_terms or ".r3,.p3,.l"
        idlimit = self.dophot_idlimit if self.dophot_idlimit is not None else 2
        dophot_base = ["pyrt-dophot", "-m0.5", "-az", f"-S{sip}",
                       "-U", terms, f"-i{idlimit}"]
        if self.dophot_max_stars:
            dophot_base += ["--max-stars", str(self.dophot_max_stars)]
        if self.dophot_model:
            dophot_base += ["-M", self.dophot_model]
        if self.dophot_catalog:
            dophot_base += ["-C", self.dophot_catalog]
        if self.dophot_maglim is not None:
            dophot_base.append(f"-l{self.dophot_maglim}")
        if self.dophot_enlarge is not None:
            dophot_base.append(f"-e{self.dophot_enlarge}")
        if self.makak_mode:
            dophot_base.append("-k")

        # Photometry + astrometry refit over N passes
        pass_inputs = [det_file] + [ecsv_file] * (passes - 1)
        for pass_num, input_file in enumerate(pass_inputs, start=1):
            t = time.time()
            ret = subprocess.run(
                dophot_base + [input_file],
                cwd=str(temp_dir), capture_output=True, text=True,
            )
            elapsed = time.time() - t
            if ret.returncode != 0:
                out = (ret.stdout or '').rstrip()
                err = (ret.stderr or '').rstrip()
                logger.error(f"pyrt-dophot pass {pass_num} failed after {elapsed:.3f}s"
                             + (f"\nstdout:\n{out}" if out else '')
                             + (f"\nstderr:\n{err}" if err else ''))
                return None
            if ret.stdout:
                logger.debug(f"pyrt-dophot pass {pass_num} stdout:\n{ret.stdout.rstrip()}")
            if ret.stderr:
                logger.debug(f"pyrt-dophot pass {pass_num} stderr:\n{ret.stderr.rstrip()}")
            logger.info(f"pyrt-dophot pass {pass_num} took {elapsed:.3f}s")

            if not (temp_dir / ecsv_file).exists():
                logger.error(f"ECSV {ecsv_file} missing after pyrt-dophot pass {pass_num}")
                return None

        # Quality check
        from astropy.table import Table
        meta = Table.read(str(temp_dir / ecsv_file), format='ascii.ecsv').meta
        astscatt = meta.get('ASTSCATT')
        astwssr  = meta.get('ASTWSSR')
        idnum    = meta.get('IDNUM')
        if astscatt is None:
            logger.error("ASTSCATT missing from ECSV — rejecting solution")
            return None
        if float(astscatt) >= 0.3:
            logger.error(f"ASTSCATT={float(astscatt):.3f} >= 0.3 — rejecting solution")
            return None
        if idnum is None or int(idnum) <= 20:
            logger.error(f"IDNUM={idnum} <= 20 — rejecting solution")
            return None
        logger.info(f"Solution quality ok: ASTSCATT={float(astscatt):.3f}"
                    + (f" ASTWSSR={float(astwssr):.1f}" if astwssr is not None else "")
                    + f" IDNUM={idnum}")

        return ecsv_file

    def _write_daily_summary(self, ecsv_filename: str, temp_dir: Path,
                              ctime: float) -> None:
        """Append the last line of dophot.dat to the nightly summary file.

        The file is named mr{YYYYMMDD}.dat and uses a noon-to-noon convention
        (subtract 12 h before computing the date) so that observations after
        midnight belong to the same night as observations before midnight.
        """
        if self.daily_summary_dir is None:
            return

        dophot_dat = temp_dir / "dophot.dat"
        if not dophot_dat.exists():
            logger.debug("dophot.dat not found in temp_dir; skipping daily summary")
            return

        try:
            with open(dophot_dat) as f:
                lines = [ln.rstrip() for ln in f if ln.strip()]
            if not lines:
                return

            last_line = lines[-1]
            parts = last_line.split()
            if not parts:
                return

            datum = datetime.utcfromtimestamp(ctime - 43200)
            datestr = datum.strftime("%Y%m%d")
            self.daily_summary_dir.mkdir(parents=True, exist_ok=True)
            summary_file = self.daily_summary_dir / f"mr{datestr}.dat"

            stem = Path(ecsv_filename).stem
            with open(summary_file, 'a') as out:
                fcntl.flock(out.fileno(), fcntl.LOCK_EX)
                try:
                    out.write(f"{stem} {' '.join(parts[1:])}\n")
                    out.flush()
                finally:
                    fcntl.flock(out.fileno(), fcntl.LOCK_UN)

        except Exception as e:
            logger.warning(f"Could not write daily summary: {e}")

    def save_results(self, ecsv_filename: str, temp_dir: Path, ctime: float,
                     keep_image: bool = False) -> Tuple[str, Optional[str]]:
        """Move ECSV and PNG files to their permanent locations.

        Returns (ecsv_path, fits_path) where fits_path is the path of the
        kept calibrated FITS file (only when keep_image=True and the file
        exists), otherwise None.
        """
        ym = self._get_year_month_code(ctime)
        phdb_dir = self.phdb_root / ym
        png_dir = self.png_root / ym
        phdb_dir.mkdir(parents=True, exist_ok=True)
        png_dir.mkdir(parents=True, exist_ok=True)

        # Write daily summary before moving files out of temp_dir
        self._write_daily_summary(ecsv_filename, temp_dir, ctime)

        dst_ecsv = phdb_dir / ecsv_filename
        shutil.move(str(temp_dir / ecsv_filename), str(dst_ecsv))
        logger.debug(f"ECSV: {dst_ecsv}")

        for png in temp_dir.glob("*.png"):
            shutil.move(str(png), str(png_dir / png.name))

        kept_fits = None
        if keep_image:
            updated = ecsv_filename.replace('.ecsv', 'dft.fits')
            src = temp_dir / updated
            if src.exists():
                dst = Path.cwd() / updated
                shutil.copy2(str(src), str(dst))
                logger.info(f"Kept calibrated image: {dst}")
                kept_fits = str(dst)

        return str(dst_ecsv), kept_fits

    def _process_makak_dark(self, image_path: Path) -> None:
        """Process a Makak dark frame: compute darksig and write to nightly stats.

        Dark frames are identified by slitposx < 0.5 (slit closed = zenith camera
        shutter blocking the sky).  The corrected-temperature dark noise (darksig)
        is written to mr{YYYYMMDD}d.dat in daily_summary_dir.
        """
        import numpy as np
        from asarina.calib.smart_dark import smart_dark, image_bgsigma

        if self.smart_dark_calib is None:
            logger.warning(f"Dark frame {image_path.name}: no --smart-dark calib; skipping")
            return
        if self.daily_summary_dir is None:
            logger.debug(f"Dark frame {image_path.name}: no --daily-summary dir; skipping")
            return

        try:
            with fits.open(str(image_path)) as hdul:
                header = hdul[0].header
                data = hdul[0].data.astype(np.float64)
                c_time = header['CTIME']
                usec = header.get('USEC', 0)
                temp = header.get('CCD_TEMP', 20.0)

            corrected, corr_temp = smart_dark(data, self.smart_dark_calib,
                                              initial_temp=temp)
            darksig = image_bgsigma(corrected)

            datum = datetime.utcfromtimestamp(c_time - 43200)
            datestr = datum.strftime("%Y%m%d")
            self.daily_summary_dir.mkdir(parents=True, exist_ok=True)
            dark_file = self.daily_summary_dir / f"mr{datestr}d.dat"

            with open(dark_file, 'a') as f:
                f.write(f"{c_time + usec / 1e6:.6f} {darksig:.3f} {corr_temp:.3f}\n")

            logger.info(f"Dark {image_path.name}: darksig={darksig:.3f} "
                        f"corr_temp={corr_temp:.3f}")

        except Exception as e:
            logger.error(f"Error processing dark frame {image_path.name}: {e}")

    def _write_stat(self, ecsv_path: str) -> None:
        """Write a stat record for a successfully processed image."""
        if self.stat_dir is None:
            return
        try:
            from asarina.observe.stat import record_from_ecsv, write_stat_record
            record = record_from_ecsv(ecsv_path)
            if record is not None:
                write_stat_record(record, stat_dir=str(self.stat_dir))
                logger.debug(f"Stat record written for {record['image']}")
            else:
                logger.warning(f"Could not extract stat record from {ecsv_path}")
        except Exception as e:
            logger.warning(f"Stat write failed for {ecsv_path}: {e}")

    # ------------------------------------------------------------------
    # High-level entry point (used by watch)
    # ------------------------------------------------------------------

    def process_image(self, image_path: str, force: bool = False,
                      keep_image: bool = False) -> Optional[Tuple[str, Optional[str]]]:
        """Process a single image through the complete pipeline.

        Returns (ecsv_path, fits_path) on success, or None on failure.
        fits_path is the kept calibrated FITS when keep_image=True, else None.
        """
        image_path = Path(image_path)
        logger.debug(f"××××× {image_path.name} ×××××")

        # Makak dark frame detection (slitposx < 0.5 = shutter closed)
        if self.makak_mode:
            try:
                with fits.open(str(image_path)) as hdul:
                    slitposx = hdul[0].header.get('slitposx', 1.0)
            except Exception:
                slitposx = 1.0
            if float(slitposx) < 0.5:
                self._process_makak_dark(image_path)
                return None

        if not force:
            existing = self._check_existing_ecsv(str(image_path))
            if existing:
                logger.info(f"Result already exists: {existing}")
                return existing, None

        ctime = self._get_header_value(str(image_path), 'CTIME')
        if ctime is None:
            logger.error(f"Cannot read CTIME from {image_path}")
            return None

        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp)

            fits_file = self.calibrate(image_path, temp_dir)
            if fits_file is None:
                return None

            logger.info(f"Running photometry for: {fits_file}")

            if not self.solve(fits_file, temp_dir):
                return None

            ecsv = self.photometry(fits_file, temp_dir)
            if ecsv is None:
                return None

            result = self.save_results(ecsv, temp_dir, ctime, keep_image)
            self._write_stat(result[0])
            return result

    def process_images(self, image_paths: List[str], force: bool = False,
                       keep_image: bool = False) -> List[str]:
        """Process multiple images. Returns list of created ECSV paths."""
        successful = []
        for p in image_paths:
            result = self.process_image(p, force, keep_image)
            if result is not None:
                ecsv_path, _ = result
                successful.append(ecsv_path)
        logger.info(f"Successfully processed {len(successful)}/{len(image_paths)} images")
        return successful

