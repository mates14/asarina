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
                 phdb_by_night: bool = False,
                 daily_summary_dir: str = None,
                 stat_dir: str = None,
                 # Photometry overrides
                 phcat_fwhm: float = None,
                 phcat_back_size: int = None,
                 phcat_aperture: float = None,
                 phcat_noiraf: bool = False,
                 dophot_model: str = None,
                 dophot_catalog: str = None,
                 dophot_maglim: float = None,
                 dophot_enlarge: float = None,
                 dophot_terms: str = None,
                 dophot_idlimit: int = None,
                 dophot_max_stars: int = 1000,
                 dophot_workers: int = None,
                 dophot_refit_zpn: bool = False,
                 dophot_zpn_from_camera: bool = False,
                 dophot_filter_terms: dict = None,
                 # Fixed camera: start from the last solution instead of solving
                 wcs_guess_dir: str = None,
                 wcs_guess_keep: bool = False,
                 min_idnum: int = 20,
                 # Makak-mode bundle
                 makak_mode: bool = False):

        self.phdb_root = Path(phdb_root).expanduser()
        self.png_root = Path(png_root).expanduser()
        self.calib_root = calib_root
        self.calib_dir_template = calib_dir_template

        self.smart_dark_calib = smart_dark_calib
        self.pixel_scale = pixel_scale
        self.phcat_fwhm = phcat_fwhm
        self.phcat_back_size = phcat_back_size
        self.phcat_aperture = phcat_aperture
        self.phcat_noiraf = phcat_noiraf
        self.phdb_date_fmt = phdb_date_fmt
        # Date the phdb subdirectory by the night (UT - 12 h, as the RTS2
        # archive does) rather than by the local calendar date, so that a
        # night's frames stay together, e.g. in %y/%m%d.
        self.phdb_by_night = phdb_by_night
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
        # Processes for pyrt-dophot's stepwise term search (--workers).  None
        # leaves pyrt's default, in-process: its trials are single linear
        # solves, cheaper than starting worker processes.  N > 1 asks for a
        # pool of N (0 is not passed on, so pyrt's own 0 = one per CPU is not
        # reachable from here).
        self.dophot_workers = dophot_workers
        # Refit the ZPN radial projection terms (-z). Off by default: on a
        # subsampled star list the ZPN refit is under-constrained and can blow
        # up (e.g. PV2_5 -> -5290), producing a wild astrometric solution.
        # With it off, the nominal header ZPN is kept and SIP absorbs the
        # residual distortion.
        self.dophot_refit_zpn = dophot_refit_zpn
        # Start from the camera ZPN model hardcoded in pyrt (CRPIX, PV2_*
        # held fixed, only CD and CRVAL fitted) instead of the field-solve
        # TAN, which cannot follow a wide field's distortion and so only
        # matches stars near its reference point (SBT).
        self.dophot_zpn_from_camera = dophot_zpn_from_camera
        # Extra terms per filter (config filter_terms_<f>), appended to the
        # -U prescription - e.g. fixed colour terms of an unfiltered camera.
        self.dophot_filter_terms = dophot_filter_terms or {}

        # makak_mode enables:
        #   - dark frame detection via slitposx < 0.5
        #   - default pixel scale hint of 55 arcsec/px (overrideable by pixel_scale)
        #   - -k flag in pyrt-dophot
        #   - camera crop for mi0315 in smart_dark path
        self.makak_mode = makak_mode

        # A fixed camera sees the same alt-az field all the time, so its last
        # good WCS, turned in RA by the sidereal angle since, is a solution
        # for the next frame - good to the drift of the mount-less camera and
        # refraction, well inside what pyrt-dophot's refit pulls in.  That
        # skips pyrt-field-solve, and with it the minutes it spends failing
        # on a cloudy frame.  A guess the refit rejects falls back to
        # field-solve (field_solve()), and every accepted solution becomes
        # the next guess, so a moved camera heals itself.
        self.wcs_guess_dir = Path(wcs_guess_dir).expanduser() if wcs_guess_dir else None
        self.used_wcs_guess = False
        self.wcs_guess_fresh = False
        self.wcs_guess_keep = wcs_guess_keep
        self.min_idnum = min_idnum

        # The calibrated frame of the last calibrate(), for the failure summary
        self.calibrated = None

        # Why the last phase failed, None while all went well: 'nodark',
        # 'noflat', 'unreadable', 'notobject' (calibrate), 'nostars',
        # 'nosolve' (solve), 'catalog', 'dophot', 'astqual', 'idnum'
        # (photometry).  asarina-imgproc exits with a code for each.
        self.failure = None

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
        if self.phdb_by_night:
            return datetime.utcfromtimestamp(ctime - 43200).strftime(self.phdb_date_fmt)
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
        processed = processor.process_all_objects(output_dir=str(temp_dir), overwrite=True, photometry=False)
        if not processed:
            self.failure = processor.failure or ('unreadable' if processor.objects else 'notobject')
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
        from pyrt.utils.chip_id import load_chip_id
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
            self.failure = 'unreadable'
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

        if result is not None:
            self.calibrated = temp_dir / result
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
        phcat_cmd = ["pyrt-phcat"]
        if self.phcat_fwhm is not None:
            phcat_cmd += ["-f", str(self.phcat_fwhm)]
        if self.phcat_back_size is not None:
            phcat_cmd += ["--back-size", str(self.phcat_back_size)]
        if self.phcat_aperture is not None:
            phcat_cmd += ["--aperture", str(self.phcat_aperture)]
        if self.phcat_noiraf:
            phcat_cmd.append("-I")
        ret = subprocess.run(
            phcat_cmd + [fits_file],
            cwd=str(temp_dir), capture_output=True, text=True,
        )
        if ret.returncode != 0:
            out = (ret.stdout or '').rstrip()
            err = (ret.stderr or '').rstrip()
            self.failure = 'nostars'
            logger.error(f"pyrt-phcat failed"
                         + (f"\nstdout:\n{out}" if out else '')
                         + (f"\nstderr:\n{err}" if err else ''))
            return False
        logger.info(f"pyrt-phcat took {time.time()-t:.3f}s")

        if self._apply_wcs_guess(temp_dir / fits_file):
            return True
        return self.field_solve(fits_file, temp_dir)

    def field_solve(self, fits_file: str, temp_dir: Path) -> bool:
        """Blind astrometric solve with pyrt-field-solve (after pyrt-phcat)."""
        self.used_wcs_guess = False
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
            self.failure = 'nosolve'
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
        sip is the SIP polynomial order for the first pass (0, 1 or 2);
        later passes escalate to 2 unless SIP is disabled entirely.
        """
        det_file  = fits_file.replace('.fits', '.det')
        ecsv_file = fits_file.replace('.fits', '.ecsv')
        from astropy.io import fits
        fltr = str(fits.getval(str(temp_dir / fits_file), 'FILTER', default='')).lower()

        # Catalog → detection matching
        t = time.time()
        ret = subprocess.run(
            ["pyrt-cat2det", fits_file],
            cwd=str(temp_dir), capture_output=True, text=True,
        )
        if ret.returncode != 0:
            out = (ret.stdout or '').rstrip()
            err = (ret.stderr or '').rstrip()
            self.failure = 'catalog'
            logger.error(f"pyrt-cat2det failed"
                         + (f"\nstdout:\n{out}" if out else '')
                         + (f"\nstderr:\n{err}" if err else ''))
            return None
        logger.info(f"pyrt-cat2det took {time.time()-t:.3f}s")

        # Build pyrt-dophot command
        # Both sites drive the terms from the command line - BART's Kodak
        # chips need the .l linearity term, D50's do not and prefer the "&"
        # speed forms - so this default only ever covers a bare invocation.
        terms = self.dophot_terms or ".r3,.p3,.l"
        if fltr in self.dophot_filter_terms:
            terms += "," + self.dophot_filter_terms[fltr]
            logger.info(f"Filter {fltr}: dophot terms {terms}")
        idlimit = self.dophot_idlimit if self.dophot_idlimit is not None else 2
        # -a always refits the astrometry (CD + SIP).  -S and -z are not here:
        # both escalate per pass, and neither is applied to the first one.
        dophot_base = ["pyrt-dophot", "-m0.5", "-a",
                       "-U", terms, f"-i{idlimit}"]
        if self.dophot_max_stars:
            dophot_base += ["--max-stars", str(self.dophot_max_stars)]
        if self.dophot_workers:
            dophot_base += ["--workers", str(self.dophot_workers)]
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

        # pyrt-dophot implements SIP orders 0 (off), 1 and 2 and nothing above
        # 2, so a higher request is clamped rather than passed through.
        sip = min(sip, 2)

        # Photometry + astrometry refit over N passes.  pyrt-dophot judges its
        # own solution: ASTQUAL=OK or PARTIAL describes the WCS it wrote; a
        # refused refit leaves the input WCS with that input's own AST* keys,
        # i.e. no ASTQUAL at all unless the input was itself a verified pass.
        if not self._dophot_passes(dophot_base, temp_dir, ecsv_file,
                                   [det_file] + [ecsv_file] * (passes - 1),
                                   sip, self.dophot_refit_zpn):
            return None
        qual = self._astqual(temp_dir / ecsv_file)

        # A partial solution is sane but does not cover the chip: continue
        # from it - pyrt starts the next refit from the WCS it finds.
        if qual == 'PARTIAL':
            logger.warning("Partial astrometric solution, extending it")
            if not self._dophot_passes(dophot_base, temp_dir, ecsv_file,
                                       [ecsv_file] * max(1, passes - 1),
                                       sip, self.dophot_refit_zpn, first_pass=2):
                return None
            qual = self._astqual(temp_dir / ecsv_file)

        # No verified solution: start over from the field-solve WCS with the
        # conservative prescription - no ZPN refit, SIP at most 1.
        if qual is None:
            logger.warning("No verified astrometric solution, full rerun without ZPN refit, SIP<=1")
            if not self._dophot_passes(dophot_base, temp_dir, ecsv_file,
                                       [det_file] + [ecsv_file] * (passes - 1),
                                       min(sip, 1), False, sip_max=1):
                return None
            qual = self._astqual(temp_dir / ecsv_file)

        # Fixed camera through clouds: the stars do not cover the chip and the
        # refit cannot be verified, but the guess is the last verified solution
        # turned by the sidereal angle.  Photometry on it as it is, no refit;
        # the ECSV then has no ASTQUAL and the guess is not updated from it.
        # Only a guess at most WCS_GUESS_KEEP_AGE away: the pointing departs
        # from the sidereal turn (Makak: 0.06 deg in RA after 1 h, 0.15 after
        # 2 h, 0.8 after 7 h), and the camera's tilt is changed by day.  An
        # older guess matches the wrong stars (the frame is field-solved).
        if qual != 'OK' and self.wcs_guess_keep and self.used_wcs_guess and self.wcs_guess_fresh:
            logger.warning(f"ASTQUAL={qual}, photometry on the WCS guess as it is")
            # the refit's frame would pass for the solution downstream
            (temp_dir / fits_file.replace('.fits', 't.fits')).unlink(missing_ok=True)
            if not self._dophot_passes([a for a in dophot_base if a != "-a"], temp_dir,
                                       ecsv_file, [det_file], 0, False):
                return None
            qual = None
        elif qual != 'OK':
            self.failure = 'astqual'
            logger.error(f"ASTQUAL={qual} — rejecting solution")
            return None

        from astropy.table import Table
        meta = Table.read(str(temp_dir / ecsv_file), format='ascii.ecsv').meta
        idnum = meta.get('IDNUM')
        if idnum is None or int(idnum) <= self.min_idnum:
            self.failure = 'idnum'
            logger.error(f"IDNUM={idnum} <= {self.min_idnum} — rejecting solution")
            return None
        logger.info(f"Solution quality ok: ASTQUAL={qual} ASTCOVER={meta.get('ASTCOVER')}"
                    f" ASTSIGMA={meta.get('ASTSIGMA')} IDNUM={idnum}")

        solved = temp_dir / fits_file.replace('.fits', 't.fits')
        if self.wcs_guess_dir is not None and qual == 'OK' and solved.exists():
            self._store_wcs_guess(solved)

        return ecsv_file

    def _dophot_passes(self, dophot_base: List[str], temp_dir: Path, ecsv_file: str,
                       inputs: List[str], sip: int, refit_zpn: bool,
                       first_pass: int = 1, sip_max: int = 2) -> bool:
        """Run pyrt-dophot once per input; False if a pass fails outright.

        The first pass works off the rough field-solve WCS (or, with
        zpn_from_camera, the fixed camera ZPN model), so it stays deliberately
        conservative: the requested SIP order, and never the ZPN refit.
        Later passes have a real solution to build on, so SIP escalates (to
        sip_max) and the ZPN radial terms are refitted when refit_zpn asks for
        it.  An explicitly disabled SIP (0) is left alone throughout.
        """
        for pass_num, input_file in enumerate(inputs, start=first_pass):
            pass_sip = sip if (pass_num == 1 or sip <= 0) else sip_max
            extra = [f"-S{pass_sip}"]
            if pass_num == 1 and self.dophot_zpn_from_camera:
                extra.append("--zpn-from-camera")
            if pass_num > 1 and refit_zpn:
                extra.append("-z")
            t = time.time()
            logger.debug(f"pyrt-dophot pass {pass_num} cmd: "
                         f"{dophot_base + extra + [input_file]}")

            ret = subprocess.run(
                dophot_base + extra + [input_file],
                cwd=str(temp_dir), capture_output=True, text=True,
            )
            elapsed = time.time() - t
            if ret.returncode != 0:
                out = (ret.stdout or '').rstrip()
                err = (ret.stderr or '').rstrip()
                self.failure = 'dophot'
                logger.error(f"pyrt-dophot pass {pass_num} failed after {elapsed:.3f}s"
                             + (f"\nstdout:\n{out}" if out else '')
                             + (f"\nstderr:\n{err}" if err else ''))
                return False
            if ret.stdout:
                logger.debug(f"pyrt-dophot pass {pass_num} stdout:\n{ret.stdout.rstrip()}")
            if ret.stderr:
                logger.debug(f"pyrt-dophot pass {pass_num} stderr:\n{ret.stderr.rstrip()}")
            logger.info(f"pyrt-dophot pass {pass_num} took {elapsed:.3f}s")

            if not (temp_dir / ecsv_file).exists():
                self.failure = 'dophot'
                logger.error(f"ECSV {ecsv_file} missing after pyrt-dophot pass {pass_num}")
                return False
        return True

    # Sidereal rotation of the sky, degrees of RA per second of time
    SIDEREAL_DEG_PER_S = 360.98564736629 / 86400.0
    # Photometry on an unverified guess only up to this far from it (s)
    WCS_GUESS_KEEP_AGE = 3600.0
    # WCS keywords carried from one frame's solution to the next
    _WCS_GUESS_KEYS = ('WCSAXES', 'CTYPE', 'CUNIT', 'CRVAL', 'CRPIX', 'CD1_', 'CD2_',
                       'PV1_', 'PV2_', 'LONPOLE', 'LATPOLE', 'RADESYS', 'EQUINOX',
                       'A_', 'B_', 'AP_', 'BP_')

    def _wcs_guess_file(self, header) -> Optional[Path]:
        if self.wcs_guess_dir is None:
            return None
        from pyrt.utils.chip_id import get_camera_id
        return self.wcs_guess_dir / f"{get_camera_id(header)}.wcs"

    def _apply_wcs_guess(self, fits_path: Path) -> bool:
        """Put the last good solution, turned to this frame's time, into fits_path."""
        self.used_wcs_guess = False
        self.wcs_guess_fresh = False
        if self.wcs_guess_dir is None:
            return False
        try:
            with fits.open(str(fits_path), mode='update') as hdul:
                hdr = hdul[0].header
                guess_file = self._wcs_guess_file(hdr)
                if guess_file is None or not guess_file.exists():
                    return False
                guess = fits.Header.fromtextfile(str(guess_file))
                ctime = float(hdr['CTIME']) + hdr.get('USEC', 0) / 1e6
                dt = ctime - float(guess['GUESSCT'])
                self.wcs_guess_fresh = abs(dt) <= self.WCS_GUESS_KEEP_AGE
                for key in list(guess.keys()):
                    if key.startswith(self._WCS_GUESS_KEYS):
                        hdr[key] = guess[key]
                hdr['CRVAL1'] = (float(guess['CRVAL1']) + dt * self.SIDEREAL_DEG_PER_S) % 360.0
                hdul.flush()
        except Exception as e:
            logger.warning(f"WCS guess not applied: {e}")
            return False
        self.used_wcs_guess = True
        logger.info(f"WCS guess from {guess_file.name}, {dt:.0f}s old: "
                    f"CRVAL={hdr['CRVAL1']:.4f},{hdr['CRVAL2']:.4f}")
        return True

    def _store_wcs_guess(self, solved_path: Path) -> None:
        """Keep solved_path's WCS as the guess for the frames processed next."""
        try:
            hdr = fits.getheader(str(solved_path))
            guess_file = self._wcs_guess_file(hdr)
            if guess_file is None:
                return
            ctime = float(hdr['CTIME']) + hdr.get('USEC', 0) / 1e6
            out = fits.Header()
            out['GUESSCT'] = (ctime, 'CTIME of the frame this WCS solves')
            for card in hdr.cards:
                if card.keyword.startswith(self._WCS_GUESS_KEYS):
                    out.append(card)
            guess_file.parent.mkdir(parents=True, exist_ok=True)
            # The last one solved wins, whichever way in time the frames go:
            # going back through an archive (newest first) the next frame is
            # the one before, and parallel workers finish only seconds apart.
            # A newest-wins rule kept the night's last solution for the whole
            # night there, hours off the sidereal turn.
            with open(guess_file.with_suffix('.lock'), 'w') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                tmp = guess_file.with_suffix('.tmp')
                out.totextfile(str(tmp), overwrite=True)
                os.replace(tmp, guess_file)
        except Exception as e:
            logger.warning(f"WCS guess not stored: {e}")

    @staticmethod
    def _astqual(ecsv_path: Path) -> Optional[str]:
        """ASTQUAL of the WCS in the ECSV: 'OK', 'PARTIAL', or None if unverified."""
        from astropy.table import Table
        return Table.read(str(ecsv_path), format='ascii.ecsv').meta.get('ASTQUAL')

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

    def write_failure_summary(self) -> None:
        """Makak: a calibrated frame that failed later goes to mr{YYYYMMDD}f.dat.

        ctime, BGSIGMA, fitted CCD_TEMP and the failure: the sky brightness of
        a cloudy or moonlit frame, which the nightly graph shows even without a
        zeropoint (the old makak.py wrote these as "ctime - bgnoise temp").
        """
        if not (self.makak_mode and self.daily_summary_dir and self.calibrated
                and self.calibrated.exists()):
            return
        try:
            hdr = fits.getheader(str(self.calibrated))
            ctime = float(hdr['CTIME'])
            datestr = datetime.utcfromtimestamp(ctime - 43200).strftime("%Y%m%d")
            self.daily_summary_dir.mkdir(parents=True, exist_ok=True)
            with open(self.daily_summary_dir / f"mr{datestr}f.dat", 'a') as f:
                f.write(f"{ctime + hdr.get('USEC', 0) / 1e6:.6f} {hdr.get('BGSIGMA', 0):.3f} "
                        f"{hdr.get('CCD_TEMP', 0):.3f} {self.failure}\n")
        except Exception as e:
            logger.warning(f"Could not write failure summary: {e}")

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
            if ecsv is None and self.used_wcs_guess:
                logger.warning(f"Rejected from the WCS guess ({self.failure}), field-solving")
                self.failure = None
                if self.field_solve(fits_file, temp_dir):
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

