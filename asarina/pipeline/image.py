#!/usr/bin/env python3

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

import numpy as np
from astropy.io import fits
import scipy.stats as stats

from asarina.chip_id import load_chip_id

# Camera-specific crop regions applied after dark/flat correction.
# FITS section [x1:x2, y1:y2] (1-indexed inclusive) → numpy [y1-1:y2, x1-1:x2]
#   mi6166, mi6167 (ex C1, C2): FITS [30:4127, 11:4108]
#   mi0008         (ex C3):     FITS [35:2218,  5:1476]
CAMERA_CROPS = {
    'mi6166': np.s_[10:4108, 29:4127],
    'mi6167': np.s_[10:4108, 29:4127],
    'mi0008': np.s_[4:1476,  34:2218],
    'mi0315': np.s_[19:1218, 215:1414],   # Makak zenith camera
}

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Astronomical image processing pipeline for dark/flat calibration."""

    def __init__(self, temp_grouping: int = 5, exposure_tolerance: float = 3.0,
                 calib_dir_template: str = "/home/mates/flat{year}/",
                 calib_root: str = None,
                 max_year_search: int = 5,
                 sip: int = 1,
                 passes: int = 2,
                 dophot_model: str = None,
                 dophot_catalog: str = None,
                 dophot_maglim: float = None,
                 dophot_enlarge: float = None,
                 dophot_terms: str = None,
                 dophot_idlimit: int = None,
                 dophot_max_stars: int = 1000):
        self.temp_grouping = temp_grouping
        self.exposure_tolerance = exposure_tolerance
        self.calib_dir_template = calib_dir_template
        self.max_year_search = max_year_search
        self.calib_root = Path(calib_root) if calib_root else Path.home() / 'calib'
        self.sip = sip
        self.passes = passes
        self.dophot_model = dophot_model
        self.dophot_catalog = dophot_catalog
        self.dophot_maglim = dophot_maglim
        self.dophot_enlarge = dophot_enlarge
        self.dophot_terms = dophot_terms
        self.dophot_idlimit = dophot_idlimit
        self.dophot_max_stars = dophot_max_stars
        self.master_darks = {}
        self.master_flats = {}
        self.objects = []
        self.loaded_files = set()
        self.loaded_calib_years = set()
        self.observation_year = None
        self.ccd_name = None
        self.chip_id = None
        
    def _round_temperature(self, temp: float) -> int:
        """Round temperature to nearest group."""
        if temp >= 0:
            return self.temp_grouping * int(temp / self.temp_grouping + 0.5)
        else:
            return self.temp_grouping * int(temp / self.temp_grouping + 0.5 - 1)
    
    def _get_header_value(self, header: fits.Header, key: str, default=None):
        """Safely get header value with default."""
        try:
            return header[key]
        except KeyError:
            if default is not None:
                logger.warning(f"Header key '{key}' not found, using default: {default}")
                return default
            else:
                raise KeyError(f"Required header key '{key}' not found")
    
    def _extract_observation_year(self, filepath: str) -> Optional[int]:
        """Extract observation year from DATE-OBS keyword."""
        try:
            with fits.open(filepath) as hdul:
                date_obs = hdul[0].header.get('DATE-OBS')
                if date_obs:
                    # Handle various DATE-OBS formats (YYYY-MM-DD, YYYY-MM-DDTHH:MM:SS, etc.)
                    year_str = date_obs.split('-')[0]
                    return int(year_str)
        except Exception as e:
            logger.debug(f"Could not extract year from {filepath}: {e}")
        return None

    def _extract_ccd_name(self, filepath: str) -> Optional[str]:
        """Extract CCD_NAME from header."""
        try:
            with fits.open(filepath) as hdul:
                ccd_name = hdul[0].header.get('CCD_NAME')
                if ccd_name:
                    return ccd_name.strip()
        except Exception as e:
            logger.debug(f"Could not extract CCD_NAME from {filepath}: {e}")
        return None
    
    def _load_calibration_year(self, year: int) -> Tuple[int, int]:
        """Load calibration frames for a specific year from all known paths.

        Tries the following paths in order, accumulating from all that exist:
        1. New path: ~/calib/{chip_id}/{year}/
        2. Legacy SBT multi-camera: flat{year}/{ccd_name}/
        3. Legacy D50 single-camera: flat{year}/

        Already-loaded files are skipped, so the new path takes priority for
        any frames present in both locations.  Already-loaded years are no-ops.

        Returns:
            Tuple of (num_darks, num_flats) loaded from this year's directories
        """
        if year in self.loaded_calib_years:
            return 0, 0
        self.loaded_calib_years.add(year)

        total_darks = total_flats = 0

        # 1. New path: ~/calib/{chip_id}/{year}/
        if self.chip_id:
            new_path = self.calib_root / self.chip_id / str(year)
            if new_path.exists():
                logger.info(f"Loading calibration frames from {new_path}")
                d, f = self._load_fits_dir(new_path)
                total_darks += d
                total_flats += f
            else:
                logger.debug(f"New calib path not found: {new_path}")

        # 2. Legacy multi-camera path: flat{year}/{ccd_name}/
        legacy_base = Path(self.calib_dir_template.format(year=year))
        if self.ccd_name:
            multi_path = legacy_base / self.ccd_name
            if multi_path.exists():
                logger.info(f"Loading calibration frames from {multi_path}")
                d, f = self._load_fits_dir(multi_path)
                total_darks += d
                total_flats += f

        # 3. Legacy single-camera path: flat{year}/
        elif legacy_base.exists():
            logger.info(f"Loading calibration frames from {legacy_base}")
            d, f = self._load_fits_dir(legacy_base)
            total_darks += d
            total_flats += f

        return total_darks, total_flats

    def _year_search_order(self) -> List[int]:
        """Return years to search, closest to observation_year first (excluding it)."""
        if self.observation_year is None:
            return []
        offsets = []
        for i in range(1, self.max_year_search + 1):
            offsets.append(-i)
            offsets.append(i)
        return [self.observation_year + o for o in offsets]

    def _load_fits_dir(self, directory: Path) -> Tuple[int, int]:
        """Collect and load all FITS files from a directory, skipping duplicates."""
        fits_files = []
        for pattern in ['*.fits', '*.fit', '*.FITS', '*.FIT']:
            fits_files.extend(directory.glob(pattern))

        new_files = [str(f) for f in fits_files if str(f) not in self.loaded_files]
        if not new_files:
            logger.debug(f"No new calibration files in {directory}")
            return 0, 0

        logger.debug(f"Found {len(new_files)} new calibration files in {directory}")
        return self._load_calibration_files(new_files)

    def _load_calibration_files(self, file_paths: List[str]) -> Tuple[int, int]:
        """Internal method to load calibration frames from file list.
        
        Returns:
            Tuple of (num_darks, num_flats) loaded
        """
        num_darks = 0
        num_flats = 0
        
        for filepath in file_paths:
            # Skip if already loaded
            if filepath in self.loaded_files:
                continue
                
            try:
                with fits.open(filepath) as hdul:
                    header = hdul[0].header
                    chip = load_chip_id(header)
                    
                    try:
                        image_type = header['IMAGETYP']
                    except KeyError:
                        logger.debug(f"{filepath}: No IMAGETYP keyword, skipping")
                        continue
                    
                    binx = self._get_header_value(header, 'BINX', 1)
                    
                    if image_type == 'mdark':
                        temp = self._round_temperature(header['CCD_TEMP'])
                        expt = header['EXPTIME']
                        
                        # Initialize nested dictionaries
                        if chip not in self.master_darks:
                            self.master_darks[chip] = {}
                        if temp not in self.master_darks[chip]:
                            self.master_darks[chip][temp] = {}
                        if binx not in self.master_darks[chip][temp]:
                            self.master_darks[chip][temp][binx] = {}
                        
                        # Only add if not already present (avoid overwriting)
                        if expt not in self.master_darks[chip][temp][binx]:
                            self.master_darks[chip][temp][binx][expt] = filepath
                            num_darks += 1
                        
                    elif image_type == 'mflat':
                        fltr = header['FILTER']
                        
                        # Initialize nested dictionaries
                        if chip not in self.master_flats:
                            self.master_flats[chip] = {}
                        if fltr not in self.master_flats[chip]:
                            self.master_flats[chip][fltr] = {}
                        
                        # Only add if not already present
                        if binx not in self.master_flats[chip][fltr]:
                            self.master_flats[chip][fltr][binx] = filepath
                            num_flats += 1
                        
                    elif image_type == 'object':
                        self.objects.append(filepath)
                
                # Mark as loaded
                self.loaded_files.add(filepath)
                        
            except Exception as e:
                logger.error(f"Error df correcting 2 {filepath}: {e}")
                continue
        
        return num_darks, num_flats

    def load_calibration_frames(self, file_paths: List[str]) -> Tuple[int, int]:
        """Load master darks and flats from file list, plus calibration directory.

        Returns:
            Tuple of (num_darks, num_flats) loaded
        """
        # First pass: determine observation year, CCD name, and chip_id from first file
        if file_paths:
            if self.observation_year is None:
                self.observation_year = self._extract_observation_year(file_paths[0])
            if self.ccd_name is None:
                self.ccd_name = self._extract_ccd_name(file_paths[0])
                if self.ccd_name:
                    logger.debug(f"Determined CCD name: {self.ccd_name}")
            if self.chip_id is None:
                try:
                    with fits.open(file_paths[0]) as hdul:
                        self.chip_id = load_chip_id(hdul[0].header)
                        if self.chip_id == 'unknown':
                            self.chip_id = None
                        else:
                            logger.debug(f"Determined chip_id: {self.chip_id}")
                except Exception as e:
                    logger.debug(f"Could not determine chip_id from {file_paths[0]}: {e}")

        # Load files from command line
        num_darks, num_flats = self._load_calibration_files(file_paths)

        # Load additional calibration frames from directory
        if self.observation_year is None:
            logger.warning("No observation year determined, skipping calibration directory")
            dir_darks = dir_flats = 0
        else:
            dir_darks, dir_flats = self._load_calibration_year(self.observation_year)

        total_darks = num_darks + dir_darks
        total_flats = num_flats + dir_flats

        logger.info(f"Loaded {total_darks} darks, {total_flats} flats, {len(self.objects)} objects")
        if dir_darks > 0 or dir_flats > 0:
            logger.debug(f"From calibration directory: {dir_darks} darks, {dir_flats} flats")

        return total_darks, total_flats
    
    def _parse_ccdsec(self, header: fits.Header) -> Optional[np.s_]:
        """Return numpy slice for a windowed frame, or None for a full-frame image.

        Reads CCDSEC '[x1:x2,y1:y2]' (1-indexed, FITS column-first convention).
        Falls back to LTV1/LTV2 + NAXIS when CCDSEC is absent.
        """
        import re
        ccdsec = header.get('CCDSEC')
        if ccdsec:
            m = re.match(r'\[(\d+):(\d+),(\d+):(\d+)\]', ccdsec.strip())
            if not m:
                logger.warning(f"Cannot parse CCDSEC: {ccdsec}")
                return None
            x1, x2, y1, y2 = (int(g) for g in m.groups())
            return np.s_[y1-1:y2, x1-1:x2]

        ltv1 = header.get('LTV1', 0.0)
        ltv2 = header.get('LTV2', 0.0)
        if ltv1 == 0.0 and ltv2 == 0.0:
            return None
        naxis1 = header.get('NAXIS1')
        naxis2 = header.get('NAXIS2')
        if not naxis1 or not naxis2:
            return None
        x1 = int(ltv1) + 1
        y1 = int(ltv2) + 1
        return np.s_[y1-1:y1-1+naxis2, x1-1:x1-1+naxis1]

    def _find_best_dark_in_memory(self, chip: str, rounded_temp: int, binx: int,
                                   expt: float) -> Optional[str]:
        """Search already-loaded darks for the best match (no I/O)."""
        if (chip not in self.master_darks or
                rounded_temp not in self.master_darks[chip] or
                binx not in self.master_darks[chip][rounded_temp]):
            return None

        available_exposures = self.master_darks[chip][rounded_temp][binx]

        min_diff = float('inf')
        best_exposure = None
        for available_expt in available_exposures:
            diff = abs(expt - available_expt)
            if diff < min_diff:
                min_diff = diff
                best_exposure = available_expt

        if min_diff <= self.exposure_tolerance:
            return available_exposures[best_exposure]
        return None

    def find_best_dark(self, chip: str, temp: float, binx: int, expt: float) -> Optional[str]:
        """Find the best matching dark frame, falling back to adjacent years if needed."""
        rounded_temp = self._round_temperature(temp)

        result = self._find_best_dark_in_memory(chip, rounded_temp, binx, expt)
        if result is not None:
            return result

        # Try adjacent years, closest first
        for year in self._year_search_order():
            darks, _ = self._load_calibration_year(year)
            if darks > 0:
                result = self._find_best_dark_in_memory(chip, rounded_temp, binx, expt)
                if result is not None:
                    logger.info(f"Using dark from {year} as fallback "
                                f"(not available for {self.observation_year})")
                    return result

        return None

    def _find_flat_in_memory(self, chip: str, fltr: str, binx: int) -> Optional[str]:
        """Search already-loaded flats for a match (no I/O)."""
        if (chip not in self.master_flats or
                fltr not in self.master_flats[chip] or
                binx not in self.master_flats[chip][fltr]):
            return None
        return self.master_flats[chip][fltr][binx]

    def find_flat(self, chip: str, fltr: str, binx: int) -> Optional[str]:
        """Find matching flat field, falling back to adjacent years if needed."""
        result = self._find_flat_in_memory(chip, fltr, binx)
        if result is not None:
            return result

        # Try adjacent years, closest first
        for year in self._year_search_order():
            _, flats = self._load_calibration_year(year)
            if flats > 0:
                result = self._find_flat_in_memory(chip, fltr, binx)
                if result is not None:
                    logger.info(f"Using flat from {year} as fallback "
                                f"(not available for {self.observation_year})")
                    return result

        return None
    
    def process_object(self, obj_path: str, output_dir: str = ".",
                      overwrite: bool = False,
                      photometry: bool = True) -> Optional[str]:
        """Process a single object frame.

        Performs dark/flat correction and, by default, a full solve+photometry
        pass (same iterative dophot as the automated pipeline).  The FITS and
        its paired ECSV are written to output_dir only when the astrometric
        quality check passes; on failure nothing is written.

        Pass photometry=False to get the raw dark/flat-corrected file only.

        Returns the output FITS path on success, None otherwise.
        """
        try:
            with fits.open(obj_path) as hdul:
                header = hdul[0].header
                data = hdul[0].data

                # Extract metadata
                chip = load_chip_id(header)
                c_time = header['CTIME']
                fltr = header['FILTER']
                usec = header['USEC']
                temp = header['CCD_TEMP']
                expt = header['EXPTIME']
                binx = self._get_header_value(header, 'BINX', 1)

                # Generate output filename
                datum = datetime.utcfromtimestamp(c_time)
                datestr = datum.strftime("%Y%m%d%H%M%S")
                output_name = f"{datestr}-{usec//1000:03d}-{fltr}-{expt:03.0f}-df.fits"
                output_path = Path(output_dir) / output_name

                if output_path.exists() and not overwrite:
                    logger.warning(f"{output_name}: Would overwrite existing file")
                    return None

                # Find calibration frames
                dark_path = self.find_best_dark(chip, temp, binx, expt)
                flat_path = self.find_flat(chip, fltr, binx)

                if dark_path is None:
                    logger.error(f"{output_name}: No suitable dark frame")
                    return None

                if flat_path is None:
                    logger.error(f"{output_name}: No suitable flat field")
                    return None

                # Load calibration data
                with fits.open(dark_path) as dark_hdul:
                    dark_data = dark_hdul[0].data

                with fits.open(flat_path) as flat_hdul:
                    flat_data = flat_hdul[0].data

                # Crop calibration frames to match a windowed science frame
                window = self._parse_ccdsec(header)
                if window is not None:
                    logger.info(f"{output_name}: windowed frame, cropping calibrations to {header.get('CCDSEC', 'LTV-derived')}")
                    dark_data = dark_data[window]
                    flat_data = flat_data[window]

                # Perform calibration
                calibrated = (data - dark_data) / flat_data

                # Crop to active area — only meaningful for full-frame readouts
                crop = None if window is not None else CAMERA_CROPS.get(chip)
                if crop is not None:
                    calibrated = calibrated[crop]
                    if 'CRPIX1' in header:
                        header['CRPIX1'] -= crop[1].start  # column offset
                    if 'CRPIX2' in header:
                        header['CRPIX2'] -= crop[0].start  # row offset

                output_hdu = fits.PrimaryHDU(data=calibrated, header=header)

                logger.debug(f"  {output_name}: mean={np.mean(calibrated):.1f}, "
                             f"std={np.std(calibrated):.1f}, median={np.median(calibrated):.1f}")

            # --- dark/flat only ---
            if not photometry:
                output_hdu.writeto(output_path, overwrite=overwrite)
                return str(output_path)

            # --- solve + photometry; write only on quality-checked success ---
            from asarina.pipeline.ingest import PhotometryPipeline  # local: avoids circular import
            pipeline = PhotometryPipeline(
                dophot_model=self.dophot_model,
                dophot_catalog=self.dophot_catalog,
                dophot_maglim=self.dophot_maglim,
                dophot_enlarge=self.dophot_enlarge,
                dophot_terms=self.dophot_terms,
                dophot_idlimit=self.dophot_idlimit,
                dophot_max_stars=self.dophot_max_stars,
            )

            with tempfile.TemporaryDirectory() as tmp_str:
                tmp = Path(tmp_str)
                output_hdu.writeto(str(tmp / output_name))

                if not pipeline.solve(output_name, tmp):
                    logger.warning(f"{output_name}: astrometric solve failed — skipped")
                    return None

                ecsv_name = pipeline.photometry(output_name, tmp,
                                                sip=self.sip, passes=self.passes)
                if ecsv_name is None:
                    logger.warning(f"{output_name}: photometry/quality check failed — skipped")
                    return None

                # pyrt-dophot writes the astrometrised image as {base}t.fits, not in-place.
                dft_name = output_name.replace('.fits', 't.fits')
                if not (tmp / dft_name).exists():
                    logger.warning(f"{dft_name}: astrometrised FITS not found — skipped")
                    return None

                ecsv_output = output_path.with_suffix('.ecsv')
                if ecsv_output.exists() and not overwrite:
                    logger.warning(f"{ecsv_output.name}: Would overwrite existing file")
                    return None

                # Rename dft.fits → df.fits so FITS and ECSV share the same stem.
                shutil.move(str(tmp / dft_name), str(output_path))
                shutil.move(str(tmp / ecsv_name), str(ecsv_output))

            logger.info(f"Wrote {output_path.name} + {ecsv_output.name}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error processing {obj_path}: {e}")
            return None
    
    def process_all_objects(self, output_dir: str = ".",
                           overwrite: bool = False,
                           photometry: bool = True) -> List[str]:
        """Process all loaded object frames.

        Returns:
            List of successfully processed output files
        """
        successful = []

        for obj_path in self.objects:
            result = self.process_object(obj_path, output_dir, overwrite, photometry)
            if result:
                successful.append(result)

        logger.debug(f"Successfully processed {len(successful)}/{len(self.objects)} frames")
        return successful


def _process_via_pipeline(pipeline, image_path: Path, output_dir: Path,
                          no_photometry: bool, overwrite: bool,
                          sip: int = 1, passes: int = 2) -> None:
    """Process one image through PhotometryPipeline, writing results to output_dir.

    Used when --smart-dark or --makak are given (the ImageProcessor master-dark
    path is bypassed entirely).
    """
    from asarina.pipeline.ingest import PhotometryPipeline  # already imported by caller
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)

        fits_file = pipeline.calibrate(image_path, tmp)
        if fits_file is None:
            return

        output_fits = output_dir / fits_file

        if no_photometry:
            shutil.copy(str(tmp / fits_file), str(output_fits))
            logger.info(f"Wrote {output_fits}")
            return

        if not pipeline.solve(fits_file, tmp):
            logger.warning(f"{fits_file}: astrometric solve failed — skipped")
            return

        ecsv_name = pipeline.photometry(fits_file, tmp, sip=sip, passes=passes)
        if ecsv_name is None:
            logger.warning(f"{fits_file}: photometry/quality check failed — skipped")
            return

        dft_name = fits_file.replace('.fits', 't.fits')
        if not (tmp / dft_name).exists():
            logger.warning(f"{dft_name}: astrometrised FITS not found — skipped")
            return

        if output_fits.exists() and not overwrite:
            logger.warning(f"{output_fits.name}: already exists, use --overwrite")
            return

        ecsv_output = output_fits.with_suffix('.ecsv')
        shutil.move(str(tmp / dft_name), str(output_fits))
        shutil.move(str(tmp / ecsv_name), str(ecsv_output))
        logger.info(f"Wrote {output_fits.name} + {ecsv_output.name}")


def main():
    """Command line interface."""
    from asarina.config import pre_parse, load_config, as_argparse_defaults, SYSTEM_CONFIG_FILE, USER_CONFIG_FILE

    config_file, camera, remaining = pre_parse()
    cfg = load_config(config_file, camera)
    defaults = as_argparse_defaults(cfg)

    parser = argparse.ArgumentParser(
        description="Produce calibrated FITS files (dark/flat correction + optional solve/dophot)"
    )
    parser.add_argument('files', nargs='+', help='Input FITS files')
    parser.add_argument('-o', '--output-dir', default='.',
                       help='Output directory (default: current directory)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output files')
    parser.add_argument('--no-photometry', action='store_true',
                       help='Skip solve+photometry; write dark/flat-corrected FITS only')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')

    std = parser.add_argument_group('standard calibration (master dark/flat)')
    std.add_argument('--temp-grouping', type=int, default=5,
                     help='Temperature grouping for darks (default: 5)')
    std.add_argument('--exposure-tolerance', type=float, default=3.0,
                     help='Max exposure time difference for dark matching (default: 3.0)')
    std.add_argument('--calib-dir', default='/home/mates/flat{year}/',
                     help='Calibration directory template (default: /home/mates/flat{year}/)')

    alt = parser.add_argument_group('alternative calibration / camera')
    alt.add_argument('--smart-dark', metavar='CALIB.npy',
                     help='Per-pixel dark model; bypasses master dark+flat entirely')
    alt.add_argument('--makak', action='store_true',
                     help='Makak camera: dark-frame detection, 55"/px hint, mi0315 crop')
    alt.add_argument('--pixel-scale', type=float, metavar='ARCSEC',
                     help='Pixel scale hint for pyrt-field-solve (arcsec/px)')

    phot = parser.add_argument_group('photometry (pyrt-dophot)')
    phot.add_argument('--sip', type=int, default=1, metavar='N',
                      help='SIP polynomial order for pyrt-dophot (default: 1)')
    phot.add_argument('--passes', type=int, default=2, metavar='N',
                      help='Number of pyrt-dophot passes (default: 2)')
    phot.add_argument('--dophot-model', metavar='FILE')
    phot.add_argument('--dophot-catalog', metavar='NAME')
    phot.add_argument('--dophot-maglim', type=float, metavar='N')
    phot.add_argument('--dophot-enlarge', type=float, metavar='N')
    phot.add_argument('--dophot-terms', metavar='TERMS')
    phot.add_argument('--dophot-idlimit', type=int, metavar='N')
    phot.add_argument('--dophot-max-stars', type=int, default=1000, metavar='N')

    cfg_grp = parser.add_argument_group('configuration')
    cfg_grp.add_argument('--config', metavar='FILE',
                         help=f'Config file (overrides cascade: {SYSTEM_CONFIG_FILE}, {USER_CONFIG_FILE})')
    cfg_grp.add_argument('--camera', metavar='NAME',
                         help='Camera section in config to apply '
                              '(default: auto-detected from CCD_NAME header)')

    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining)

    logging.basicConfig(
        stream=sys.stderr,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smart_dark or args.makak:
        # Smart-dark / Makak: ImageProcessor master-dark path is wrong here;
        # route everything through PhotometryPipeline which owns the smart-dark branch.
        from asarina.pipeline.ingest import PhotometryPipeline
        pipeline = PhotometryPipeline(
            smart_dark_calib=args.smart_dark,
            makak_mode=args.makak,
            pixel_scale=args.pixel_scale,
            dophot_model=args.dophot_model,
            dophot_catalog=args.dophot_catalog,
            dophot_maglim=args.dophot_maglim,
            dophot_enlarge=args.dophot_enlarge,
            dophot_terms=args.dophot_terms,
            dophot_idlimit=args.dophot_idlimit,
            dophot_max_stars=args.dophot_max_stars,
        )
        for f in args.files:
            _process_via_pipeline(pipeline, Path(f), output_dir,
                                  args.no_photometry, args.overwrite,
                                  sip=args.sip, passes=args.passes)
    else:
        # Standard path: ImageProcessor finds and applies master darks/flats,
        # then optionally hands off to PhotometryPipeline for solve+dophot.
        processor = ImageProcessor(
            temp_grouping=args.temp_grouping,
            exposure_tolerance=args.exposure_tolerance,
            calib_dir_template=args.calib_dir,
            sip=args.sip,
            passes=args.passes,
            dophot_model=args.dophot_model,
            dophot_catalog=args.dophot_catalog,
            dophot_maglim=args.dophot_maglim,
            dophot_enlarge=args.dophot_enlarge,
            dophot_terms=args.dophot_terms,
            dophot_idlimit=args.dophot_idlimit,
            dophot_max_stars=args.dophot_max_stars,
        )
        processor.load_calibration_frames(args.files)
        processor.process_all_objects(args.output_dir, args.overwrite,
                                      photometry=not args.no_photometry)


if __name__ == "__main__":
    main()
