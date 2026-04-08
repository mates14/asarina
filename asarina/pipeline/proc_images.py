#!/usr/bin/env python3

import argparse
import os
import sys
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
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ImageProcessor:
    """Astronomical image processing pipeline for dark/flat calibration."""
    
    def __init__(self, temp_grouping: int = 5, exposure_tolerance: float = 3.0,
                 calib_root: str = "/home/mates/calib",
                 calib_dir_template: str = "/home/mates/flat{year}/",
                 max_year_search: int = 5):
        self.temp_grouping = temp_grouping
        self.exposure_tolerance = exposure_tolerance
        self.calib_root = Path(calib_root)
        self.calib_dir_template = calib_dir_template
        self.max_year_search = max_year_search
        self.master_darks = {}
        self.master_flats = {}
        self.objects = []
        self.loaded_files = set()  # Track loaded files to avoid duplicates
        self.loaded_calib_years = set()  # Track which years' calib dirs were loaded
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
                      overwrite: bool = False) -> Optional[str]:
        """Process a single object frame.
        
        Returns:
            Output filename if successful, None otherwise
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
                
                # Perform calibration
                calibrated = (data - dark_data) / flat_data

                # Crop to active area if defined for this camera
                crop = CAMERA_CROPS.get(chip)
                if crop is not None:
                    calibrated = calibrated[crop]
                    # crop is (row_slice, col_slice); adjust reference pixel accordingly
                    if 'CRPIX1' in header:
                        header['CRPIX1'] -= crop[1].start  # column offset
                    if 'CRPIX2' in header:
                        header['CRPIX2'] -= crop[0].start  # row offset

                # Write output
                output_hdu = fits.PrimaryHDU(data=calibrated, header=header)
                output_hdu.writeto(output_path, overwrite=overwrite)
                
                # Log statistics
                mean_val = np.mean(calibrated)
                std_val = np.std(calibrated)
                median_val = np.median(calibrated)
                
                logger.debug(f"  {output_name}: mean={mean_val:.1f}, std={std_val:.1f}, "
                           f"median={median_val:.1f}")
                
                return str(output_path)
                
        except Exception as e:
            logger.error(f"Error df correcting 1 {obj_path}: {e}")
            return None
    
    def process_all_objects(self, output_dir: str = ".", 
                           overwrite: bool = False) -> List[str]:
        """Process all loaded object frames.
        
        Returns:
            List of successfully processed output files
        """
        successful = []
        
        for obj_path in self.objects:
            result = self.process_object(obj_path, output_dir, overwrite)
            if result:
                successful.append(result)
        
        logger.debug(f"Successfully df corrected {len(successful)}/{len(self.objects)} frames")
        return successful


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Correct astronomical images with darks+flats"
    )
    parser.add_argument('files', nargs='+', help='Input FITS files')
    parser.add_argument('-o', '--output-dir', default='.',
                       help='Output directory (default: current directory)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output files')
    parser.add_argument('--temp-grouping', type=int, default=5,
                       help='Temperature grouping for darks (default: 5)')
    parser.add_argument('--exposure-tolerance', type=float, default=3.0,
                       help='Maximum exposure time difference for dark matching (default: 3.0)')
    parser.add_argument('--calib-dir', default='/home/mates/flat{year}/',
                       help='Calibration directory template with {year} placeholder '
                            '(default: /home/mates/flat{year}/)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create processor
    processor = ImageProcessor(
        temp_grouping=args.temp_grouping,
        exposure_tolerance=args.exposure_tolerance,
        calib_dir_template=args.calib_dir
    )
    
    # Load calibration frames
    processor.load_calibration_frames(args.files)
    
    # Process objects
    processor.process_all_objects(args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()
