#!/usr/bin/env python3

import argparse
import os
import sys
import subprocess
import tempfile
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple
import logging
from datetime import datetime

from astropy.io import fits
from asarina.pipeline.proc_images import ImageProcessor

logger = logging.getLogger(__name__)


class PhotometryPipeline:
    """Complete photometry pipeline from raw images to ECSV catalogs."""

    def __init__(self,
                 phdb_root: str = "~/phdb",
                 png_root: str = "~/png",
                 calib_root: str = "/home/mates/calib",
                 calib_dir_template: str = "/home/mates/flat{year}/"):

        self.phdb_root = Path(phdb_root).expanduser()
        self.png_root = Path(png_root).expanduser()
        self.calib_root = calib_root
        self.calib_dir_template = calib_dir_template
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
        return datetime.fromtimestamp(ctime).strftime("%y%m")

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

    def calibrate(self, image_path: Path, temp_dir: Path) -> Optional[str]:
        """Apply dark/flat correction.

        Returns the basename of the calibrated FITS file inside temp_dir,
        or None on failure.
        """
        processor = ImageProcessor(calib_root=self.calib_root,
                                   calib_dir_template=self.calib_dir_template)
        processor.load_calibration_frames([str(image_path)])
        processed = processor.process_all_objects(output_dir=str(temp_dir), overwrite=True)
        if not processed:
            logger.error(f"Dark/flat correction produced no output for {image_path.name}")
            return None
        return Path(processed[0]).name

    def solve(self, fits_file: str, temp_dir: Path) -> bool:
        """Source detection + astrometric solve.

        Runs pyrt-phcat then pyrt-field-solve.  On success the calibrated
        FITS file in temp_dir contains a valid WCS.  No ECSV is produced here.

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
        t = time.time()
        ret = subprocess.run(
            ["pyrt-field-solve", fits_file],
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

    def photometry(self, fits_file: str, temp_dir: Path) -> Optional[str]:
        """Photometry: catalog matching + dophot with astrometry refit.

        Runs pyrt-cat2det to match the catalog to detections, then pyrt-dophot
        on the resulting .det file (which also refits the astrometry).
        Returns the ECSV basename on success, None on failure.
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

        # Photometry + astrometry refit on .det
        t = time.time()
        ret = subprocess.run(
            ["pyrt-dophot", "-m0.5", "-azS1", "-U", ".r3,.p2", "-i2", det_file],
            cwd=str(temp_dir), capture_output=True, text=True,
        )
        elapsed = time.time() - t
        if ret.returncode != 0:
            out = (ret.stdout or '').rstrip()
            err = (ret.stderr or '').rstrip()
            logger.error(f"pyrt-dophot failed after {elapsed:.3f}s"
                         + (f"\nstdout:\n{out}" if out else '')
                         + (f"\nstderr:\n{err}" if err else ''))
            return None
        if ret.stdout:
            logger.debug(f"pyrt-dophot stdout:\n{ret.stdout.rstrip()}")
        if ret.stderr:
            logger.debug(f"pyrt-dophot stderr:\n{ret.stderr.rstrip()}")
        logger.info(f"pyrt-dophot took {elapsed:.3f}s")

        if not (temp_dir / ecsv_file).exists():
            logger.error(f"ECSV {ecsv_file} missing after pyrt-dophot")
            return None

        return ecsv_file

    def save_results(self, ecsv_filename: str, temp_dir: Path, ctime: float,
                     keep_image: bool = False) -> Optional[str]:
        """Move ECSV and PNG files to their permanent locations.

        Returns the path to the stored ECSV, or None on failure.
        """
        ym = self._get_year_month_code(ctime)
        phdb_dir = self.phdb_root / ym
        png_dir = self.png_root / ym
        phdb_dir.mkdir(parents=True, exist_ok=True)
        png_dir.mkdir(parents=True, exist_ok=True)

        dst_ecsv = phdb_dir / ecsv_filename
        shutil.move(str(temp_dir / ecsv_filename), str(dst_ecsv))
        logger.debug(f"ECSV: {dst_ecsv}")

        for png in temp_dir.glob("*.png"):
            shutil.move(str(png), str(png_dir / png.name))

        if keep_image:
            updated = ecsv_filename.replace('.ecsv', 'dft.fits')
            src = temp_dir / updated
            if src.exists():
                dst = Path.cwd() / updated
                shutil.copy2(str(src), str(dst))
                logger.info(f"Kept calibrated image: {dst}")

        return str(dst_ecsv)

    # ------------------------------------------------------------------
    # High-level entry point (used by c0_pipeline)
    # ------------------------------------------------------------------

    def process_image(self, image_path: str, force: bool = False,
                      keep_image: bool = False) -> Optional[str]:
        """Process a single image through the complete pipeline.

        Returns path to the output ECSV, or None on failure.
        """
        image_path = Path(image_path)
        logger.debug(f"××××× {image_path.name} ×××××")

        if not force:
            existing = self._check_existing_ecsv(str(image_path))
            if existing:
                logger.info(f"Result already exists: {existing}")
                return existing

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

            return self.save_results(ecsv, temp_dir, ctime, keep_image)

    def process_images(self, image_paths: List[str], force: bool = False,
                       keep_image: bool = False) -> List[str]:
        """Process multiple images. Returns list of created ECSV paths."""
        successful = []
        for p in image_paths:
            result = self.process_image(p, force, keep_image)
            if result:
                successful.append(result)
        logger.info(f"Successfully processed {len(successful)}/{len(image_paths)} images")
        return successful


def main():
    parser = argparse.ArgumentParser(
        description="Complete photometry pipeline: calibration + dophot3"
    )
    parser.add_argument('images', nargs='+', help='Input FITS images')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Redo even if ECSV exists')
    parser.add_argument('-i', '--keep-image', action='store_true',
                        help='Keep calibrated FITS after processing')
    parser.add_argument('--phdb-root', default='~/phdb')
    parser.add_argument('--png-root', default='~/png')
    parser.add_argument('--calib-dir', default='/home/mates/flat{year}/')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(levelname)s: %(message)s')

    pipeline = PhotometryPipeline(
        phdb_root=args.phdb_root,
        png_root=args.png_root,
        calib_dir_template=args.calib_dir,
    )
    pipeline.process_images(args.images, args.force, args.keep_image)


if __name__ == "__main__":
    main()
