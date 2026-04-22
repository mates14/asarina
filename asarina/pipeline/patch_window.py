#!/usr/bin/env python3
"""
Patch FITS headers with windowing keywords for RTS2 frames that lack them.

RTS2 centers the window on the full detector (including overscan) using:
    x0 = chip_width  // 2 - window_size // 2   (0-indexed)
    y0 = chip_height // 2 - window_size // 2

Window size is inferred from NAXIS1/NAXIS2 (must be square: 512 or 1024).
Use --shift to test ±1 pixel offsets until calibration frames align correctly.
"""
import argparse
import sys
from pathlib import Path
from astropy.io import fits

# Full detector size including overscan (RTS2 has no overscan concept,
# so this is simply the full readout frame)
DEFAULT_CHIP_WIDTH  = 4144
DEFAULT_CHIP_HEIGHT = 4127

KNOWN_WINDOW_SIZES = (512, 1024)


def compute_keywords(window_size: int, chip_w: int, chip_h: int, shift: int) -> dict:
    x0 = chip_w // 2 - window_size // 2  # RTS2 0-indexed start
    y0 = chip_h // 2 - window_size // 2

    # shift=0 assumes RTS2 is 0-indexed → FITS 1-indexed start = x0 + 1
    fits_x1 = x0 + 1 + shift
    fits_y1 = y0 + 1 + shift
    fits_x2 = fits_x1 + window_size - 1
    fits_y2 = fits_y1 + window_size - 1

    return {
        'LTV1':   (float(fits_x1 - 1), 'X offset in CCD frame'),
        'LTV2':   (float(fits_y1 - 1), 'Y offset in CCD frame'),
        'LTM1_1': (1.0,                'X scale (1/binning)'),
        'LTM2_2': (1.0,                'Y scale (1/binning)'),
        'CCDSEC': (f'[{fits_x1}:{fits_x2},{fits_y1}:{fits_y2}]', 'CCD region read (1-indexed)'),
        'DATASEC':(f'[1:{window_size},1:{window_size}]',          'Image array region'),
        'DETSIZE':(f'[1:{chip_w},1:{chip_h}]',                    'Full detector size'),
    }


def patch_file(filepath: str, window_size: int, chip_w: int, chip_h: int,
               shift: int, dry_run: bool):
    keywords = compute_keywords(window_size, chip_w, chip_h, shift)

    print(f"\n{filepath}  (window={window_size}, chip={chip_w}x{chip_h}, shift={shift:+d})")
    for key, (val, comment) in keywords.items():
        print(f"  {key:8s} = {str(val):30s}  / {comment}")

    if dry_run:
        print("  [dry run — not written]")
        return

    with fits.open(filepath, mode='update') as hdul:
        h = hdul[0].header
        for key, (val, comment) in keywords.items():
            h[key] = (val, comment)
        hdul.flush()
    print("  written.")


def main():
    parser = argparse.ArgumentParser(
        description="Insert window keywords into RTS2 FITS files that lack them",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
defaults:
  chip size  {DEFAULT_CHIP_WIDTH}x{DEFAULT_CHIP_HEIGHT} (full readout including overscan)
  shift      0  (assumes RTS2 uses 0-indexed pixels → FITS start = x0+1)

centering on {DEFAULT_CHIP_WIDTH}x{DEFAULT_CHIP_HEIGHT} with shift=0:
  1024-window: x0={DEFAULT_CHIP_WIDTH//2-512}, y0={DEFAULT_CHIP_HEIGHT//2-512}
               CCDSEC [{DEFAULT_CHIP_WIDTH//2-512+1}:{DEFAULT_CHIP_WIDTH//2+512},{DEFAULT_CHIP_HEIGHT//2-512+1}:{DEFAULT_CHIP_HEIGHT//2+512}]
  512-window:  x0={DEFAULT_CHIP_WIDTH//2-256}, y0={DEFAULT_CHIP_HEIGHT//2-256}
               CCDSEC [{DEFAULT_CHIP_WIDTH//2-256+1}:{DEFAULT_CHIP_WIDTH//2+256},{DEFAULT_CHIP_HEIGHT//2-256+1}:{DEFAULT_CHIP_HEIGHT//2+256}]

if calibration crop is off by one pixel, retry with --shift -1 or --shift +1.
        """
    )
    parser.add_argument('files', nargs='+', help='FITS files to patch')
    parser.add_argument('--chip-width',  type=int, default=DEFAULT_CHIP_WIDTH,
                        help=f'Full detector width  (default: {DEFAULT_CHIP_WIDTH})')
    parser.add_argument('--chip-height', type=int, default=DEFAULT_CHIP_HEIGHT,
                        help=f'Full detector height (default: {DEFAULT_CHIP_HEIGHT})')
    parser.add_argument('--window-size', type=int, choices=KNOWN_WINDOW_SIZES,
                        help='Force window size instead of inferring from NAXIS1/NAXIS2')
    parser.add_argument('--shift', type=int, default=0,
                        help='Offset to add to FITS start coordinate (default: 0)')
    parser.add_argument('-n', '--dry-run', action='store_true',
                        help='Show what would be written without modifying files')
    args = parser.parse_args()

    errors = 0
    for filepath in args.files:
        if not Path(filepath).exists():
            print(f"ERROR: {filepath}: not found", file=sys.stderr)
            errors += 1
            continue

        try:
            with fits.open(filepath) as hdul:
                h = hdul[0].header
                naxis1 = h.get('NAXIS1')
                naxis2 = h.get('NAXIS2')

            if args.window_size:
                window_size = args.window_size
            elif naxis1 == naxis2 and naxis1 in KNOWN_WINDOW_SIZES:
                window_size = naxis1
            else:
                print(f"ERROR: {filepath}: NAXIS1={naxis1} NAXIS2={naxis2} — "
                      f"not a recognised square window size, use --window-size",
                      file=sys.stderr)
                errors += 1
                continue

            patch_file(filepath, window_size, args.chip_width, args.chip_height,
                       args.shift, args.dry_run)

        except Exception as e:
            print(f"ERROR: {filepath}: {e}", file=sys.stderr)
            errors += 1

    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
