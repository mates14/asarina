#!/usr/bin/env python3
"""Distortion playground: one Makak frame through the realtime pipeline, then plots.

    makak-distortion.py FRAME.fits OUTDIR [extra pyrt-dophot options...]

Calibrates and solves FRAME exactly as the service does (etc/asarina.conf), keeps
the working files in OUTDIR, and runs a final pyrt-dophot -p pass on the result
with the extra options appended (e.g. -S2, -z, --zpn-from-camera, -U ...), so
that OUTDIR/*-arrows.png and the residual plots show what a change to the model
does.  Rerun with other options to compare; the calibrated frame is reused.
"""

import subprocess
import sys
from pathlib import Path

from asarina.config import load_config, filter_terms
from asarina.pipeline.ingest import PhotometryPipeline

CONFIG = Path(__file__).resolve().parent.parent / 'etc' / 'asarina.conf'


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    frame, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    extra = sys.argv[3:]
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(CONFIG))

    pipeline = PhotometryPipeline(
        smart_dark_calib=cfg['smart_dark'], pixel_scale=cfg.get('pixel_scale'),
        phcat_fwhm=cfg.get('phcat_fwhm'), phcat_back_size=cfg.get('phcat_back_size'),
        phcat_aperture=cfg.get('phcat_aperture'),
        dophot_model=cfg.get('dophot_model'), dophot_catalog=cfg.get('dophot_catalog'),
        dophot_maglim=cfg.get('dophot_maglim'), dophot_enlarge=cfg.get('dophot_enlarge'),
        dophot_terms=cfg.get('dophot_terms'), dophot_idlimit=cfg.get('dophot_idlimit'),
        dophot_max_stars=cfg.get('dophot_max_stars'),
        dophot_zpn_from_camera=cfg.get('zpn_from_camera', False),
        dophot_filter_terms=filter_terms(cfg),
        wcs_guess_dir=cfg.get('wcs_guess_dir'), makak_mode=True)

    calibrated = sorted(out.glob('*-df.fits'))
    if calibrated:
        fits_file = calibrated[0].name
        print(f"reusing {fits_file}")
    else:
        fits_file = pipeline.calibrate(frame, out)
        if fits_file is None or not pipeline.solve(fits_file, out):
            sys.exit(f"calibrate/solve failed: {pipeline.failure}")
        if pipeline.photometry(fits_file, out, passes=cfg.get('passes', 2)) is None:
            print(f"service pipeline rejects this frame: {pipeline.failure}", file=sys.stderr)

    # The plotting pass: the service's dophot command plus -p and the extras
    ecsv = fits_file.replace('.fits', '.ecsv')
    src = ecsv if (out / ecsv).exists() else fits_file.replace('.fits', '.det')
    cmd = ["pyrt-dophot", "-m0.5", "-a", "-p", "-k",
           "-U", cfg['dophot_terms'], f"-i{cfg['dophot_idlimit']}",
           "-M", cfg['dophot_model'], "-C", cfg['dophot_catalog'],
           f"-l{cfg['dophot_maglim']}", f"-e{cfg['dophot_enlarge']}", "-S2"] + extra + [src]
    print(' '.join(cmd))
    ret = subprocess.run(cmd, cwd=str(out), capture_output=True, text=True)
    for line in (ret.stdout + ret.stderr).splitlines():
        if 'quality' in line or 'WSSR/NDF' in line or 'SIP order' in line:
            print(line)
    for png in sorted(out.glob('*.png')):
        print(png)


if __name__ == '__main__':
    main()
