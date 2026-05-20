# asarina

Operational tooling for the pyrt photometry pipeline: calibration frame creation and real-time image processing. Depends on [pyrt](../pyrt) being installed.

## Components

### `asarina.chip_id`

Camera identification registry. Maps FITS header fields (`CCD_SER`, `CCD_TYPE`, etc.) to short camera IDs (e.g. `mi6166`, `andor46`, `fli534`). Used by both calib and pipeline. Single source of truth for all camera identification.

### `asarina.calib`

Creates master dark and flat-field frames from raw calibration images.

- `make_calib.py` — main orchestrator: scans an archive for raw frames by year and camera, filters by quality, pairs darks with flats, produces master calibration frames. Uses a SQLite cache for frame statistics to avoid recomputation.
- `mixdark.py` / `mixflat.py` — combine sets of dark/flat frames into masters.
- `iraf_combine_dark.sh` / `iraf_combine_flat.sh` — IRAF-based frame combination scripts.

Output structure: `~/calib/{camera_id}/{year}/`

### `asarina.pipeline`

Real-time image processing pipeline for automated nightly observations.

- `imgproc.py` — single-image processor: dark/flat → astrometry → dophot → ECSV/PNG + corrwerr to stdout for RTS2. Called directly by the RTS2 imgproc daemon.
- `watch.py` — file-watching daemon: polls for new FITS images from a camera, hands each to `PhotometryPipeline`, uploads results. Integrates with systemd (watchdog, journal logging).
- `ingest.py` — `PhotometryPipeline` class: dark/flat → astrometry → dophot → ECSV + DB upload. CLI entry point (`asarina-ingest`) for batch photometry on a list of files.
- `image.py` — `ImageProcessor` class: dark/flat correction + optional solve/photometry. CLI entry point (`asarina-image`) for producing calibrated FITS files.
- `pipeline_utils.py` — shared utilities: `HealthChecker`, `PngCleaner`, `setup_logging`.
- `transient_daemon.py` — monitors ECSV output for transient events and triggers follow-up.

## Installation

```
pip install -e /home/mates/asarina
```

pyrt must be installed first:

```
pip install -e /home/mates/pyrt
```

## CLI entry points

| Command | Entry point |
|---|---|
| `asarina-make-calib` | `asarina.calib.make_calib:main` |
| `asarina-image` | `asarina.pipeline.image:main` |
| `asarina-ingest` | `asarina.pipeline.ingest:main` |
| `asarina-watch` | `asarina.pipeline.watch:main` |
| `asarina-imgproc` | `asarina.pipeline.imgproc:main` |

## systemd

Service unit files are in `systemd/`:

- `c0-pipeline.service` — runs `asarina-watch`, restarts on failure, 2 GB RAM / 80% CPU limits, watchdog every 120 s.

Copy to `/etc/systemd/system/` and run `systemctl enable --now <service>`.
