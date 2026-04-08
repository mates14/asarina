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

- `c0_pipeline.py` — main daemon: watches for new FITS images, runs the photometry chain, uploads results. Integrates with systemd (watchdog, journal logging).
- `c0_pipeline_single.py` — single-image variant of the pipeline.
- `get_ecsv.py` — wraps the pyrt photometry chain (`pyrt-phcat` → `pyrt-field-solve` → `pyrt-cat2det` → `pyrt-dophot`) into a `PhotometryPipeline` class; checks output quality.
- `proc_images.py` — applies dark and flat calibration to raw object frames; handles camera-specific cropping.
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
| `asarina-proc-images` | `asarina.pipeline.proc_images:main` |
| `asarina-pipeline` | `asarina.pipeline.c0_pipeline:main` |

## systemd

Service unit files are in `systemd/`:

- `c0-pipeline.service` — runs `c0_pipeline.py`, restarts on failure, 2 GB RAM / 80% CPU limits, watchdog every 120 s.

Copy to `/etc/systemd/system/` and run `systemctl enable --now <service>`.
