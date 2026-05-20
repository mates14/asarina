# asarina usage

## Tools at a glance

| Command | What it does |
|---|---|
| `asarina-imgproc` | Process one image: calibrate → solve → dophot → ECSV. The core processor — everything else is built on top of it. |
| `asarina-watch` | Daemon: watches a directory for new images, calls `asarina-imgproc` per file. |
| `asarina-image` | Human peek: produce dark/flat-corrected FITS files (optionally with solve+dophot). |
| `asarina-make-calib` | Build master dark and flat-field frames from a raw calibration archive. |
| `asarina-patch-window` | Fix missing LTV1/2 windowing keywords in SBT windowed frames. |

---

## asarina-imgproc — single image processor

The core tool. Takes one raw FITS, runs the full pipeline, writes ECSV to `~/phdb` and PNG to `~/png`.

### Offline / human use (no RTS2)

```bash
asarina-imgproc image.fits
```

Calibrates, solves, runs dophot, saves ECSV and PNG. No corrwerr, no web preview, no WCS written back to the raw file. Upload to DB requires `--ssh-key`:

```bash
asarina-imgproc --ssh-key ~/.ssh/id_pipeline image.fits
```

Skip if ECSV already exists (default). Force reprocessing:

```bash
asarina-imgproc -f image.fits
```

### RTS2 real-time mode

Called by the RTS2 `imgproc` daemon. `--realtime` enables the three RTS2-specific steps: corrwerr to stdout, WCS written back to the raw image, web preview generation. Nothing else may appear on stdout.

```bash
asarina-imgproc --realtime --ssh-key ~/.ssh/id_pipeline image.fits
```

### SBT-specific options

SBT frames may be windowed (missing LTV1/2 keywords) and benefit from higher-order astrometry and more dophot passes:

```bash
asarina-imgproc --sbt-window-patch --sip 2 --passes 3 image.fits
```

### Makak

```bash
asarina-imgproc \
  --smart-dark /home/mates/makak-reloaded/makak-dark-response.npy \
  --makak \
  --phdb-root /home/mates/makak-reloaded/ecsv \
  --phdb-date-fmt '%Y%m%d' \
  --daily-summary /home/mates/makak-reloaded/nght \
  --dophot-model /home/mates/makak-reloaded/model.mod \
  --dophot-catalog makak \
  --dophot-maglim 8 \
  --dophot-enlarge 1.5 \
  --dophot-terms '.p4,.r4,RC,RO,RS' \
  --dophot-idlimit 5 \
  --dophot-max-stars 0 \
  image.fits
```

### Batch reprocessing

`asarina-imgproc` takes one file at a time; use a shell loop:

```bash
for f in /images/2025/20251201/C0/*.fits; do
    asarina-imgproc --ssh-key ~/.ssh/id_pipeline "$f"
done
```

Or in parallel (GNU parallel):

```bash
find /images/2025 -name '*.fits' | \
    parallel -j4 asarina-imgproc --ssh-key ~/.ssh/id_pipeline {}
```

Already-processed files are skipped unless `-f` is given.

---

## asarina-watch — file-watching daemon

Polls for new images and hands each to `asarina-imgproc`. All imgproc flags are forwarded; the watcher adds its own polling/systemd/cleanup flags.

### D50 / C0 camera

```bash
asarina-watch \
  --camera-pattern C0 \
  --ssh-key ~/.ssh/id_pipeline \
  --systemd
```

With systemd, run via `c0-pipeline.service`. See `systemd/c0-pipeline.service`.

### Makak

```bash
asarina-watch \
  --search-root /storage/archive-images/MAKAK/images \
  --camera-pattern '' \
  --ssh-key ~/.ssh/id_makak \
  --smart-dark /home/mates/makak-reloaded/makak-dark-response.npy \
  --makak \
  --phdb-root /home/mates/makak-reloaded/ecsv \
  --phdb-date-fmt '%Y%m%d' \
  --daily-summary /home/mates/makak-reloaded/nght \
  --dophot-model /home/mates/makak-reloaded/model.mod \
  --dophot-catalog makak \
  --dophot-maglim 8 \
  --dophot-enlarge 1.5 \
  --dophot-terms '.p4,.r4,RC,RO,RS' \
  --dophot-idlimit 5 \
  --dophot-max-stars 0 \
  --max-workers 4
```

### Key watcher-only flags

| Flag | Default | Purpose |
|---|---|---|
| `--search-root DIR` | `/images` | Root of the image archive tree |
| `--camera-pattern PAT` | `C0` | Camera subdirectory name to look inside |
| `--poll-interval S` | `1.0` | Seconds between filesystem polls |
| `--max-workers N` | `3` | Parallel imgproc processes |
| `--png-cleanup-days N` | `7` | Delete PNGs older than N days |
| `--systemd` | off | Systemd journal logging + watchdog |
| `--realtime` | off | Forward `--realtime` to imgproc (web preview, corrwerr, WCS-to-raw) |

---

## asarina-image — get calibrated FITS files

For when you want to inspect or stack images yourself. Produces dark/flat-corrected FITS in an output directory. By default also runs solve+dophot and writes ECSV alongside; use `--no-photometry` for just the calibrated FITS.

```bash
# Dark/flat correction only
asarina-image --no-photometry -o ./corrected/ night/*.fits

# Full calibration + astrometry + dophot, keep FITS
asarina-image -o ./processed/ night/*.fits
```

---

## asarina-make-calib — build master calibration frames

Scans an archive for raw dark and flat frames, groups them by camera/temperature/exposure, and produces master frames under `~/calib/{camera_id}/{year}/`.

```bash
asarina-make-calib --year 2025 --camera mi6166
```

See `asarina-make-calib --help` for the full flag set (archive paths, temperature grouping, etc.).

---

## Flag reference for asarina-imgproc

### Output

| Flag | Default | Purpose |
|---|---|---|
| `--phdb-root DIR` | `/home/mates/phdb` | Root for ECSV output |
| `--phdb-date-fmt FMT` | `%y%m` | Subdirectory date format |
| `--png-root DIR` | `/home/mates/png` | Root for PNG output |
| `--daily-summary DIR` | off | Append nightly summary line to `mr{YYYYMMDD}.dat` |
| `--ssh-key FILE` | off | SSH key for DB upload (skipped if omitted) |

### Calibration

| Flag | Default | Purpose |
|---|---|---|
| `--smart-dark CALIB.npy` | off | Per-pixel analytical dark model; bypasses dark+flat database |
| `--sbt-window-patch` | off | Patch missing LTV1/2 keywords before calibration |

### Astrometry

| Flag | Default | Purpose |
|---|---|---|
| `--pixel-scale ARCSEC` | from header | Pixel scale hint for `pyrt-field-solve` |

### Photometry

| Flag | Default | Purpose |
|---|---|---|
| `--sip N` | `1` | SIP distortion order for dophot (SBT uses 2) |
| `--passes N` | `2` | Number of dophot passes (SBT uses 3) |
| `--dophot-model FILE` | none | Model file (`-M`) |
| `--dophot-catalog NAME` | none | Reference catalog (`-C`) |
| `--dophot-maglim N` | none | Magnitude limit (`-l`) |
| `--dophot-enlarge N` | none | Enlarge factor (`-e`) |
| `--dophot-terms TERMS` | `.r3,.p3,.l` | Uncertainty terms (`-U`) |
| `--dophot-idlimit N` | `2` | ID-limit iterations (`-i`) |
| `--dophot-max-stars N` | `1000` | Max stars (0 = no limit) |

### Camera bundles

| Flag | Purpose |
|---|---|
| `--makak` | Bundles: dark-frame detection, 55"/px scale hint, `-k` in dophot, `mi0315` crop |

### Behaviour

| Flag | Purpose |
|---|---|
| `-f / --force` | Reprocess even if ECSV already exists |
| `-r / --realtime` | RTS2 mode: corrwerr to stdout, WCS to raw, web preview |
| `-v / --verbose` | Debug logging |
