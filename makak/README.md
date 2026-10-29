# asarina + makak

Makak (mi0315 on morpheus, 10 s frames every 11.5 s) processed in real time by
asarina (main; the Makak work was branch `makak-realtime`, merged 2026-10-29),
replacing `~/test/crawl` + `~/test/makak.py` (and makak-reloaded's crawler).
This directory (`makak/` in the asarina repo) is the source of the deployment:
`deploy.sh` copies it to `~/asarina+makak` on morpheus (the service's runtime
directory: state, ECSVs, nightly summaries, the graph) and on zeus/hog. It produces the same products as makak-reloaded:

| Output | What |
|---|---|
| `ecsv/YYYYMMDD/*.ecsv` | pyrt-dophot catalog per frame, by night (UT−12 h) |
| `nght/mrYYYYMMDD.dat` | dophot summary line per frame: JD, IDNUM, zeropoint ± err, limit, WSSR/NDF, BGSIGMA, fitted CCD temp |
| `nght/mrYYYYMMDDd.dat` | dark frames (slitposx < 0.5): ctime, darksig, fitted temp |
| `state/mi0315.wcs` | last good WCS, the next frame's starting solution |

## Layout

- `etc/asarina.conf`: everything Makak-specific (smart dark, ZPN camera model, WCS
  guess, dophot settings as in makak-reloaded but `.p4,.r4,RS` (RC/RO made the zeropoint jump 0.3 mag), output dirs). Could become
  `/etc/asarina/config` on morpheus later.
- `systemd/makak-asarina.service`: `asarina-watch` with newest-only (`--latest-only`), 3 workers,
  180 s frame timeout, nice 10.
- `web/makak-web` + `systemd/makak-web.service`: whenever tonight's `nght/mr*.dat`
  change, `web/makak-plot.sh` (zeus's makak-reloaded graph, since 2026-10-27) draws
  zeropoint, sky brightness, CLOUD (TEMP_DIFF from D50's `/var/log/rts2-meteo-*.log`)
  and temperatures, and uploads `makak/makak.png`, `makak/mrpng/mrYYYYMMDD.png` and
  `makak/dat/mrYYYYMMDD.dat` to lascaux. The old graph and its files are in
  `~/old/makak-web-legacy` on morpheus.
- `deploy.sh code|files|units`: the code is a `git pull` of main (asarina and pyrt
  are editable installs: `~/src` on morpheus, `~/asarina` and `~/pyrt` in zeus's and
  hog's shared home); `files` copies this directory to `~/asarina+makak` on morpheus
  and in the zeus/hog home; `units` installs the systemd units on morpheus.
- `data/makak-model.mod`: dophot's model. The dark model (`makak-dark-response.npy`,
  48 MB) is not in the repo: `~/asarina+makak/data/` on morpheus and zeus/hog.
- `web/makak-plot.sh` uploads with morpheus's passwordless key, which lascaux allows
  only `rrsync -wo ~/public_html/makak` and lascaux50 only `rrsync -ro /var/log`
  (2026-10-29), hence the paths relative to those.
- `bin/makak-distortion.py FRAME OUTDIR [dophot opts]`: one frame through the service
  pipeline, then a `pyrt-dophot -p` pass with your options. It writes `*-arrows.png`
  and residual plots for working on the distortion model.
- `dev/`: `bench.sh`, `cloudbench.sh`, `sim.sh` (the tests below), `astcmp.py`,
  `cloudcmp.py`, `test.conf`. Their data (`testdata/`: 140 frames of 2026-10-24, every
  50th plus 80 consecutive from 22:00 UT; `bench/`, `sim/`) is not in the repo, only in
  the laptop's `~/asarina+makak`, where they run.

## What asarina needed (now in main)

- `imgproc --makak`: dark frames go to `mr*d.dat` and stop (previously they went to
  pyrt-field-solve, 160 s, then failed).
- `--wcs-guess DIR` / `wcs_guess_dir`: fixed camera, so the last accepted WCS with
  CRVAL1 advanced by the sidereal angle starts the next frame, with no field-solve.
  Rejected → field-solve fallback; every accepted solution updates the guess.
- `watch`: forwards only the options given, so the config file applies (it used to
  force `--sip 1 --passes 2 --dophot-max-stars 1000`). Also adds `--config/--camera`,
  optional `--ssh-key`, `--no-raw-update`, `--latest-only`, `--frame-timeout`
  (kills the process group) and `--min-age`, and fixes `shutdown(timeout=)` in `stop()`.
- config: `no_raw_update` is a bool key.
- Clouds (2026-10-26, moonlit passing clouds): `phcat_fwhm`/`phcat_back_size` go
  to pyrt-phcat (`-f`, `--back-size`, pyrt), since the
  256 px background mesh took the lit cloud blobs for 30-120 px stars.
  `wcs_guess_keep`: when stars through clouds cover too little of the chip
  to verify the refit (ASTQUAL=PARTIAL), photometry runs on the guess as it is
  instead of rejecting the frame. `min_idnum` replaces the fixed IDNUM > 20.

D50/SBT: every new path is gated by `--makak`, `wcs_guess_dir` or `--latest-only`.
The watcher's imgproc command for `c0-pipeline.service` loses only options equal
to imgproc's defaults.

## Test results (laptop, i7-8565U; morpheus is a Core2 Q9550)

8 consecutive frames, 22:00 UT (one dark):

| Setup | Sky frames OK | s/frame |
|---|---|---|
| makak-usage.md flags (field-solve TAN) | 5/7 (2× ASTQUAL=PARTIAL, CHI2C 3.3–4.3) | 18–26 |
| + `zpn_from_camera` | 7/7, ~30 more stars matched | 18–22 |
| + WCS guess (configured) | 7/7, identical photometry | 13–19 |
| + `passes = 1` | 7/7, one frame off by 0.3 mag | 10–13 |

Realtime simulation pinned to 2 cores, frames every 11.5 s: 40/40 processed
(4 darks), ~15 s latency, none skipped. Flood test at 2 s cadence: backlog
dropped, always the newest frame.

Clouds, `cloudbench.sh NAME [imgproc opts]` + `cloudcmp.py NAME...` (61 frames
of 2026-10-26 19:45-20:31 UT, moon and passing clouds, every 4th; 55 sky frames):

| Setup | With zeropoint | Stars (clear) | s/frame |
|---|---|---|---|
| as deployed first | 47 (8 astqual/dophot) | ~250-290 | 20 (max 55) |
| + `phcat_back_size 32`, `phcat_fwhm 4` | 53 | ~320 | 17.6 |
| + `wcs_guess_keep`, `min_idnum 10` (configured) | 55 | ~320 | 17.3 |
| same with `phcat_fwhm 5` | 55 | ~300 | 13.2 |

| + `phcat_aperture 4` (configured) | 55 | ~320 | 12.9 |
| + `sip 0` (configured) | 55, identical zeropoints | ~320 | 12.9 |

`sip = 1` took SIP 2 in pass 2 on 30 of 55 frames (5 of 14 clear) for ASTSIGMA
0.098 instead of 0.105 px and no change in the matches or the zeropoint, so the
astrometry is stiff: camera ZPN fixed, CRVAL and CD refitted (`astcmp.py`).
The fixed 4 px aperture: clear-sky zeropoint 18.62, MAD 0.007.
`phcat_noiraf` (pyrt-phcat `-I`, 2026-10-27, configured): SExtractor's MAG_APER in the
same 4 px radius with a local sky annulus (BACKPHOTO_THICK 12 px; the global
background map left 0.15 mag scatter through clouds) agrees with IRAF phot to
0.017 mag per star on clear sky and 0.036 through clouds; zeropoint MAD 0.008 and
0.031, 0.015 mag lower. On morpheus 27.1 instead of 29.0 s per frame (24 cloud
frames, one at a time).
`-S0` needed a pyrt fix (2d639f0): SIP order 0 is a
fit with nothing free, which scipy 1.16 passes through and scipy 1.11 (morpheus)
crashed on, so every frame failed there 00:14-00:18 CET on 2026-10-27.

The cloud passage reads as the zeropoint going from 18.0 to 18.6. On clear
frames (2026-10-24) `phcat_fwhm 4` gives a zeropoint 0.1 mag below the
estimated FWHM's and 5 matches it, but the estimate itself jumped 2.9-5.7 px.

## Sky map (2026-10-27)

`asarina-skymap` as on SBT, with `--profile makak` (asarina 5b42746): one band
(G, as calibrated), HEALPix nside 8 cells, 51.5"/px TAN tiles (HiPS order 3),
64 px background boxes. Only clean frames go in: ASTQUAL and PHQUAL OK,
FWHM <= 6, DMAGZERO <= 0.03, MAGZERO >= 18.45, Sun below -12 deg (asarina
39c086f; clouds and twilight stay out). The clear-sky zeropoint moves with the
aerosols, so 18.45 may need changing on other nights. The weight is 1/sky
noise^2 scaled by dophot's per-pixel zeropoint model, so moonlit frames count less. The service adds every such frame
to `/storage/makak-skymap` (HDD: one frame rewrites ~21 tiles, ~70 MB). It costs
about 3 s per frame on morpheus (1.2 s on the laptop). It needs
`python3-astropy-healpix` (apt). Morpheus has no Java, so the HiPS is built on hog
(`ssh h`) by `bin/makak-hips.sh` from hog's crontab at 08:00. It rsyncs the tiles with
`~/.ssh/hog-morpheus-rsync`, which morpheus allows only `rrsync -ro
/storage/makak-skymap`, and builds `/storage2/reprocess/makak/{skymap,export,hips}`
next to SBT's (pixelCut -20000 300000 asinh, asarina 74869b4; log in `hips.log`).
Serving it is still to be done (SBT's is served by a Python httpd per request).

The field covers Dec ~63-84 (not the zenith). Night 2026-10-26 was reprocessed
on zeus (2026-10-27, `bin/makak-rep-one.sh`, 20 frames in parallel, 3708 frames
in 1.5 h) and
`asarina-skymap merge`d into the map: 2994 frames with a zeropoint (2479 live),
rerun with the stricter gates: 1605 frames into the map. The live outputs of
that night are in `~/old/live-20261026` on morpheus; the reprocessing directories on
zeus and the first map were deleted.

## Reprocessing the archive (zeus, from 2026-10-29)

The whole Makak archive on zeus (`/storage/archive-images/MAKAK/images`, 895
nights 2021-05-31..2026-10-27, 5.2 TB, ~1.3 M frames) goes through
`asarina-reprocess`, as D50 and SBT do on hog, newest night first:
`bin/makak-reprocess.sh start|stop|report|list|retry` with
`etc/reprocess-zeus.ini` (8 workers on zeus's 12 cores, nice 19, idle I/O;
~22 s a frame, i.e. about six weeks). It does not restart by itself after a
reboot of zeus: `makak-reprocess.sh start` again, it resumes.

- ECSVs in `/storage/home/mates/phdb-makak` (by night), not in phdb and not in
  the photometric database (`photdb = no`): those are D50 and SBT.
- Queue, logs, nightly summaries, WCS guess and the sky map of the whole
  archive in `/storage/home/mates/makak-reprocess`.
- Darks (shutter closed) are `skipped dark`; imgproc only adds them to the
  night's dark statistics.
- 2023-10..12 has no usable frames (the camera at ~58 C; the legacy code had
  only dark statistics too): those nights end `nosolve`.

Sky map gates (asarina bba8b7e, live on morpheus too): ASTQUAL and PHQUAL OK,
FWHM <= 6, DMAGZERO <= 0.03, Sun below -12 deg, Moon below the horizon. No
MAGZERO limit: the clear sky gives 18.96 in 2021-12 and ~18.6 in 2024-26, and
the weight goes with the zeropoint. The WCS guess is kept as it is (clouds)
only within its own night: after a tilt change the frame is field-solved
(asarina 9f0cafa).

## Deployed on morpheus 2026-10-26

`makak-asarina` + `makak-web` replace `makak-crawl` + `makak-plot` (unit files in
`~/old/systemd`). ~20 s per frame on morpheus. The asarina zeropoint is ~0.29 mag
above the old one (Gaia G versus Sloan r), and the night of 2026-10-26 has the
step at 18:21 CET. The home directory was tidied: everything outside
`asarina+makak bin cat src` is in `~/old`.

## Open

- pyrt-dophot crashes (`add_mask(None)`, exit 33) on frames with very few stars,
  instead of failing cleanly.
- In the arrows plot, the upper-right corner of the square has visible stars but no matches.
