# Makak pipeline in asarina

Makak is a zenith all-sky camera at Ondřejov (camera `mi0315`, Moravian G1-2000,
serial 00315, Sony ICX274, ~55 arcsec/px).  Its processing was previously handled
by the standalone `makak-reloaded` project.  It has been folded into asarina
because the pipeline is identical in structure — calibrate → solve → dophot →
save ECSV — with a handful of camera-specific parameters.

## What is different about Makak

**Dark correction** — Makak uses a per-pixel analytical dark model instead of
master dark frames.  The model file (`makak-dark-response.npy`) encodes
coefficients A, B, C per pixel such that `dark(T) = A·B^T + C`.  The optimal
temperature is found numerically by minimising residual image noise, compensating
for thermistor offset and non-uniformity.  Pass `--smart-dark CALIB.npy` to
activate this path; it bypasses the dark+flat database entirely (no flat either).

**No flat field** — the smart-dark path skips flat correction.  If you ever need
flats for Makak, drop `--smart-dark` and let the normal `ImageProcessor` path
run.

**Pixel scale hint** — Makak headers do not carry a pixel scale, so the
astrometric solver needs a hint.  `--makak` implies 55 arcsec/px; override with
`--pixel-scale N` if the optics change.

**Dark frame detection** — in the normal RTS2 workflow for cooled cameras, dark
frames never reach the pipeline.  Makak sends them through the same FITS stream,
distinguished by `slitposx < 0.5` (slit closed = zenith camera shutter blocking
the sky).  `--makak` enables this detection: dark frames are not processed through
solve+dophot but their corrected noise (`darksig`) and fitted temperature are
written to `mr{YYYYMMDD}d.dat` in `--daily-summary`.

**Dophot model and catalog** — Makak uses a custom photometric model file and a
dedicated reference catalog.  The `-k`/`--makak` flag inside pyrt-dophot enables
Makak-specific tweaks; `--makak` on the asarina side passes `-k` automatically.

**Output directory layout** — standard asarina uses `phdb/{YYMM}/`, Makak uses
`ecsv/{YYYYMMDD}/` (day granularity, full year).  Controlled by `--phdb-root` and
`--phdb-date-fmt`.

**Nightly summary** — `--daily-summary DIR` appends the last line of pyrt-dophot's
`dophot.dat` to `mr{YYYYMMDD}.dat` after each successfully processed frame.  This
is the photometry quality statistic for the night.  Dark frame statistics go to the
paired `mr{YYYYMMDD}d.dat`.  Both use a noon-to-noon convention so that post-midnight
observations belong to the same night entry.

## Batch reprocessing (replaces `do_reprocess.sh`)

```bash
asarina-photometry \
  --smart-dark /home/mates/makak-reloaded/makak-dark-response.npy \
  --makak \
  --phdb-root   /home/mates/makak-reloaded/ecsv \
  --phdb-date-fmt '%Y%m%d' \
  --daily-summary /home/mates/makak-reloaded/nght \
  --dophot-model   /home/mates/makak-reloaded/model.mod \
  --dophot-catalog makak \
  --dophot-maglim  8 \
  --dophot-enlarge 1.5 \
  --dophot-terms   '.p4,.r4,RC,RO,RS' \
  --dophot-idlimit 5 \
  --dophot-max-stars 0 \
  /storage/archive-images/MAKAK/images/2025/20250129/*.fits
```

For a full archive sweep:

```bash
find /storage/archive-images/MAKAK/images -name '*.fits' -print0 \
  | sort -z \
  | xargs -0 asarina-photometry \
      --smart-dark /home/mates/makak-reloaded/makak-dark-response.npy \
      --makak \
      --phdb-root   /home/mates/makak-reloaded/ecsv \
      --phdb-date-fmt '%Y%m%d' \
      --daily-summary /home/mates/makak-reloaded/nght \
      --dophot-model   /home/mates/makak-reloaded/model.mod \
      --dophot-catalog makak \
      --dophot-maglim  8 \
      --dophot-enlarge 1.5 \
      --dophot-terms   '.p4,.r4,RC,RO,RS' \
      --dophot-idlimit 5 \
      --dophot-max-stars 0
```

Already-processed files are skipped automatically (unless `-f`/`--force`).

## Real-time daemon (replaces `makak-reloaded` live mode)

```bash
asarina-pipeline \
  --search-root /storage/archive-images/MAKAK/images \
  --camera-pattern '' \
  --smart-dark /home/mates/makak-reloaded/makak-dark-response.npy \
  --makak \
  --phdb-root   /home/mates/makak-reloaded/ecsv \
  --phdb-date-fmt '%Y%m%d' \
  --daily-summary /home/mates/makak-reloaded/nght \
  --dophot-model   /home/mates/makak-reloaded/model.mod \
  --dophot-catalog makak \
  --dophot-maglim  8 \
  --dophot-enlarge 1.5 \
  --dophot-terms   '.p4,.r4,RC,RO,RS' \
  --dophot-idlimit 5 \
  --dophot-max-stars 0 \
  --max-workers 4 \
  --ssh-key ~/.ssh/id_makak
```

## Flag reference

| Flag | Default | Purpose |
|---|---|---|
| `--smart-dark CALIB.npy` | off | Pixel dark model; bypasses dark+flat database |
| `--makak` | off | Bundles: dark detection, 55"/px hint, `-k` in dophot, mi0315 crop |
| `--pixel-scale N` | none (55 with `--makak`) | Override arcsec/px hint for pyrt-field-solve |
| `--phdb-root DIR` | `~/phdb` | Root for ECSV output |
| `--phdb-date-fmt FMT` | `%y%m` | Subdirectory date format (`%Y%m%d` for day-level) |
| `--daily-summary DIR` | none | Write nightly summary lines to `mr{YYYYMMDD}.dat` |
| `--dophot-model FILE` | none | Model file (`-M`) for pyrt-dophot |
| `--dophot-catalog NAME` | none | Reference catalog (`-C`) |
| `--dophot-maglim N` | none | Magnitude limit (`-l`) |
| `--dophot-enlarge N` | none | Enlarge factor (`-e`) |
| `--dophot-terms TERMS` | `.r3,.p3,.l` | Uncertainty terms (`-U`) |
| `--dophot-idlimit N` | 2 | ID-limit iterations (`-i`) |
| `--dophot-max-stars N` | 1000 | Max stars (0 = no limit) |

## Notes on future use

The dophot options (`--dophot-model`, `--dophot-catalog`, `--daily-summary`, etc.)
are not Makak-specific and work for any camera.  In particular, `--daily-summary`
could be useful for D50 and SBT to get per-night photometry statistics without
any additional tooling.

The `--smart-dark` path is also camera-agnostic.  It was developed for Makak
because the dark current of the uncooled Sony IMX sensor is strongly
temperature-dependent and varies significantly across the chip.  It did not prove
useful for the cooled Kodak chips in SBT/D50 where the standard master-dark
approach is adequate.  The infrastructure is there if that changes.
