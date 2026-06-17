#!/usr/bin/env python3
"""Reorganize a messy RTS2 image archive into a clean camera-rooted tree.

    {dest}/{camera_id}/{year}/{night}/{darks|flats|NNNNN}/{original_name}
    e.g. .../B2-clean/andor3567/2010/20100922/darks/20100922....fitz

Everything is classified from the FITS header, never the (unreliable) directory
names:
    camera_id  <- chip_id registry (CCD_SER / CCD_TYPE)
    night      <- (DATE-OBS - 12h).date()   (noon boundary; validated vs archive)
    subdir     <- TARGET: 1 -> darks, 2 -> flats, else zero-padded target id
                  (falls back to IMAGETYP dark/flat if TARGET is missing)

Frames that cannot be classified (unknown camera, missing DATE-OBS/TARGET,
unreadable header) are routed to {dest}/unknown/<original-relative-path> so
nothing is lost and provenance is preserved.

Safety:
  * Dry-run by default. Nothing moves unless --execute is given.
  * Moves are os.link()+os.unlink() on the same filesystem: metadata-only (no
    extra space) and race-free no-overwrite (link fails if the dest exists).
  * Cross-filesystem destinations are refused (would be a slow copy, not a move).
  * Only .fits / .fitz images are touched; all other files are left in place.
  * Every action is written to a manifest TSV (src, dest, status, detail).

Usage:
  reorganize_archive.py --source /storage/user-data/mates/B2            # dry-run
  reorganize_archive.py --source .../B2 --dest .../B2-clean --execute   # move
  reorganize_archive.py --source .../B2 --limit 5000 -j 16              # sample
"""

import os
import sys
import argparse
import datetime
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from astropy.io import fits
    from astropy.utils.exceptions import AstropyWarning
except ImportError:
    print("Error: astropy is required (pip install astropy)", file=sys.stderr)
    sys.exit(1)

# The experimental-era BOOTES frames trip many FITS verify warnings (non-2880
# header sizes, null padding, stray cards). astropy still reads them; anything
# truly unreadable is caught per-file and logged as error/unknown. Silence the
# flood so progress and the summary stay legible.
warnings.simplefilter("ignore", AstropyWarning)

from asarina.chip_id import get_camera_id

IMAGE_EXTS = (".fits", ".fitz")


def _image_header(hdul):
    """Header of the HDU that holds the image (handles Rice-compressed .fitz)."""
    hdu = hdul[0]
    if hdu.data is None and len(hdul) > 1:
        hdu = hdul[1]
    return hdu.header


def classify(path: str) -> dict:
    """Read one frame's header and decide where it belongs.

    Returns dict with camera/year/night/sub (None if undetermined), a status
    ('ok' | 'unknown' | 'error'), and a detail string. Pure read-only.
    """
    try:
        with fits.open(path) as hdul:
            hd = _image_header(hdul)
            camera = get_camera_id(hd)
            dobs = hd.get("DATE-OBS")
            target = hd.get("TARGET")
            imagetyp = str(hd.get("IMAGETYP") or "").strip().lower()
    except Exception as e:
        return {"src": path, "status": "error", "detail": str(e)[:100],
                "camera": None, "year": None, "night": None, "sub": None}

    # night from DATE-OBS, noon boundary
    night = None
    if dobs:
        try:
            t = datetime.datetime.fromisoformat(str(dobs).strip())
            night = (t - datetime.timedelta(hours=12)).strftime("%Y%m%d")
        except Exception:
            night = None

    # subdir from RTS2 TARGET (1=dark, 2=flat, else science target id)
    sub = None
    if target is not None:
        try:
            ti = int(target)
            sub = "darks" if ti == 1 else "flats" if ti == 2 else f"{ti:05d}"
        except (ValueError, TypeError):
            sub = None
    if sub is None:                       # fallback to IMAGETYP for calib
        if imagetyp == "dark":
            sub = "darks"
        elif imagetyp == "flat":
            sub = "flats"

    if camera and camera != "unknown" and night and sub:
        return {"src": path, "status": "ok", "detail": "",
                "camera": camera, "year": night[:4], "night": night, "sub": sub}

    missing = []
    if not camera or camera == "unknown":
        missing.append("camera")
    if not night:
        missing.append("date-obs")
    if not sub:
        missing.append("target")
    return {"src": path, "status": "unknown", "detail": ",".join(missing),
            "camera": camera, "year": night[:4] if night else None,
            "night": night, "sub": sub}


def dest_for(rec: dict, source_root: str, dest_root: str) -> str:
    """Absolute destination path for a classification record."""
    base = os.path.basename(rec["src"])
    if rec["status"] == "ok":
        return os.path.join(dest_root, rec["camera"], rec["year"],
                            rec["night"], rec["sub"], base)
    # unknown / error -> quarantine, preserving the original relative path
    rel = os.path.relpath(rec["src"], source_root)
    return os.path.join(dest_root, "unknown", rel)


def gather_images(source_root: str, dest_root: str, limit: int = 0) -> list:
    """Walk source_root collecting .fits/.fitz files (skipping dest_root)."""
    dest_abs = os.path.abspath(dest_root)
    files = []
    for dirpath, dirnames, filenames in os.walk(source_root):
        if os.path.abspath(dirpath).startswith(dest_abs):
            dirnames[:] = []
            continue
        for fn in filenames:
            if fn.lower().endswith(IMAGE_EXTS):
                files.append(os.path.join(dirpath, fn))
                if limit and len(files) >= limit:
                    return files
    return files


def same_filesystem(a: str, b: str) -> bool:
    """True if paths a and b are on the same filesystem (device)."""
    def dev(p):
        while not os.path.exists(p):
            p = os.path.dirname(p) or "/"
        return os.stat(p).st_dev
    return dev(a) == dev(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", "-s", required=True, help="Archive root to reorganize")
    ap.add_argument("--dest", "-d", default=None,
                    help="Clean-tree root (default: <source>-clean)")
    ap.add_argument("--execute", action="store_true",
                    help="Actually move files (default: dry-run, move nothing)")
    ap.add_argument("--workers", "-j", type=int, default=min(os.cpu_count() or 1, 16))
    ap.add_argument("--limit", type=int, default=0, help="Process at most N files (testing)")
    ap.add_argument("--manifest", default=None, help="Manifest TSV path (default: ./reorg-manifest.tsv)")
    args = ap.parse_args()

    source = os.path.abspath(args.source.rstrip("/"))
    dest = os.path.abspath(args.dest) if args.dest else source + "-clean"
    manifest = args.manifest or "reorg-manifest.tsv"

    if not os.path.isdir(source):
        print(f"Error: source not found: {source}", file=sys.stderr)
        sys.exit(1)
    if args.execute and not same_filesystem(source, dest):
        print(f"Error: dest {dest} is NOT on the same filesystem as {source}.\n"
              f"A move would be a slow cross-device copy. Aborting.", file=sys.stderr)
        sys.exit(1)

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"[{mode}] source={source}")
    print(f"[{mode}] dest  ={dest}")
    print(f"[{mode}] scanning for images...")
    files = gather_images(source, dest, args.limit)
    print(f"[{mode}] {len(files)} image files found; classifying with {args.workers} workers")

    # Classify in parallel
    recs = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for rec in ex.map(classify, files, chunksize=64):
            recs.append(rec)
            done += 1
            if done % 20000 == 0:
                print(f"  classified {done}/{len(files)}", flush=True)

    # Build destinations and detect collisions (two sources -> one dest)
    dest_map = {}
    collisions = 0
    from collections import Counter
    by_status = Counter()
    by_camera = Counter()
    unknown_reason = Counter()
    for r in recs:
        r["dest"] = dest_for(r, source, dest)
        by_status[r["status"]] += 1
        by_camera[r["camera"] or "unknown"] += 1
        if r["status"] != "ok":
            unknown_reason[r["detail"] or "?"] += 1
        dest_map.setdefault(r["dest"], []).append(r["src"])
    for d, srcs in dest_map.items():
        if len(srcs) > 1:
            collisions += len(srcs) - 1

    # Report
    print("\n===== summary =====")
    print(f"  total images     : {len(recs)}")
    for st, n in by_status.most_common():
        print(f"  status {st:8s}  : {n}")
    print(f"  dest collisions  : {collisions} (same target path; extras would be skipped)")
    print("  cameras          : " + ", ".join(f"{c}={n}" for c, n in by_camera.most_common()))
    if unknown_reason:
        print("  unknown reasons  : " + ", ".join(f"{r}={n}" for r, n in unknown_reason.most_common()))
    print("  sample moves:")
    for r in recs[:8]:
        print(f"    {r['src']}\n      -> {r['dest']}  [{r['status']}]")

    # Write manifest
    with open(manifest, "w") as mf:
        mf.write("src\tdest\tstatus\tdetail\n")
        for r in recs:
            mf.write(f"{r['src']}\t{r['dest']}\t{r['status']}\t{r['detail']}\n")
    print(f"\n  manifest written: {manifest}")

    if not args.execute:
        print(f"\n[DRY-RUN] no files moved. Review {manifest}, then re-run with --execute.")
        return

    # Execute: link+unlink (same-fs, race-free no-overwrite)
    print(f"\n[EXECUTE] moving {len(recs)} files...")
    moved = skipped = errors = 0
    seen_dest = set()
    log = open(manifest + ".done", "w")
    log.write("src\tdest\taction\tdetail\n")
    for i, r in enumerate(recs):
        src, dst = r["src"], r["dest"]
        if dst in seen_dest:           # in-run collision: first wins
            skipped += 1; log.write(f"{src}\t{dst}\tskip_collision\t\n"); continue
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.link(src, dst)          # fails if dst exists -> no overwrite
            os.unlink(src)
            seen_dest.add(dst)
            moved += 1
            log.write(f"{src}\t{dst}\tmoved\t\n")
        except FileExistsError:
            skipped += 1
            log.write(f"{src}\t{dst}\tskip_exists\t\n")
        except OSError as e:
            errors += 1
            log.write(f"{src}\t{dst}\terror\t{str(e)[:100]}\n")
        if (i + 1) % 20000 == 0:
            print(f"  moved {i+1}/{len(recs)} (moved={moved} skip={skipped} err={errors})", flush=True)
    log.close()
    print(f"\n[EXECUTE] done: moved={moved} skipped={skipped} errors={errors}")
    print(f"  action log: {manifest}.done")
    print("  (source tree still holds non-image files and now-empty dirs)")


if __name__ == "__main__":
    main()
