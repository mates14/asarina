#!/usr/bin/env python3
"""
Combine flat frames with IRAF, handling arbitrary numbers of frames.

IRAF's imcombine is limited to ~150 frames per session. This script:
1. Splits input into chunks of <=150 frames (as few chunks as possible)
2. Combines each chunk in parallel using iraf_combine_flat.sh
3. Combines the partial results into the final output

For statistical soundness in CR removal, we keep chunks as large as possible
so that median rejection works effectively within each chunk. The final
combination of 2-3 pre-cleaned images doesn't need CR rejection.

Usage:
    mixflat.py output.fits input1.fits input2.fits ...
    mixflat.py output.fits *.fits
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

MAX_FRAMES_PER_CHUNK = 150
SCRIPT_DIR = Path(__file__).parent.resolve()
IRAF_SCRIPT = SCRIPT_DIR / "iraf_combine_flat.sh"


def chunk_files(files: list[str], max_per_chunk: int = MAX_FRAMES_PER_CHUNK) -> list[list[str]]:
    """
    Split files into chunks, each with at most max_per_chunk items.
    Distributes files as evenly as possible across the minimum number of chunks.

    For 151 files with max 150: -> [76, 75] (not [150, 1])
    For 300 files with max 150: -> [150, 150]
    For 301 files with max 150: -> [101, 100, 100]
    """
    n = len(files)
    if n <= max_per_chunk:
        return [files]

    num_chunks = (n + max_per_chunk - 1) // max_per_chunk  # ceil division
    base_size = n // num_chunks
    remainder = n % num_chunks

    chunks = []
    start = 0
    for i in range(num_chunks):
        # First 'remainder' chunks get one extra file
        size = base_size + (1 if i < remainder else 0)
        chunks.append(files[start:start + size])
        start += size

    return chunks


def combine_chunk(chunk_files: list[str], output_path: str, chunk_id: int) -> tuple[int, str, bool, str]:
    """
    Combine a single chunk of files using IRAF.
    Returns (chunk_id, output_path, success, error_message).
    """
    try:
        # Convert to absolute paths
        abs_files = [os.path.abspath(f) for f in chunk_files]
        abs_output = os.path.abspath(output_path)

        cmd = [str(IRAF_SCRIPT), abs_output] + abs_files
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per chunk
        )

        if result.returncode != 0:
            return (chunk_id, output_path, False, result.stderr or result.stdout)

        if not os.path.exists(output_path):
            return (chunk_id, output_path, False, "Output file not created")

        return (chunk_id, output_path, True, "")

    except subprocess.TimeoutExpired:
        return (chunk_id, output_path, False, "Timeout after 1 hour")
    except Exception as e:
        return (chunk_id, output_path, False, str(e))


def combine_final(partial_files: list[str], output_path: str) -> bool:
    """
    Combine partial results into final output.
    Since each partial is already CR-cleaned and normalized, we can use
    the same IRAF approach (which will work fine for 2-3 images).
    """
    if len(partial_files) == 1:
        # Just one chunk, copy it
        shutil.copy(partial_files[0], output_path)
        return True

    # Use IRAF to combine the partial results
    # With only 2-3 images, the median is essentially an average but that's fine
    # since CRs are already removed in the partial combines
    abs_files = [os.path.abspath(f) for f in partial_files]
    abs_output = os.path.abspath(output_path)

    cmd = [str(IRAF_SCRIPT), abs_output] + abs_files
    result = subprocess.run(cmd, capture_output=True, text=True)

    return result.returncode == 0 and os.path.exists(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Combine flat frames with IRAF, handling arbitrary numbers of frames."
    )
    parser.add_argument("output", help="Output FITS file")
    parser.add_argument("inputs", nargs="+", help="Input FITS files")
    parser.add_argument(
        "-j", "--jobs", type=int, default=None,
        help="Number of parallel jobs (default: number of CPU cores)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    # Check output doesn't exist
    if os.path.exists(args.output):
        print(f"Error: output file {args.output} already exists", file=sys.stderr)
        return 1

    # Check all inputs exist
    missing = [f for f in args.inputs if not os.path.exists(f)]
    if missing:
        print(f"Error: input files not found: {missing[:5]}{'...' if len(missing) > 5 else ''}", file=sys.stderr)
        return 1

    # Check IRAF script exists
    if not IRAF_SCRIPT.exists():
        print(f"Error: IRAF script not found: {IRAF_SCRIPT}", file=sys.stderr)
        return 1

    n_files = len(args.inputs)
    chunks = chunk_files(args.inputs, MAX_FRAMES_PER_CHUNK)
    n_chunks = len(chunks)

    if args.verbose or n_chunks > 1:
        print(f"Combining {n_files} frames in {n_chunks} chunk(s): {[len(c) for c in chunks]}")

    if n_chunks == 1:
        # Simple case: just run IRAF directly
        success, _, ok, err = combine_chunk(chunks[0], args.output, 0)
        if not ok:
            print(f"Error: {err}", file=sys.stderr)
            return 1
        if args.verbose:
            print(f"Created {args.output}")
        return 0

    # Multiple chunks: process in parallel, then combine
    # Use ~/tmp for space (system /tmp and /var/tmp may be too small)
    user_tmp = Path.home() / "tmp"
    user_tmp.mkdir(exist_ok=True)
    tempdir = tempfile.mkdtemp(prefix="mixflat_", dir=user_tmp)
    try:
        partial_outputs = []

        # Run chunks in parallel (limit to 4 by default - IRAF is I/O bound)
        n_jobs = args.jobs or min(n_chunks, 4)
        if args.verbose:
            print(f"Running {n_chunks} chunks with {n_jobs} parallel jobs")

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {}
            for i, chunk in enumerate(chunks):
                partial_path = os.path.join(tempdir, f"partial_{i}.fits")
                partial_outputs.append(partial_path)
                future = executor.submit(combine_chunk, chunk, partial_path, i)
                futures[future] = i

            # Wait for all chunks to complete
            for future in as_completed(futures):
                chunk_id, output_path, success, error = future.result()
                if args.verbose:
                    status = "done" if success else f"FAILED: {error}"
                    print(f"  Chunk {chunk_id + 1}/{n_chunks}: {status}")
                if not success:
                    print(f"Error in chunk {chunk_id}: {error}", file=sys.stderr)
                    return 1

        # Combine partial results
        if args.verbose:
            print(f"Combining {n_chunks} partial results...")

        if not combine_final(partial_outputs, args.output):
            print("Error: failed to combine partial results", file=sys.stderr)
            return 1

        if args.verbose:
            print(f"Created {args.output}")

        return 0

    finally:
        # Clean up temp directory
        shutil.rmtree(tempdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
