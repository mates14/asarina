#!/bin/bash
# Simple IRAF wrapper to combine dark frames.
# This script handles at most 150 frames.
# For larger sets, use mixdark.py which calls this script in parallel.
#
# Usage: iraf_combine_dark.sh output.fits input1.fits input2.fits ...

set -e

if [ -z "$1" ]; then
    echo "Usage: iraf_combine_dark.sh output.fits input1.fits input2.fits ..."
    exit 1
fi

if [ -e "$1" ]; then
    echo "Error: output file $1 already exists"
    exit 1
fi

result=$1
shift

N=$#
if [ $N -gt 150 ]; then
    echo "Error: this script handles at most 150 frames (got $N)"
    echo "Use mixdark.py for larger sets"
    exit 1
fi

if [ $N -eq 0 ]; then
    echo "Error: no input files provided"
    exit 1
fi

# Create temp directory (use ~/tmp for space)
user_tmp="$HOME/tmp"
mkdir -p "$user_tmp"
tempdir=$(mktemp -d "$user_tmp/iraf_dark.XXXXXX")
trap "rm -rf $tempdir" EXIT

# Stage inputs into the temp dir as plain single-HDU FITS. Archive frames may be
# Rice tile-compressed (.fitz: empty primary HDU + image in extension 1), which
# IRAF cannot read. cfitsio's imcopy selects the image HDU and writes it as the
# output primary; a primary HDU cannot be tile-compressed, so this decompresses.
# .fitz keep the image in extension [1], plain .fits in the primary [0].
# Sequential names avoid collisions between same-named frames from other nights.
> "$tempdir/files.lst"
i=0
for f in "$@"; do
    out=$(printf "in_%05d.fits" "$i")
    case "$f" in
        *.fitz) sec="[1]" ;;
        *)      sec="[0]" ;;
    esac
    imcopy "$f$sec" "$tempdir/$out"
    echo "$out" >> "$tempdir/files.lst"
    i=$((i+1))
done
cd "$tempdir"

# Isolate IRAF state to avoid conflicts when running in parallel
export HOME="$tempdir"

# Find IRAF command (cl or irafcl depending on installation)
if command -v cl &>/dev/null; then
    IRAF_CL=cl
elif command -v irafcl &>/dev/null; then
    IRAF_CL=irafcl
else
    echo "Error: IRAF not found (tried 'cl' and 'irafcl')" >&2
    exit 1
fi

# files.lst was written during staging above (in_NNNNN.fits, one per line)

# Calculate rejection parameters
nrej=$((N / 5))

# Run IRAF (with timeout to prevent zombie ecl.e processes)
echo 'images
imsurfit @files.lst @files.lst//-f 1 3 xmedian=64 ymedian=64 type_ou=fit
imcombine @files.lst//-f ave.fits combine=average
imarith @files.lst - @files.lst//-f @files.lst//-r
imcombine @files.lst//-r res.fits combine=average reject=minmax nlow='$nrej' nhigh='$nrej'
imarith ave.fits + res.fits result.fits
logout
' | timeout -s KILL 1200 $IRAF_CL

# Kill any orphaned IRAF processes from this script
pkill -9 -P $$ ecl.e 2>/dev/null || true

# Set header and copy result
fitsheader -w IMAGETYP=mdark result.fits

# Handle both relative and absolute output paths
if [[ "$result" = /* ]]; then
    cp result.fits "$result"
else
    cp result.fits "$OLDPWD/$result"
fi
