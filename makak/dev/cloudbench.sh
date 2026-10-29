#!/bin/bash
# cloudbench.sh NAME EXTRA-IMGPROC-ARGS... : the 2026-10-26 moonlit-cloud sample in sequence
# (one WCS-guess chain, like the service), then a per-frame summary
name=$1; shift
out=/home/mates/asarina+makak/bench/$name
rm -rf $out; mkdir -p $out/state
cp state/mi0315.wcs $out/state/
export TERM=xterm OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for f in testdata/20261026/*-RA.fits; do
  t0=$(date +%s.%N)
  asarina-imgproc --config etc/asarina.conf --wcs-guess $out/state --phdb-root $out/ecsv \
    --daily-summary $out/nght --png-root $out/png "$@" $f 2>$out/$(basename $f .fits).log
  echo "$(basename $f .fits) rc=$? $(echo "$(date +%s.%N) - $t0" | bc | cut -c1-5)s"
done > $out/times
