#!/bin/bash
# bench.sh NAME EXTRA-IMGPROC-ARGS... : run the burst sample through imgproc, report time and quality
name=$1; shift
out=/home/mates/asarina+makak/bench/$name
rm -rf $out; mkdir -p $out
frames=$(ls testdata/images/2026/20261024/2026102422{00,01,02}*.fits | head -8)
for f in $frames; do
  t0=$(date +%s.%N)
  asarina-imgproc --config etc/test.conf --no-raw-update --phdb-root $out/ecsv --daily-summary $out/nght --png-root $out/png "$@" $f 2>$out/$(basename $f .fits).log
  rc=$?
  echo "$(basename $f) rc=$rc $(echo "$(date +%s.%N) - $t0" | bc | cut -c1-5)s"
done | tee $out/times
cat $out/nght/mr*.dat 2>/dev/null | grep -v "d.dat" | awk '{print $1, "id="$7, "zp="$8, "zerr="$9, "lim="$10, "wssr="$11}'
