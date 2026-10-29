#!/bin/bash
# makak-rep-one.sh FRAME: reprocess a frame on zeus through the Makak service
# config, outputs under $R (default ~/makak-rep); run over a night with
#   R=... xargs -P 20 -n 1 -a LIST makak-rep-one.sh >> $R/times
R=${R:-/home/mates/makak-rep}
export PATH=/usr/local/bin:/usr/bin:/bin TERM=xterm OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
t0=$(date +%s.%N)
asarina-imgproc --config /home/mates/asarina+makak/etc/asarina.conf --wcs-guess $R/state \
  --phdb-root $R/ecsv --daily-summary $R/nght --png-root $R/png --skymap-dir $R/map \
  "$1" > $R/log/$(basename $1 .fits).log 2>&1
echo "$(basename $1 .fits) rc=$? $(echo "$(date +%s.%N) - $t0" | bc)"
