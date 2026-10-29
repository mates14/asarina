#!/bin/bash
# sim.sh N CADENCE : feed N burst frames into sim/images/2026/20261024 like RTS2 does
src=/home/mates/asarina+makak/testdata/images/2026/20261024
dst=/home/mates/asarina+makak/sim/images/2026/20261024
mkdir -p $dst
for f in $(ls $src/2026102422*.fits | head -n $1); do
  cp $f $dst/.tmp.fits && mv $dst/.tmp.fits $dst/$(basename $f)
  echo "$(date +%T) arrived $(basename $f)"
  sleep $2
done
