#!/bin/bash
# makak-renight.sh run|status|swap NIGHT...: reprocess whole nights on morpheus
# through the service config, in daytime only, into $R (default
# ~/asarina+makak/renight), with a fresh WCS guess (the first frame
# field-solves).  For nights processed on a wrong guess, e.g. after the camera
# was moved (2026-11-07 and 11-13, when field-solve was broken).
#   run     frames of the nights not yet done, in time order, 3 at a time at
#           nice 19; stops starting frames outside $FROM..$UNTIL h (local time),
#           so it can run from cron every morning until it is done:
#             30 7 * * * flock -n ~/asarina+makak/renight/lock ~/asarina+makak/bin/makak-renight.sh run 20261107 20261113
#   status  frames done / total per night
#   swap    the new ecsv/, nght/ and png/ of the nights replace the service's;
#           the old ones go to ~/old/renight-YYYYMMDD
# The sky map gets the new frames directly (the gate kept the bad ones out).
set -u
B=/home/mates/asarina+makak
R=${R:-$B/renight}
FROM=${FROM:-7}
UNTIL=${UNTIL:-16}
export PATH=/usr/local/bin:/usr/bin:/bin TERM=xterm
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

frames() {
	for n in "$@"; do ls /images/${n:0:4}/$n/*.fits; done | sort -t/ -k5
}

one() {
	h=$(date +%-H)
	[ "$h" -ge "$FROM" ] && [ "$h" -lt "$UNTIL" ] || exit 0
	f=$1 n=$(basename $1 .fits)
	t0=$(date +%s)
	nice -n 19 ionice -c3 asarina-imgproc --config $B/etc/asarina.conf \
		--wcs-guess $R/state --phdb-root $R/ecsv --daily-summary $R/nght \
		--png-root $R/png --skymap-dir /storage/makak-skymap \
		"$f" > $R/log/$n.log 2>&1
	echo "$n rc=$? $(( $(date +%s) - t0 ))s" >> $R/done
}

cmd=${1:-}; shift || true
[ $# -gt 0 ] || { sed -n '2,19p' "$0"; exit 1; }
mkdir -p $R/state $R/log
touch $R/done
case "$cmd" in
run)
	export -f one; export R B FROM UNTIL
	frames "$@" | grep -vFf <(cut -d' ' -f1 $R/done | sed 's/$/.fits/') |
		xargs -r -P 3 -n 1 bash -c 'one "$0"'
	;;
status)
	for n in "$@"; do
		all=$(frames $n | xargs -n 1 basename | sed 's/\.fits$//')
		echo "$n: $(echo "$all" | grep -cxFf <(cut -d' ' -f1 $R/done)) of $(echo "$all" | wc -l) frames done"
	done
	awk '{print $2}' $R/done | sort | uniq -c
	;;
swap)
	O=/home/mates/old/renight-$(date +%Y%m%d)
	mkdir -p $O/ecsv $O/nght $O/png
	for n in "$@"; do
		[ -d $R/ecsv/$n ] || { echo "$n: nothing new, left alone"; continue; }
		mv $B/ecsv/$n $O/ecsv/ 2>/dev/null; mv $R/ecsv/$n $B/ecsv/
		mv $B/nght/mr$n*.dat $O/nght/ 2>/dev/null
		mv $R/nght/mr$n*.dat $B/nght/
		mv $B/png/$n $O/png/ 2>/dev/null; [ -d $R/png/$n ] && mv $R/png/$n $B/png/
		echo "$n swapped, old in $O"
	done
	;;
*)
	sed -n '2,19p' "$0"; exit 1
	;;
esac
