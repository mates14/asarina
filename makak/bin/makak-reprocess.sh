#!/bin/bash
# makak-reprocess.sh [run|report|list|retry ...]: on zeus, asarina-reprocess of
# the whole Makak archive (etc/reprocess-zeus.ini), with zeus's asarina (main,
# editable in ~/asarina).  "start" runs the service detached, at idle I/O
# priority, its log appended to /storage/home/mates/makak-reprocess/run.log;
# "stop" stops it (the frames being processed go back to the queue).
export PATH=/usr/local/bin:/usr/bin:/bin TERM=xterm
SETUP=/home/mates/asarina+makak/etc/reprocess-zeus.ini
R=/storage/home/mates/makak-reprocess
# the service's command line (no shell that merely mentions it)
PAT="^[^ ]*python3 [^ ]*asarina-reprocess --setup ${SETUP//+/\\+} run"
case "$1" in
start)
	pgrep -f "$PAT" > /dev/null && { echo "already running"; exit 1; }
	mkdir -p $R
	setsid nohup ionice -c3 asarina-reprocess --setup $SETUP run >> $R/run.log 2>&1 < /dev/null &
	;;
stop)
	pkill -TERM -f "$PAT" || exit 0
	while pgrep -f "$PAT" > /dev/null; do sleep 2; done
	;;
*)
	exec asarina-reprocess --setup $SETUP "$@"
	;;
esac
