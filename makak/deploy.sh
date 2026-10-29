#!/bin/bash
# deploy.sh code|files|units: the Makak service from this directory (makak/ in
# the asarina repo) to morpheus (ssh mo) and to zeus/hog (ssh z, shared home).
#   code   git pull of main: asarina and pyrt are editable installs (~/src on
#          morpheus, ~/asarina and ~/pyrt in the zeus/hog home).  imgproc runs
#          per frame, so the services take it up with the next frame.
#   files  this directory to ~/asarina+makak (the runtime directory with state,
#          ecsv, nght and the graph, which are left alone)
#   units  the systemd units on morpheus (root), then restarts them
set -euo pipefail
cd "$(dirname "$0")"
FILES="README.md etc bin web systemd data/makak-model.mod"
case "${1:-}" in
code)
	ssh mo 'for r in asarina pyrt; do git -C ~/src/$r pull -q --ff-only origin main; git -C ~/src/$r log --oneline -1; done'
	ssh z 'for r in asarina pyrt; do git -C ~/$r pull -q --ff-only origin main; git -C ~/$r log --oneline -1; done'
	;;
files)
	for h in mo z; do
		rsync -aR $FILES $h:asarina+makak/
	done
	;;
units)
	scp systemd/*.service root@mo:/etc/systemd/system/
	ssh root@mo 'systemctl daemon-reload && systemctl restart makak-asarina makak-web && systemctl --no-pager status makak-asarina makak-web | grep -E "^.*(service|Active)"'
	;;
*)
	sed -n '2,10p' "$0"
	exit 1
	;;
esac
