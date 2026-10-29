#!/bin/bash
# makak-hips.sh: on hog, the Makak HiPS from morpheus's sky map tiles
# (/storage/makak-skymap, written by the service), next to SBT's in
# /storage2/reprocess: makak/{skymap,export,hips}, with hog's asarina
# (asarina-skymap --profile makak).
set -e
R=/storage2/reprocess/makak
mkdir -p $R
# tiles are replaced whole (os.replace), so a copy is never half a tile.
# The key is limited on morpheus to command="rrsync -ro /storage/makak-skymap",
# so the source path is relative to the map; no agent, so it works from cron.
rsync -a --delete --exclude '*.lock' --exclude '*.tmp' \
	-e 'ssh -i /home/mates/.ssh/hog-morpheus-rsync -o IdentitiesOnly=yes -o IdentityAgent=none -o BatchMode=yes -o ForwardX11=no' \
	morpheus.asu.cas.cz: $R/skymap/
exec nice /usr/local/bin/asarina-skymap --profile makak hips \
	--dir $R/skymap --export $R/export --hips $R/hips \
	--jar /storage2/reprocess/Hipsgen.jar --memory 16g --threads 8
