#!/usr/bin/env python3
"""
kinseries - real-time runner of the progressive kinetic-series light curve (kinstep).

asarina-imgproc, having saved the ECSV of a kinetic series (ACQMODE 3, KINNUM >= 12,
KINFILE), drops a job into <dir>/spool/ and starts this runner (as the pipeline user,
detached, in its own session).  Only one runner works at a time (flock on <dir>/lock);
a second one exits at once.  With nothing to do for --idle seconds the runner exits;
the next kinetic series starts it again.

The runner groups the series by observation (OBSID, filter): with n series of an
observation in, it runs the newest "trivial" step k <= n not done yet (k = 1, 2, 3, 4, 6,
12, see kinstep.py) - a step that has gone stale while an earlier one was running is
skipped, the newest data are what matters.  After a step, the flare search
(pyrt-transients --flare-rec) runs on its bins in the background.

Layout of <dir> (config key kinseries_dir, e.g. /home/mates/kinseries):
  spool/<stem>.json                 jobs from imgproc
  obs/<YYYYMMDD>-<obsid>-<filter>/  one per observation:
      series/<stem>.json            its series (the jobs, moved here)
      state.json                    steps done / failed / reported
      step<kk>/                     kinstep output: per-bin ECSVs, deep ECSV, step.json,
                                    kinstep.log, transients.log, flares.txt
  kinseries.log                     the runner log
  lock

Usage:
  asarina-kinseries --dir /home/mates/kinseries          # as imgproc starts it
  asarina-kinseries --dir /home/mates/kinseries --once   # one pass, then exit (testing)
"""

import argparse
import fcntl
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import time

log = logging.getLogger("kinseries")

TRIVIAL = (1, 2, 3, 4, 6, 12)
POINTS = 12
RAW_WAIT = 120        # [s] how long a series may wait for its raw frame to be archived
OBS_ACTIVE = 4 * 3600  # [s] observations untouched for longer are not looked at


# ---------------------------------------------------------------- jobs

def queue_job(root, hdr, raw_path, ecsv_path):
    """Called by imgproc: drop a job for a kinetic series, start the runner.
    Returns the job path, or None when the frame is not a kinetic series."""
    acq = str(hdr.get('ACQMODE', '')).strip().lower()
    if acq not in ('3', 'kinetic series', 'kinetics'):
        return None
    kinnum = int(hdr.get('KINNUM') or 0)
    if kinnum < POINTS or not hdr.get('KINFILE') or not hdr.get('KINCYCLE'):
        return None
    stem = os.path.basename(str(raw_path))[:18]
    job = {
        'stem': stem, 'raw': str(raw_path), 'ecsv': str(ecsv_path), 'kinfile': hdr['KINFILE'],
        'obsid': int(hdr.get('OBSID') or 0), 'target': hdr.get('TARGET'),
        'object': str(hdr.get('OBJECT', '')), 'filter': str(hdr.get('FILTER', 'UNK')),
        'kinnum': kinnum, 'kincycle': float(hdr['KINCYCLE']),
        'ctime': float(hdr.get('CTIME', 0)) + float(hdr.get('USEC', 0)) / 1e6,
        'ccd': hdr.get('CCD_NAME'), 'queued': time.time(),
    }
    spool = os.path.join(root, 'spool')
    os.makedirs(spool, exist_ok=True)
    tmp = os.path.join(spool, f".{stem}.tmp")
    with open(tmp, 'w') as f:
        json.dump(job, f)
    path = os.path.join(spool, f"{stem}.json")
    os.replace(tmp, path)   # atomic: the runner never sees half a job
    start_runner(root)
    return path


def start_runner(root):
    """Start a runner detached from the caller (own session: imgproc's timeout or exit
    does not take it down).  It exits at once if another one holds the lock."""
    import pwd
    pw = pwd.getpwuid(os.getuid())
    env = dict(os.environ, HOME=pw.pw_dir, USER=pw.pw_name, LOGNAME=pw.pw_name)
    env.setdefault('TERM', 'xterm')
    subprocess.Popen([sys.executable, '-m', 'asarina.pipeline.kinseries', '--dir', root],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     cwd=root, env=env, start_new_session=True, close_fds=True)


# ---------------------------------------------------------------- state

def _load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _save(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def ingest(root):
    """Move the spooled jobs into their observation directories."""
    n = 0
    for path in sorted(glob.glob(os.path.join(root, 'spool', '*.json'))):
        job = _load(path, None)
        if job is None:
            log.warning(f"unreadable job {path}, dropped")
            os.rename(path, path + '.bad')
            continue
        key = f"{job['obsid']}-{job['filter']}"
        hits = glob.glob(os.path.join(root, 'obs', f"*-{key}"))
        obs = hits[0] if hits else os.path.join(root, 'obs', f"{job['stem'][:8]}-{key}")
        os.makedirs(os.path.join(obs, 'series'), exist_ok=True)
        os.replace(path, os.path.join(obs, 'series', f"{job['stem']}.json"))
        log.info(f"{os.path.basename(obs)}: series {job['stem']} ({job['object']}, "
                 f"{job['kinnum']} x {job['kincycle']:.3f} s)")
        n += 1
    return n


def raw_ready(job):
    """The raw frame (archived by RTS2 after imgproc's corrwerr) and the KINFILE exist."""
    from asarina.pipeline.kinphot import find_raw
    if job['raw'].endswith('-kin.fits'):
        return os.path.exists(job['raw'])   # a cube (offline tests)
    if not (os.path.exists(job['raw']) or find_raw(job['stem'])):
        return False
    kin = job['kinfile']
    return os.path.exists(kin) or os.path.exists(kin.replace('/var/tmp', '/images/kinmode'))


# ---------------------------------------------------------------- steps

def due_step(nseries, state):
    done = state.get('done', []) + state.get('failed', [])
    avail = [k for k in TRIVIAL if k <= nseries]
    if not avail:
        return None
    k = avail[-1]
    if k in done or k <= max(done, default=0):
        return None
    return k


def run_step(obs, series, k, args):
    """kinstep on the first k series of the observation; True if it produced bins."""
    out = os.path.join(obs, f"step{k:02d}")
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    nsub = min(j['kinnum'] for j in series[:k]) // POINTS * POINTS
    cmd = [sys.executable, '-m', 'asarina.pipeline.kinstep', '--step', str(k), '--nsub', str(nsub),
           '-j', str(args.jobs), '-o', out] + args.kinstep_args.split()
    for j in series[:k]:
        cmd += ['--ref', j['ecsv']]
    cmd += [j['raw'] for j in series[:k]]
    log.info(f"{os.path.basename(obs)}: step {k} ({k} series, bins of {nsub * k // POINTS})")
    t = time.time()
    with open(os.path.join(out, 'kinstep.log'), 'w') as lf:
        try:
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=args.timeout, cwd=out)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = 'timeout'
    res = _load(os.path.join(out, 'step.json'), {})
    if not res.get('bins_ok'):
        log.error(f"{os.path.basename(obs)}: step {k} failed (rc {rc}, {time.time() - t:.0f}s), "
                  f"see {out}/kinstep.log")
        return False
    log.info(f"{os.path.basename(obs)}: step {k} {res['bins_ok']}/{res['bins']} bins of {res['bin_seconds']} s, "
             f"deep limit {res['deep_limit']:.2f} ({time.time() - t:.0f}s, rc {rc})")
    flare_search(out)
    return True


def flare_search(out):
    """pyrt-transients --flare-rec on the bins, in the background; it leaves flares.txt
    (the FLARES section of its output) when done."""
    bins = sorted(glob.glob(os.path.join(out, '*-k[0-9]*b*.ecsv')))
    if not bins:
        return
    script = ('pyrt-transients --flare-rec rec.ecsv "$@" > transients.log 2>&1; '
              'sed -n "/FLARES START/,/FLARES END/p" transients.log > flares.tmp; mv flares.tmp flares.txt')
    subprocess.Popen(['sh', '-c', script, 'flares'] + [os.path.basename(b) for b in bins],
                     cwd=out, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)


def report_flares(obs, state):
    """Log the flare-search results that appeared since the last pass."""
    rep = state.setdefault('reported', [])
    new = False
    for k in state.get('done', []):
        f = os.path.join(obs, f"step{k:02d}", 'flares.txt')
        if k in rep or not os.path.exists(f):
            continue
        with open(f) as fh:
            lines = [ln.rstrip() for ln in fh if ln.strip() and 'FLARES' not in ln]
        cand = [ln for ln in lines if 'FLARE' in ln]
        log.info(f"{os.path.basename(obs)}: step {k} flare search: "
                 + (f"{len(cand)} candidate(s)" if cand else "nothing"))
        for ln in cand[:5]:
            log.info(f"    {ln}")
        rep.append(k)
        new = True
    return new


def serve(root, args):
    """One pass over the spool and the active observations; True if anything was done."""
    did = ingest(root) > 0
    now = time.time()
    for obs in sorted(glob.glob(os.path.join(root, 'obs', '*'))):
        if now - os.path.getmtime(os.path.join(obs, 'series')) > OBS_ACTIVE:
            continue
        spath = os.path.join(obs, 'state.json')
        state = _load(spath, {'done': [], 'failed': []})
        if report_flares(obs, state):
            _save(spath, state)
        series = sorted((_load(p, None) for p in glob.glob(os.path.join(obs, 'series', '*.json'))),
                        key=lambda j: j['ctime'])
        k = due_step(len(series), state)
        if k is None:
            continue
        waiting = [j['stem'] for j in series[:k] if not raw_ready(j)]
        if waiting and now - max(j['queued'] for j in series[:k]) < RAW_WAIT:
            continue  # RTS2 still archiving the raw frame
        ok = run_step(obs, series, k, args)
        state.setdefault('done' if ok else 'failed', []).append(k)
        _save(spath, state)
        did = True
        break   # newest data first: go back to the spool before the next step
    return did


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True, help='runner directory (spool/, obs/, log, lock)')
    ap.add_argument('-j', '--jobs', type=int, default=3, help='parallel bins in kinstep (default 3)')
    ap.add_argument('--idle', type=float, default=1800, help='exit after this many idle seconds (default 1800)')
    ap.add_argument('--timeout', type=float, default=900, help='kinstep timeout [s] (default 900)')
    ap.add_argument('--kinstep-args', default='', help="extra kinstep arguments (one string, e.g. '--dark D --flat F')")
    ap.add_argument('--once', action='store_true', help='one pass, then exit')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    root = os.path.abspath(args.dir)
    os.makedirs(os.path.join(root, 'spool'), exist_ok=True)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        filename=None if args.once else os.path.join(root, 'kinseries.log'),
                        format='%(asctime)s [%(process)d] %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    os.environ.setdefault('TERM', 'xterm')

    lockf = open(os.path.join(root, 'lock'), 'w')
    while True:
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0   # another runner is at work, it will see our job
        log.info("runner started")
        idle_since = time.time()
        while True:
            try:
                did = serve(root, args)
            except Exception:
                log.exception("pass failed")
                did = False
                time.sleep(10)
            if args.once:
                break
            if did:
                idle_since = time.time()
                continue
            if time.time() - idle_since > args.idle:
                break
            time.sleep(2)
        fcntl.flock(lockf, fcntl.LOCK_UN)
        log.info("runner idle, exiting")
        # a job may have been spooled while we were deciding to exit, by an imgproc whose
        # runner found the lock still held: look once more after letting the lock go
        if args.once or not glob.glob(os.path.join(root, 'spool', '*.json')):
            return 0


if __name__ == '__main__':
    sys.exit(main())
