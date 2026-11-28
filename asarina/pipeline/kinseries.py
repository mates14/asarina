#!/usr/bin/env python3
"""
kinseries - real-time runner of the progressive light curve of a GRB/EP sequence.

Two kinds of input, both from asarina-imgproc once it has saved a frame's ECSV:
  - kinetic series (D50 C0: ACQMODE 3, KINNUM >= 12, KINFILE): the sub-frames, kinstep.py,
  - short frames (SBT C1+C2, config kinseries_frames: a trigger target, EXPTIME <=
    kinseries_max_exptime): imgproc copies the calibrated frame along, framestep.py.

imgproc drops a job into <dir>/spool/ and starts this runner (as the pipeline user,
detached, in its own session).  Only one runner works at a time (flock on <dir>/lock);
a second one exits at once.  With nothing to do for --idle seconds the runner exits;
the next kinetic series starts it again.

The runner groups the jobs by observation (OBSID, filter; C1 and C2 of SBT together).
Frames of different exposures (FRAM: a 20 s fast block, then 60 s) share one light curve
in bins of equal summed exposure (framestep.exposure_bins): a step runs on the longest
prefix of the frames that splits into 12 such bins exactly, using as much of a leading block
of other exposures as fits (frame_starts).
Kinetic series: with n series in, it runs the newest "trivial" step k <= n not done yet
(k = 1, 2, 3, 4, 6, 12, see kinstep.py).  Frames: with n frames in, step k = n // 12
(12 bins of k frames).  A step that has gone stale while an earlier one was running is
skipped, the newest data are what matters.  After a step, the flare search
(pyrt-transients --flare-rec) runs on its bins in the background.

Layout of <dir> (config key kinseries_dir, e.g. /home/mates/kinseries):
  spool/<stem>.json                 jobs from imgproc
  obs/<YYYYMMDD>-<obsid>-<filter>/  one per observation:
      series/<stem>.json            its series or frames (the jobs, moved here)
      frames/<stem>.fits            the calibrated frames (frame jobs; removed when the
                                    observation is over)
      state.json                    steps done / failed / reported
      step<kk>/                     kinstep output: per-bin ECSVs, deep ECSV, step.json,
                                    kinstep.log, transients.log, flares.txt
  kinseries.log                     the runner log
  lock
  step-hook                         optional, executable: run with the observation directory
                                    after each step's report (trigreport update)

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
MAX_FRAME_STEP = 12   # frames: up to 12 x 12 frames (12 min of SBT 5 s frames); framestep's
                      # cube mode held the step's whole cube (~60 MB per SBT frame); with
                      # --mode combine (bins by pyrt-combine) the ladder can go on
FRAME_LADDER = (1, 2, 3, 5, 8, 12, 18, 26, 40, 60)   # frames per bin, ~1.5^n: a step costs ~its
                      # frames, so with the linear 1..12 long series spent hours on steps 9-12
# Long frames (FRAM: 60 s + ~10 s overhead) would make the first light curve wait for
# 12 frames, a quarter of an hour.  From EARLY_EXPTIME up, a step 0 runs on the first
# EARLY_FRAMES frames as they are (EARLY_FRAMES points), then the usual 12 bins of k.
EARLY_EXPTIME = 30.0
EARLY_FRAMES = 6
RAW_WAIT = 120        # [s] how long a series may wait for its raw frame to be archived
REPORT_POLL = 5       # [s] while a step runs, look for the previous step's search results this often
OBS_ACTIVE = 4 * 3600  # [s] observations untouched for longer are not looked at


# ---------------------------------------------------------------- jobs

def is_trigger(hdr):
    """A GRB/EP/... trigger target: RTS2 type G, or the trigger position in the header."""
    return str(hdr.get('TARTYPE', '')).strip().upper() == 'G' or hdr.get('GRB_RA') is not None


def queue_job(root, hdr, raw_path, ecsv_path, calibrated=None, frames=False, max_exptime=15.0):
    """Called by imgproc: drop a job for a kinetic series (or, with frames=True, a short
    frame of a trigger target, whose calibrated image is copied along), start the runner.
    Returns the job path, or None when the frame is neither."""
    stem = os.path.basename(str(raw_path))[:18]
    job = {
        'stem': stem, 'raw': os.path.abspath(str(raw_path)), 'ecsv': os.path.abspath(str(ecsv_path)),
        'obsid': int(hdr.get('OBSID') or 0), 'target': hdr.get('TARGET'),
        'object': str(hdr.get('OBJECT', '')), 'filter': str(hdr.get('FILTER', 'UNK')),
        'ctime': float(hdr.get('CTIME', 0)) + float(hdr.get('USEC', 0)) / 1e6,
        'exptime': float(hdr.get('EXPTIME', 0)), 'ccd': hdr.get('CCD_NAME'), 'queued': time.time(),
    }
    root = os.path.abspath(root)
    spool = os.path.join(root, 'spool')
    acq = str(hdr.get('ACQMODE', '')).strip().lower()
    kinnum = int(hdr.get('KINNUM') or 0)
    if acq in ('3', 'kinetic series', 'kinetics'):
        if kinnum < POINTS or not hdr.get('KINFILE') or not hdr.get('KINCYCLE'):
            return None
        job.update(kind='kinetic', kinfile=hdr['KINFILE'], kinnum=kinnum, kincycle=float(hdr['KINCYCLE']))
        os.makedirs(spool, exist_ok=True)
    elif frames and calibrated and is_trigger(hdr) and 0 < job['exptime'] <= max_exptime:
        job['kind'] = 'frame'
        os.makedirs(spool, exist_ok=True)
        tmp = os.path.join(spool, f".{stem}.fits.tmp")
        shutil.copyfile(str(calibrated), tmp)
        os.replace(tmp, os.path.join(spool, f"{stem}.fits"))
        job['fits'] = os.path.join(spool, f"{stem}.fits")   # moved with the job (ingest)
    else:
        return None
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
        if job.get('kind') == 'frame':
            os.makedirs(os.path.join(obs, 'frames'), exist_ok=True)
            dst = os.path.join(obs, 'frames', os.path.basename(job['fits']))
            if os.path.exists(job['fits']):
                os.replace(job['fits'], dst)
            job['fits'] = dst
            _save(path, job)
            log.info(f"{os.path.basename(obs)}: frame {job['stem']} {job['ccd']} ({job['object']}, {job['exptime']:g} s)")
        else:
            log.info(f"{os.path.basename(obs)}: series {job['stem']} ({job['object']}, "
                     f"{job['kinnum']} x {job['kincycle']:.3f} s)")
        os.replace(path, os.path.join(obs, 'series', f"{job['stem']}.json"))
        n += 1
    return n


def raw_ready(job):
    """The raw frame (archived by RTS2 after imgproc's corrwerr) and the KINFILE exist."""
    from asarina.pipeline.kinphot import find_raw
    if job.get('kind') == 'frame':
        return os.path.exists(job['fits'])
    if job['raw'].endswith('-kin.fits'):
        return os.path.exists(job['raw'])   # a cube (offline tests)
    if not (os.path.exists(job['raw']) or find_raw(job['stem'])):
        return False
    kin = job['kinfile']
    return os.path.exists(kin) or os.path.exists(kin.replace('/var/tmp', '/images/kinmode'))


# ---------------------------------------------------------------- steps

def frame_starts(exps, state):
    """Where a frame step may start: from the last step's start (0 before any) up to the start
    of the last run of one exposure time.  A leading block of other exposures (FRAM: 20 x 20 s,
    then 60 s) is used from its rear edge as far as it fills whole bins together with what
    follows: 18 x 20 s + 6 x 60 s in 60 s bins, 18 x 20 + 18 x 60 s in 120 s bins, 12 x 20 +
    44 x 60 s in 240 s bins, ..., rather than not at all.  The start never moves back."""
    lo = state.get('first', 0)
    tail = len(exps) - 1
    while tail > 0 and exps[tail - 1] == exps[-1]:
        tail -= 1
    return list(range(lo, max(lo, tail) + 1))


def due_frame_step(series, state):
    """Frames: (step number, first frame, end frame, points) of the step to run now, or None.
    The longest prefix past the last step's that splits into POINTS bins of equal summed
    exposure longer than the last step's bins (equal exposures: 12, 24, 36, ... frames;
    12 x 20 s + 8 x 60 s splits too), at most MAX_FRAME_STEP x POINTS frames, from the
    earliest start frame_starts allows.
    Before the first such step, frames of EARLY_EXPTIME and more get step 0: the first
    EARLY_FRAMES of them, one point each."""
    from asarina.pipeline.framestep import exposure_bins
    done = state.get('done', []) + state.get('failed', [])
    last_n = state.get('last_n', 0)
    exps = [j.get('exptime', 0.0) for j in series]
    starts = frame_starts(exps, state)
    # each step's bins are longer than the last one's: the start moves only to grow them
    prev = sum(exps[starts[0]:last_n]) / POINTS if last_n else 0.0
    # the step's size: the largest rung of FRAME_LADDER above the last step's with rung x POINTS
    # frames in (rungs the runner fell behind on are skipped), the longest prefix of at most
    # that many frames that splits evenly
    span = -(-(last_n - starts[0]) // POINTS) if last_n and max(done, default=0) > 0 else 0   # not step 0
    last_k = min([k for k in FRAME_LADDER if k >= span], default=FRAME_LADDER[-1]) if span else 0
    for k in reversed([k for k in FRAME_LADDER
                       if last_k < k <= MAX_FRAME_STEP and len(series) - starts[0] >= k * POINTS]):
        best = None   # the most frames, then the newest
        for n in range(len(series), max(last_n, starts[0]), -1):
            for s0 in starts:
                if n - s0 < POINTS:
                    break
                if n - s0 <= k * POINTS and (best is None or (n - s0, n) > best[0]) \
                        and sum(exps[s0:n]) / POINTS > prev * 1.01 \
                        and exposure_bins(exps[s0:n], POINTS) is not None:
                    best = ((n - s0, n), s0, n)
        if best:
            return max(done, default=0) + 1, best[1], best[2], POINTS
    early = [starts[0]] + ([starts[-1]] if 'first' not in state and starts[-1] > starts[0] else [])
    for s0 in early:
        if not done and len(exps) > s0 and exps[s0] >= EARLY_EXPTIME and len(series) >= s0 + EARLY_FRAMES \
                and exposure_bins(exps[s0:s0 + EARLY_FRAMES], EARLY_FRAMES) is not None:
            return 0, s0, s0 + EARLY_FRAMES, EARLY_FRAMES
    return None


def due_step(nseries, state, kind='kinetic'):
    done = state.get('done', []) + state.get('failed', [])
    avail = [k for k in TRIVIAL if k <= nseries]
    if not avail:
        return None
    k = avail[-1]
    if k in done or k <= max(done, default=0):
        return None
    return k


def run_step(obs, series, k, args, state=None, spath=None, nframes=None, points=POINTS, first=0):
    """kinstep on the first k series of the observation (framestep on its first 12k
    frames); True if it produced bins.  While it runs, the previous step's search results
    are reported as soon as they appear (state, spath: the observation's state), so that
    the first alert does not wait for the next step."""
    out = os.path.join(obs, f"step{k:02d}")
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    if series[0].get('kind') == 'frame':
        use = series[first:nframes]
        cmd = [sys.executable, '-m', 'asarina.pipeline.framestep', '--points', str(points), '--label', str(k),
               '-j', str(args.jobs), '--skel', os.path.join(obs, 'skel.hdr'),   # one grid per observation
               '-o', out] + [f"{j['fits']}:{j['ecsv']}" for j in use]
        cams = '+'.join(sorted(set(str(j['ccd']) for j in use)))
        exps = '+'.join(f"{e:g}" for e in sorted(set(j.get('exptime', 0) for j in use)))
        log.info(f"{os.path.basename(obs)}: step {k} ({len(use)} frames {cams}, {exps} s, "
                 f"{points} bins of {sum(j.get('exptime', 0) for j in use) / points:g} s)")
    else:
        nsub = min(j['kinnum'] for j in series[:k]) // POINTS * POINTS
        cmd = [sys.executable, '-m', 'asarina.pipeline.kinstep', '--step', str(k), '--nsub', str(nsub),
               '-j', str(args.jobs), '-o', out] + args.kinstep_args.split()
        for j in series[:k]:
            cmd += ['--ref', j['ecsv']]
        cmd += [j['raw'] for j in series[:k]]
        log.info(f"{os.path.basename(obs)}: step {k} ({k} series, bins of {nsub * k // POINTS})")
    t = time.time()
    with open(os.path.join(out, 'kinstep.log'), 'w') as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=out)
        while True:
            try:
                rc = p.wait(timeout=REPORT_POLL)
                break
            except subprocess.TimeoutExpired:
                pass
            if state is not None:
                try:
                    if report_flares(obs, state):
                        _save(spath, state)
                except Exception as e:
                    log.warning(f"{os.path.basename(obs)}: report while step {k} runs: {e}")
            if time.time() - t > args.timeout:
                p.kill()
                p.wait()
                rc = 'timeout'
                break
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
    """pyrt-transients on the bins, in the background: the flare search (--flare-rec) and the
    afterglow assessment, its candidates also from the step's deep catalogue (--deep: a faint
    afterglow is significant only there).  Leaves flares.txt and afterglow.txt (the FLARES and
    AFTERGLOW sections of its output; afterglow.ecsv in full) when done."""
    bins = sorted(glob.glob(os.path.join(out, '*-[kf][0-9]*b*.ecsv')))
    if not bins:
        return
    deep = sorted(glob.glob(os.path.join(out, '*-deep*.ecsv')))
    opt = ('--deep %s ' % os.path.basename(deep[0])) if deep else ''
    script = ('pyrt-transients --flare-rec rec.ecsv ' + opt + '"$@" > transients.log 2>&1; '
              'sed -n "/AFTERGLOW START/,/AFTERGLOW END/p" transients.log > afterglow.tmp; mv afterglow.tmp afterglow.txt; '
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
        # the human side: report, page on lascaux, Telegram (trigreport), detached; then the
        # site's step hook, if any (<dir>/step-hook <obsdir>, executable: FRAM LL ships the
        # observation to zeus and starts the deep search there)
        hook = os.path.join(os.path.dirname(os.path.dirname(obs)), 'step-hook')
        cmd = [sys.executable, '-m', 'asarina.pipeline.trigreport', 'update', obs]
        if os.access(hook, os.X_OK):
            cmd = ['sh', '-c', '"$0" "$@"; "$HOOK" "$OBS"'] + cmd
        subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=open(os.path.join(obs, 'trigreport.log'), 'a'),
                         stderr=subprocess.STDOUT, start_new_session=True,
                         env=dict(os.environ, HOOK=hook, OBS=obs))
        # the deep image of the older steps is not needed any more (the chart is the latest's)
        # (only those of earlier steps: a later one may be running right now, see run_step)
        newest = max(state.get('done', [k]))
        for old in glob.glob(os.path.join(obs, 'step[0-9][0-9]', 'deep.fits')):
            if int(os.path.basename(os.path.dirname(old))[4:]) < newest:
                os.remove(old)
        fa = os.path.join(obs, f"step{k:02d}", 'afterglow.txt')
        if os.path.exists(fa):
            with open(fa) as fh:
                ag = [ln.rstrip() for ln in fh if ln.strip() and 'AFTERGLOW' not in ln and 'no new' not in ln]
            log.info(f"{os.path.basename(obs)}: step {k} afterglow: " + (f"{len(ag)} new source(s), best:" if ag else "nothing"))
            for ln in ag[:3]:
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
            # over: the copies of the calibrated frames go (4 MB each, ~150 per trigger)
            for f in glob.glob(os.path.join(obs, 'step[0-9][0-9]', 'deep.fits')):
                os.remove(f)
            if os.path.isdir(os.path.join(obs, 'proj')):   # framestep's projection cache
                shutil.rmtree(os.path.join(obs, 'proj'), ignore_errors=True)
            if os.path.isdir(os.path.join(obs, 'frames')):
                shutil.rmtree(os.path.join(obs, 'frames'), ignore_errors=True)
                log.info(f"{os.path.basename(obs)}: over, calibrated frames removed")
            continue
        spath = os.path.join(obs, 'state.json')
        state = _load(spath, {'done': [], 'failed': []})
        if report_flares(obs, state):
            _save(spath, state)
        series = sorted((_load(p, None) for p in glob.glob(os.path.join(obs, 'series', '*.json'))),
                        key=lambda j: j['ctime'])
        if series and series[0].get('kind') == 'frame':
            due = due_frame_step(series, state)
            if due is None:
                continue
            k, first, nframes, points = due
            need = series[first:nframes]
        else:
            k = due_step(len(series), state, series[0].get('kind', 'kinetic') if series else 'kinetic')
            if k is None:
                continue
            first, nframes, points, need = 0, None, POINTS, series[:k]
        waiting = [j['stem'] for j in need if not raw_ready(j)]
        if waiting and now - max(j['queued'] for j in need) < RAW_WAIT:
            continue  # RTS2 still archiving the raw frame
        ok = run_step(obs, series, k, args, state, spath, nframes=nframes, points=points, first=first)
        state.setdefault('done' if ok else 'failed', []).append(k)
        if nframes:
            state['first'] = first
            state['last_n'] = nframes
            state.setdefault('step_frames', {})[str(k)] = nframes
        _save(spath, state)
        did = True
        break   # newest data first: go back to the spool before the next step
    return did


# ---------------------------------------------------------------- main

def physical_cores():
    """Physical CPU cores (SMT siblings share the FPU, and the bins are floating point)"""
    try:
        cores = set()
        phys = core = None
        with open('/proc/cpuinfo') as f:
            for line in f:
                k, _, v = line.partition(':')
                k = k.strip()
                if k == 'physical id':
                    phys = v.strip()
                elif k == 'core id':
                    core = v.strip()
                elif not k and core is not None:
                    cores.add((phys, core)); phys = core = None
        if core is not None:
            cores.add((phys, core))
        if cores:
            return len(cores)
    except OSError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True, help='runner directory (spool/, obs/, log, lock)')
    ap.add_argument('-j', '--jobs', type=int, default=physical_cores(),
                    help='parallel bins in kinstep/framestep (default: the physical cores, %(default)s here)')
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
