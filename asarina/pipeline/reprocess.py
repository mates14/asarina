#!/usr/bin/env python3
"""
Reprocess the image archives with asarina-imgproc, newest frame first.

Meant to run for weeks as a service (hog: D50 and SBT archives, ~25 TB), so
it never lists an archive as a whole: night directories are listed one at a
time, newest first, as the work reaches them, and everything known is kept in
a SQLite state file.  A restart - the service's or the machine's - resumes
where it stopped; frames that were being processed go back to the queue.

Each frame ends in one of
    done      ECSV in phdb, inserted into the photometric database (if it
              passes the db_* gate of the archive's asarina config, else the
              frame is deleted from the database)
    failed    with the reason imgproc gave (its exit code), and its rows
              deleted from the database: a frame that fails now was most
              likely a bad solution that went undetected before
    skipped   not an object frame
Failures are recoverable (a dark or flat may be supplied, the catalog come
back, a crash be fixed) or not (no stars, no solution, a rejected one); the
report says which, and lists the missing darks and flats.  "retry" puts
chosen frames back in the queue, e.g. after making the missing darks.

    asarina-reprocess run                     # the service
    asarina-reprocess report [--by year]
    asarina-reprocess list --reason nosolve --limit 20
    asarina-reprocess retry --reason nodark --chip andor46 --nights 20160101-20171231

Setup (~/.config/asarina/reprocess.ini):

    [reprocess]
    state = /storage2/reprocess/state.sqlite
    logdir = /storage2/reprocess/log
    workers = 20
    timeout = 1800          # s per frame
    nice = 10
    settle_days = 2         # leave the latest nights to real time
    photdb = yes            # insert into / delete from the photometric DB

    [archive D50]
    root = /storage/archive-images/D50/images
    config = ~/.config/asarina/d50.conf       # asarina config for imgproc
    cameras = C0                              # CCD directories, default all
    args = --sbt-window-patch                 # more imgproc options
    until = 20261028                          # last night, default all but settle_days

phdb_root, phdb_date_fmt and phdb_by_night go to the asarina configs.
"""

import argparse
import configparser
import logging
import os
import shlex
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

SETUP_FILE = '~/.config/asarina/reprocess.ini'

# Directories under a CCD directory that hold no science frames
NON_SCIENCE = {'darks', 'flats', 'masterdarks', 'masterflats'}

RECOVERABLE = {'nodark', 'noflat', 'catalog', 'dophot', 'crash', 'timeout', 'nooutput'}
IRRECOVERABLE = {'unreadable', 'nostars', 'nosolve', 'astqual', 'idnum'}

SCHEMA = """
CREATE TABLE IF NOT EXISTS nights (
    archive TEXT, night TEXT, scanned REAL, nframes INTEGER,
    PRIMARY KEY (archive, night));
CREATE TABLE IF NOT EXISTS frames (
    path TEXT PRIMARY KEY, archive TEXT, night TEXT, ccd TEXT, name TEXT,
    status TEXT, reason TEXT, detail TEXT,
    chip TEXT, imagetyp TEXT, filter TEXT, exptime REAL, temp REAL, binx INTEGER,
    ecsv TEXT, db TEXT, tries INTEGER DEFAULT 0,
    started REAL, finished REAL, seconds REAL, version TEXT);
CREATE INDEX IF NOT EXISTS frames_queue ON frames (archive, status, night, name);
"""


def failure_class(reason):
    if reason in RECOVERABLE:
        return 'recoverable'
    if reason in IRRECOVERABLE:
        return 'irrecoverable'
    return 'unknown'


def _code_version():
    """Short commits of pyrt and asarina, '+' when modified"""
    import pyrt, asarina
    out = []
    for mod in (pyrt, asarina):
        d = Path(mod.__file__).resolve().parent
        try:
            rev = subprocess.run(['git', '-C', str(d), 'rev-parse', '--short', 'HEAD'],
                                 capture_output=True, text=True).stdout.strip()
            dirty = subprocess.run(['git', '-C', str(d), 'status', '--porcelain', '-uno'],
                                   capture_output=True, text=True).stdout.strip()
            out.append(f"{mod.__name__} {rev}{'+' if dirty else ''}")
        except OSError:
            out.append(f"{mod.__name__} ?")
    return ', '.join(out)


class Archive:
    def __init__(self, name, section):
        self.name = name
        self.root = Path(section['root'])
        self.config = os.path.expanduser(section.get('config', '')) or None
        cams = section.get('cameras', '').replace(',', ' ').split()
        self.cameras = set(cams) or None
        self.args = shlex.split(section.get('args', ''))
        self.until = section.get('until')   # YYYYMMDD: no later night (it is left to real time)
        self.cfg = None     # asarina config [DEFAULT], for the phdb layout

    def pipeline(self):
        """PhotometryPipeline laid out as imgproc will be, to find its ECSVs"""
        from asarina.config import load_config
        from asarina.pipeline.ingest import PhotometryPipeline
        cfg = load_config(self.config)
        # the archive's imgproc args may put the ECSVs elsewhere
        phdb_root = cfg.get('phdb_root')
        for i, arg in enumerate(self.args):
            if arg == '--phdb-root' and i + 1 < len(self.args):
                phdb_root = self.args[i + 1]
            elif arg.startswith('--phdb-root='):
                phdb_root = arg.split('=', 1)[1]
        return PhotometryPipeline(phdb_root=phdb_root or '/home/mates/phdb',
                                  phdb_date_fmt=cfg.get('phdb_date_fmt') or '%y%m',
                                  phdb_by_night=bool(cfg.get('phdb_by_night')))

    def nights(self):
        """Night directories, newest first"""
        found = []
        for y in os.scandir(self.root):
            if not (y.is_dir() and y.name.isdigit() and len(y.name) == 4):
                continue
            for n in os.scandir(y.path):
                if n.is_dir() and n.name.isdigit() and len(n.name) == 8:
                    found.append(n.name)
        return sorted(found, reverse=True)

    def scan_night(self, night):
        """(path, ccd) of the frames of a night: target directories and queue/,
        or the frames right in the night directory"""
        frames = []
        night_dir = self.root / night[:4] / night
        for ccd in os.scandir(night_dir):
            if ccd.is_file() and ccd.name.endswith(('.fits', '.fit')):
                frames.append((ccd.path, ''))       # one camera, frames in the night (Makak)
                continue
            if not ccd.is_dir():
                continue
            for target in os.scandir(ccd.path):
                if not target.is_dir() or target.name in NON_SCIENCE:
                    continue
                for f in os.scandir(target.path):
                    if f.is_file() and f.name.endswith(('.fits', '.fit')):
                        frames.append((f.path, ccd.name))
        return frames


class State:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=120)
        self.db.executescript(SCHEMA)

    def execute(self, sql, args=()):
        cur = self.db.execute(sql, args)
        self.db.commit()
        return cur

    def query(self, sql, args=()):
        return self.db.execute(sql, args).fetchall()


def _read_header(path):
    from astropy.io import fits
    from pyrt.utils.chip_id import get_camera_id
    h = fits.getheader(path)
    return h, dict(chip=get_camera_id(h), imagetyp=str(h.get('IMAGETYP', '')).strip(),
                   filter=str(h.get('FILTER', '')).strip(), exptime=h.get('EXPTIME'),
                   temp=h.get('CCD_TEMP'), binx=h.get('BINX', 1))


class Reprocessor:
    def __init__(self, setup, archives):
        r = setup['reprocess']
        self.state = State(os.path.expanduser(r.get('state', '~/reprocess/state.sqlite')))
        self.logdir = Path(os.path.expanduser(r.get('logdir', '~/reprocess/log')))
        self.workers = r.getint('workers', max(1, (os.cpu_count() or 2) - 2))
        self.timeout = r.getfloat('timeout', 1800)
        self.nice = r.getint('nice', 10)
        self.settle_days = r.getint('settle_days', 2)
        self.use_photdb = r.getboolean('photdb', False)
        self.archives = archives
        self.pipelines = {a.name: a.pipeline() for a in archives}
        self.version = _code_version()
        self.stopping = False
        self.children = {}          # path -> Popen
        self.lock = threading.Lock()
        self.local = threading.local()
        self.night_lists = {}
        self.nights_listed = 0

    # --- queue -----------------------------------------------------------

    def _refresh_nights(self):
        if time.time() - self.nights_listed < 3600:
            return
        last = (datetime.now(timezone.utc) - timedelta(hours=12, days=self.settle_days)).strftime('%Y%m%d')
        for a in self.archives:
            scanned = {n for (n,) in self.state.query(
                "SELECT night FROM nights WHERE archive=?", (a.name,))}
            self.night_lists[a.name] = [n for n in a.nights() if n <= min(last, a.until or last)
                                        and n not in scanned]
            logger.info(f"{a.name}: {len(self.night_lists[a.name])} nights not yet listed")
        self.nights_listed = time.time()

    def _scan_next_night(self, a):
        """List the newest night not yet listed; False when there is none"""
        todo = self.night_lists.get(a.name)
        if not todo:
            return False
        night = todo.pop(0)
        t = time.time()
        frames = a.scan_night(night)
        self.state.db.executemany(
            "INSERT OR IGNORE INTO frames (path, archive, night, ccd, name, status) "
            "VALUES (?, ?, ?, ?, ?, 'pending')",
            [(p, a.name, night, ccd, os.path.basename(p)) for p, ccd in frames])
        self.state.db.execute("INSERT OR REPLACE INTO nights VALUES (?, ?, ?, ?)",
                              (a.name, night, time.time(), len(frames)))
        self.state.db.commit()
        logger.info(f"{a.name} {night}: {len(frames)} frames listed in {time.time() - t:.1f}s")
        return True

    def _next_frame(self, a, exclude):
        while True:
            sql = "SELECT path FROM frames WHERE archive=? AND status='pending'"
            args = [a.name]
            if a.cameras:
                sql += f" AND ccd IN ({','.join('?' * len(a.cameras))})"
                args += sorted(a.cameras)
            sql += " ORDER BY night DESC, name DESC LIMIT ?"
            rows = self.state.query(sql, args + [len(exclude) + 1])
            for (path,) in rows:
                if path not in exclude:
                    return path
            if not self._scan_next_night(a):
                return None

    # --- one frame (worker thread) -----------------------------------------

    def _photdb(self):
        if getattr(self.local, 'conn', None) is None:
            from asarina.pipeline import photdb
            self.local.conn = photdb.connect()
        return self.local.conn

    def _db_action(self, a, header, ecsv):
        """Insert the frame into the photometric DB, or delete it; what was done"""
        if not self.use_photdb:
            return None
        from asarina.pipeline import photdb
        from asarina.pipeline.pipeline_utils import db_gate
        from asarina.config import load_config
        try:
            if ecsv:
                from astropy.table import Table
                cfg = load_config(a.config, str(header.get('CCD_NAME', '')).strip() or None)
                reason = db_gate(Table.read(ecsv, format='ascii.ecsv').meta, cfg)
                if not reason:
                    name, n = photdb.insert_ecsv(self._photdb(), ecsv)
                    return f"inserted {n}"
            else:
                reason = 'failed'
            prefix = photdb.frame_prefix(header['CTIME'], header.get('USEC', 0))
            n = photdb.delete_frame(self._photdb(), prefix)
            return f"deleted {n}: {reason}"
        except Exception as e:
            try:
                self.local.conn.rollback()
            except Exception:
                self.local.conn = None
            return f"error: {e}"

    def _old_ecsvs(self, a, path):
        """ECSVs of the frame already in phdb"""
        found = self.pipelines[a.name]._check_existing_ecsv(path)
        if not found:
            return []
        stem = Path(path).name.replace('-RA.fits', '').replace('.fits', '')
        return list(Path(found).parent.glob(f"{stem}*.ecsv"))

    def process(self, a, path):
        t0 = time.time()
        res = dict(status='failed', reason=None, detail=None, ecsv=None, db=None)
        try:
            header, info = _read_header(path)
        except Exception as e:
            res.update(reason='unreadable', detail=str(e))
            return res, {}, ''
        if info['imagetyp'].lower() != 'object':
            res.update(status='skipped', reason=info['imagetyp'] or 'no IMAGETYP')
            return res, info, ''

        # The frame's results are those of this run: nothing earlier stays
        for old in self._old_ecsvs(a, path):
            old.unlink()

        cmd = ['nice', '-n', str(self.nice), 'asarina-imgproc', '-f', '--no-raw-update',
               '--dophot-workers', '1']
        if a.config:
            cmd += ['--config', a.config]
        cmd += a.args + [path]
        # One frame per core: numpy's BLAS must not start a thread per core in
        # each of them (load 78 on hog's 24 cores with 16 workers).
        env = dict(os.environ, TERM=os.environ.get('TERM', 'xterm'), DISPLAY='', LANG='C',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        env.pop('RTS2_STAT_DIR', None)
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=env, start_new_session=True)
        with self.lock:
            self.children[path] = proc
        try:
            out, _ = proc.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, _ = proc.communicate()
            res.update(reason='timeout', detail=f"killed after {self.timeout:.0f}s")
        finally:
            with self.lock:
                self.children.pop(path, None)

        # Stopped by us (imgproc exits 1 on SIGTERM): back to the queue.  A
        # genuine failure that ends during a stop is merely retried.
        if self.stopping and res['reason'] is None and proc.returncode != 0:
            res.update(status='pending')
            return res, info, out
        if res['reason'] is None:
            from asarina.pipeline.imgproc import EXIT_CODES
            reasons = {v: k for k, v in EXIT_CODES.items()}
            ecsvs = self._old_ecsvs(a, path)
            if proc.returncode == 0 and ecsvs:
                res.update(status='done', reason=None, ecsv=str(ecsvs[0]))
            elif proc.returncode == 0:
                res.update(reason='nooutput')
            else:
                res.update(reason=reasons.get(proc.returncode, 'crash'))
            if res['status'] == 'failed':
                # the last error imgproc logged, e.g. which dark was missing
                lines = [ln for ln in out.splitlines() if ln.strip()]
                errors = [ln for ln in lines if 'ERROR' in ln and 'FAILED:' not in ln]
                last = errors[-1] if errors else (lines[-1] if lines else f"exit {proc.returncode}")
                res['detail'] = last[-500:]
        if res['reason'] == 'notobject':
            res['status'] = 'skipped'
        elif res['status'] in ('done', 'failed'):
            res['db'] = self._db_action(a, header, res['ecsv'])
        res['seconds'] = time.time() - t0
        return res, info, out

    # --- bookkeeping (main thread) -----------------------------------------

    def _record(self, a, path, res, info, out):
        s = self.state
        if res['status'] == 'pending':
            s.execute("UPDATE frames SET status='pending' WHERE path=?", (path,))
            return
        s.execute("UPDATE frames SET status=?, reason=?, detail=?, ecsv=?, db=?, "
                  "chip=?, imagetyp=?, filter=?, exptime=?, temp=?, binx=?, "
                  "finished=?, seconds=?, version=?, tries=tries+1 WHERE path=?",
                  (res['status'], res['reason'], res['detail'], res['ecsv'], res['db'],
                   info.get('chip'), info.get('imagetyp'), info.get('filter'),
                   info.get('exptime'), info.get('temp'), info.get('binx'),
                   time.time(), res.get('seconds'), self.version, path))
        night = s.query("SELECT night FROM frames WHERE path=?", (path,))[0][0]
        log = self.logdir / a.name / f"{night}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, 'a') as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {os.path.basename(path)} "
                    f"{res['status']} {res['reason'] or ''} {res.get('seconds', 0):.0f}s "
                    f"db={res['db']}\n")
            if res['status'] == 'failed' and out:
                f.write(''.join(f"    {ln}\n" for ln in out.rstrip().splitlines()))
        logger.info(f"{a.name} {os.path.basename(path)}: {res['status']} "
                    f"{res['reason'] or ''} ({res.get('seconds', 0):.0f}s)")

    def run(self):
        def stop(*_):
            logger.info("stopping: killing running frames, they stay in the queue")
            self.stopping = True
            with self.lock:
                for proc in self.children.values():
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)

        n = self.state.execute("UPDATE frames SET status='pending' WHERE status='running'").rowcount
        if n:
            logger.info(f"{n} frames interrupted last time, back in the queue")
        logger.info(f"{self.workers} workers, {self.version}")

        running = {}            # future -> (archive, path)
        turn = 0
        with ThreadPoolExecutor(self.workers) as pool:
            while not (self.stopping and not running):
                self._refresh_nights()
                idle = 0
                while not self.stopping and len(running) < self.workers and idle < len(self.archives):
                    a = self.archives[turn % len(self.archives)]
                    turn += 1
                    path = self._next_frame(a, {p for _, p in running.values()})
                    if path is None:
                        idle += 1
                        continue
                    idle = 0
                    self.state.execute("UPDATE frames SET status='running', started=? WHERE path=?",
                                       (time.time(), path))
                    running[pool.submit(self.process, a, path)] = (a, path)
                if not running:
                    if self.stopping:
                        break
                    logger.info("nothing to do, next look in an hour")
                    self.nights_listed = 0
                    for _ in range(3600):
                        if self.stopping:
                            break
                        time.sleep(1)
                    continue
                done, _ = wait(running, timeout=60, return_when=FIRST_COMPLETED)
                for fut in done:
                    a, path = running.pop(fut)
                    try:
                        self._record(a, path, *fut.result())
                    except Exception as e:
                        logger.exception(f"{path}: {e}")
                        self.state.execute("UPDATE frames SET status='failed', reason='crash', "
                                           "detail=? WHERE path=?", (str(e)[-500:], path))


# --- reports -----------------------------------------------------------------

def _where(args):
    sql, params = [], []
    for col, val in (('archive', args.archive), ('chip', args.chip), ('ccd', args.ccd),
                     ('filter', args.filter)):
        if val:
            sql.append(f"{col}=?")
            params.append(val)
    if args.reason:
        sql.append(f"reason IN ({','.join('?' * len(args.reason))})")
        params += args.reason
    if args.status:
        sql.append("status=?")
        params.append(args.status)
    if args.nights:
        lo, _, hi = args.nights.partition('-')
        sql.append("night BETWEEN ? AND ?")
        params += [lo, hi or lo]
    return (' WHERE ' + ' AND '.join(sql)) if sql else '', params


def report(state, args):
    where, params = _where(args)
    by = {'year': "substr(night,1,4)", 'ccd': 'ccd', 'chip': 'chip', 'night': 'night'}.get(args.by)
    group = f"archive{', ' + by if by else ''}"

    print("Nights listed:")
    for archive, n, frames in state.query(
            "SELECT archive, count(*), sum(nframes) FROM nights GROUP BY archive"):
        print(f"  {archive:6s} {n:6d} nights  {frames or 0:9d} frames")

    print("\nFrames:")
    table = defaultdict(dict)
    statuses = ['pending', 'running', 'done', 'failed', 'skipped']
    for row in state.query(f"SELECT {group}, status, count(*) FROM frames{where} "
                           f"GROUP BY {group}, status", params):
        table[tuple(row[:-2])][row[-2]] = row[-1]
    print(f"  {'':16s}" + ''.join(f"{s:>10s}" for s in statuses))
    for key in sorted(table):
        print(f"  {' '.join(str(k) for k in key):16s}"
              + ''.join(f"{table[key].get(s, 0):10d}" for s in statuses))

    print("\nFailures:")
    for row in state.query(f"SELECT {group}, reason, count(*) FROM frames{where}"
                           f"{' AND' if where else ' WHERE'} status='failed' "
                           f"GROUP BY {group}, reason ORDER BY {group}, count(*) DESC", params):
        key, reason, n = row[:-2], row[-2], row[-1]
        print(f"  {' '.join(str(k) for k in key):16s} {reason:12s} {failure_class(reason):14s} {n:9d}")

    cond = f"{where}{' AND' if where else ' WHERE'} status='failed' AND reason="
    rows = state.query(f"SELECT archive, chip, substr(night,1,4), binx, round(temp), exptime, count(*) "
                       f"FROM frames{cond}'nodark' GROUP BY 1,2,3,4,5,6 ORDER BY 1,2,3,4,5,6", params)
    if rows:
        print("\nMissing darks (archive chip year bin temp exptime: frames):")
        for *key, n in rows:
            print(f"  {' '.join(str(k) for k in key)}: {n}")
    rows = state.query(f"SELECT archive, chip, substr(night,1,4), binx, filter, count(*) "
                       f"FROM frames{cond}'noflat' GROUP BY 1,2,3,4,5 ORDER BY 1,2,3,4,5", params)
    if rows:
        print("\nMissing flats (archive chip year bin filter: frames):")
        for *key, n in rows:
            print(f"  {' '.join(str(k) for k in key)}: {n}")

    rows = state.query("SELECT count(*), avg(seconds) FROM frames WHERE finished > ?",
                       (time.time() - 86400,))
    if rows and rows[0][0]:
        print(f"\nLast 24 h: {rows[0][0]} frames, {rows[0][1]:.0f} s per frame")
    rows = state.query("SELECT db, count(*) FROM frames WHERE db LIKE 'error%' GROUP BY db")
    if rows:
        print("\nPhotometric DB errors:")
        for db, n in rows:
            print(f"  {n:6d} {db[:100]}")


def list_frames(state, args):
    where, params = _where(args)
    for row in state.query(f"SELECT path, status, reason, detail FROM frames{where} "
                           f"ORDER BY night DESC, name DESC LIMIT ?", params + [args.limit]):
        print(' | '.join(str(v) for v in row if v is not None))


def retry(state, args):
    if not (args.reason or args.status):
        sys.exit("retry: give --reason or --status (e.g. --status done to redo finished frames)")
    where, params = _where(args)
    n = state.execute(f"UPDATE frames SET status='pending' {where}", params).rowcount
    print(f"{n} frames back in the queue")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--setup', default=SETUP_FILE, help=f'default {SETUP_FILE}')
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('run', help='process the archives (the service)')
    for name, hlp in (('report', 'counts, failures, missing calibrations'),
                      ('list', 'frames matching the filters'),
                      ('retry', 'put matching frames back in the queue')):
        p = sub.add_parser(name, help=hlp)
        p.add_argument('--archive')
        p.add_argument('--chip', help='camera id (andor46, mi6166, ...)')
        p.add_argument('--ccd', help='CCD directory (C0, C1, ...)')
        p.add_argument('--filter')
        p.add_argument('--reason', action='append', help='failure reason, repeatable')
        p.add_argument('--status', help='pending, running, done, failed, skipped')
        p.add_argument('--nights', metavar='YYYYMMDD[-YYYYMMDD]')
        if name == 'report':
            p.add_argument('--by', choices=['year', 'ccd', 'chip', 'night'])
        if name == 'list':
            p.add_argument('--limit', type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format='%(asctime)s %(levelname)s: %(message)s')
    setup = configparser.ConfigParser(inline_comment_prefixes=('#',))
    if not setup.read(os.path.expanduser(args.setup)):
        sys.exit(f"cannot read {args.setup}")
    if 'reprocess' not in setup:
        setup['reprocess'] = {}

    if args.cmd == 'run':
        archives = [Archive(s.split(None, 1)[1], setup[s]) for s in setup.sections()
                    if s.startswith('archive ')]
        if not archives:
            sys.exit(f"{args.setup}: no [archive NAME] section")
        Reprocessor(setup, archives).run()
        return

    state = State(os.path.expanduser(setup['reprocess'].get('state', '~/reprocess/state.sqlite')))
    {'report': report, 'list': list_frames, 'retry': retry}[args.cmd](state, args)


if __name__ == '__main__':
    main()
