#!/usr/bin/env python3
"""
The photometric database (PostgreSQL "mates" on hog), what lcgen reads.

One img row per frame, named by the ECSV's FITSFILE
(20261022171815-224-r-120-df.fits), and its obs rows.  This is what
~/ecsv/mk-img.py on hog does for the real-time upload, plus deletion, which
reprocessing needs: a frame that fails now was most likely a bad solution
that went undetected before, and its old rows must go.

A frame is matched by the prefix of its name, YYYYMMDDHHMMSS-mmm- (CTIME and
USEC), not the full name: the filter and exposure parts were spelled
differently over the years (R/r), and a new record must replace the old one
rather than sit beside it.
"""

import argparse
import logging
import sys
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DBNAME = 'mates'

# mk-img.py: FILTER substring -> inst id
_INSTRUMENTS = [('B', 1), ('V', 2), ('R', 3), ('I', 4),
                ('Sloan_g', 5), ('Sloan_r', 6), ('Sloan_i', 7), ('Sloan_z', 8),
                ('N', 9), ('g', 5), ('r', 6), ('i', 7), ('z', 8)]

# obs rows are kept to 5 sigma (mk-img.py)
MAX_MAGERR = 1.091 / 5


def connect(dbname=DBNAME):
    import psycopg2
    return psycopg2.connect(dbname=dbname)


def _instrument(filter_name):
    for key, code in _INSTRUMENTS:
        if key in filter_name:
            return code
    return None


def frame_prefix(ctime, usec):
    """Name prefix of the frame in the database, as calibrate names it"""
    t = datetime.fromtimestamp(int(ctime), tz=timezone.utc)
    return f"{t:%Y%m%d%H%M%S}-{int(usec) // 1000:03d}-"


def _delete(cur, prefix):
    """Delete the img and obs rows of the frame (no commit); number of img rows"""
    # LIKE alone cannot use the index on name, which is en_US collated; the
    # range over the 14 timestamp digits can (digits sort as expected there,
    # unlike '-', which that collation ignores) and LIKE then filters the few
    # rows of that second.
    stamp = prefix[:14]
    cur.execute("SELECT id FROM img WHERE name >= %s AND name < %s AND name LIKE %s",
                (stamp, str(int(stamp) + 1), prefix + '%'))
    ids = [r[0] for r in cur.fetchall()]
    if ids:
        cur.execute("DELETE FROM obs WHERE img = ANY(%s)", (ids,))
        cur.execute("DELETE FROM img WHERE id = ANY(%s)", (ids,))
    return len(ids)


def delete_frame(conn, prefix):
    """Delete the frame's img and obs rows; number of img rows deleted"""
    with conn.cursor() as cur:
        n = _delete(cur, prefix)
    conn.commit()
    return n


def insert_ecsv(conn, ecsv_path):
    """Replace the frame's rows by the ECSV; (name, number of obs rows)"""
    from astropy.table import Table
    from psycopg2.extras import execute_values

    data = Table.read(str(ecsv_path), format='ascii.ecsv')
    meta = data.meta
    name = meta.get('FITSFILE')
    if not name:
        raise ValueError(f"{ecsv_path}: no FITSFILE in the metadata")
    filter_name = str(meta.get('FILTER', ''))

    with conn.cursor() as cur:
        _delete(cur, name[:19])
        cur.execute("INSERT INTO img (name, exptime, jd, target, filter, inst) "
                    "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                    (name, float(meta.get('EXPTIME', 0)), float(meta.get('JD', 0)),
                     meta.get('TARGET'), filter_name, _instrument(filter_name)))
        img_id = cur.fetchone()[0]
        rows = [(float(r['ALPHA_J2000']), float(r['DELTA_J2000']),
                 float(r['MAG_CALIB']), float(r['MAGERR_CALIB']), img_id)
                for r in data if float(r['MAGERR_CALIB']) < MAX_MAGERR]
        if rows:
            execute_values(cur, "INSERT INTO obs (ra, dec, mag, dmag, img) VALUES %s", rows)
    conn.commit()
    return name, len(rows)


def main():
    parser = argparse.ArgumentParser(description='Photometric database: insert or delete frames')
    parser.add_argument('ecsv', nargs='*', help='ECSV files to insert (replacing the frame)')
    parser.add_argument('--delete', metavar='PREFIX', action='append', default=[],
                        help='Delete the frame(s) whose name starts with PREFIX '
                             '(e.g. 20261022171815-224-); repeatable')
    parser.add_argument('--dbname', default=DBNAME)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    conn = connect(args.dbname)
    try:
        for prefix in args.delete:
            if len(prefix) < 19:
                sys.exit(f"--delete {prefix}: give the whole YYYYMMDDHHMMSS-mmm- prefix")
            logger.info(f"{prefix}: {delete_frame(conn, prefix)} deleted")
        for path in args.ecsv:
            name, n = insert_ecsv(conn, path)
            logger.info(f"{name}: {n} records")
    finally:
        conn.close()


if __name__ == '__main__':
    main()
