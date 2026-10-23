#!/usr/bin/env python3
"""
Sky map: every good g/r/i frame is weight-added into TAN tiles, one per
HEALPix cell (nside 64, ~0.9 deg), from which Hipsgen builds a browseable HiPS.

A tile is an ordinary TAN image at 3.14"/px covering its whole cell, rotated
to fit it (45 deg along the equatorial belt, whatever the cell needs in the
polar caps), 1056-1730 px.  It keeps the sums, so the next frame can be added
at any time:

    <dir>/<band>/<ipix // 1024>/<ipix>.fits
        primary   sum of w * flux          (with the tile WCS)
        WEIGHT    sum of w
        FRAMES    the frames added (name prefix), a frame is never added twice

flux is per tile pixel, on the AB scale with zero point 25 (m = 25 - 2.5 log f):
the frame's background is subtracted and its ADU are scaled by dophot's
photometric model (Z and the spatial terms, at zero color); w = 1/sigma^2 of
the sky noise, so frames count by their depth.  A running weighted mean cannot
reject outliers: trails and cosmics fade with the number of frames, no more.

`export` writes sum/weight per tile (NaN where nothing was added), the input
directory for Hipsgen:

    java -Xmx64g -jar Hipsgen.jar in=<export>/r out=<hips>/r id=ASU/P/SBT/r \\
         mode=overlayMean
"""

import argparse
import fcntl
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

NSIDE = 64
SCALE = 3.14                # arcsec/px
MARGIN = 8                  # px around the cell, for Hipsgen's bilinear resampling
ZP = 25.0                   # flux scale of the map
BANDS = {'Sloan_g': 'g', 'Sloan_r': 'r', 'Sloan_i': 'i'}

FWHM_MAX = 2.5              # px
DMAGZERO_MAX = 0.05

BOX = 256                   # background grid, px (~13' on SBT)
GRID = 16                   # tile px between exact WCS evaluations

_WCS_KEY = re.compile(r'^(WCSAXES|CTYPE\d|CUNIT\d|CRVAL\d|CRPIX\d|CD\d_\d|PC\d_\d|CDELT\d|'
                      r'PV\d_\d+|LONPOLE|LATPOLE|EQUINOX|RADESYS|'
                      r'A_ORDER|B_ORDER|AP_ORDER|BP_ORDER|A_\d_\d|B_\d_\d|AP_\d_\d|BP_\d_\d)$')


def _healpix():
    from astropy_healpix import HEALPix
    return HEALPix(NSIDE, order='nested')


# --- tiles ---------------------------------------------------------------------

def tile_header(ipix):
    """TAN header of tile ipix: centered on the cell, rotated to fit it"""
    from astropy.io import fits
    from astropy.wcs import WCS
    import astropy.units as u

    hp = _healpix()
    lon, lat = hp.healpix_to_lonlat([ipix])
    blon, blat = hp.boundaries_lonlat([ipix], step=16)
    ra, dec = lon.to_value(u.deg)[0], lat.to_value(u.deg)[0]
    scale = SCALE / 3600

    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [0, 0]
    w.wcs.cd = [[-scale, 0], [0, scale]]
    x, y = w.wcs_world2pix(blon.to_value(u.deg)[0], blat.to_value(u.deg)[0], 0)
    angs = np.radians(np.arange(0.0, 90.0, 1.0))[:, None]
    c, s = np.cos(angs), np.sin(angs)
    half = np.maximum(np.abs(c * x + s * y).max(1), np.abs(-s * x + c * y).max(1))
    k = half.argmin()
    naxis = int(2 * (np.ceil(half[k]) + MARGIN))
    # pixel axes turned by the angle that fits the cell best
    ck, sk = float(np.cos(angs[k, 0])), float(np.sin(angs[k, 0]))

    hdr = fits.Header()
    hdr['NAXIS'] = 2
    hdr['NAXIS1'] = naxis
    hdr['NAXIS2'] = naxis
    hdr['CTYPE1'] = 'RA---TAN'
    hdr['CTYPE2'] = 'DEC--TAN'
    hdr['CRVAL1'] = ra
    hdr['CRVAL2'] = dec
    hdr['CRPIX1'] = (naxis + 1) / 2
    hdr['CRPIX2'] = (naxis + 1) / 2
    hdr['CD1_1'] = -scale * ck
    hdr['CD1_2'] = scale * sk
    hdr['CD2_1'] = scale * sk
    hdr['CD2_2'] = scale * ck
    hdr['RADESYS'] = 'ICRS'
    hdr['EQUINOX'] = 2000.0
    hdr['HPXNSIDE'] = (NSIDE, 'HEALPix nside of the tiling (nested)')
    hdr['HPXIPIX'] = (ipix, 'HEALPix cell of this tile')
    return hdr


def tile_path(mapdir, band, ipix):
    return Path(mapdir) / band / f"{ipix // 1024:02d}" / f"{ipix:05d}.fits"


# --- frame ---------------------------------------------------------------------

def frame_band(meta):
    """Map band of the frame, None if it does not go to the map"""
    band = BANDS.get(str(meta.get('FILTER', '')).strip())
    if band and str(meta.get('PHFILTER', '')).strip() == str(meta.get('FILTER')).strip():
        return band
    return None


def frame_gate(meta):
    """Why the frame does not go to the map, None if it does"""
    if frame_band(meta) is None:
        return f"filter {meta.get('FILTER')} (calibrated to {meta.get('PHFILTER')})"
    if meta.get('ASTQUAL') != 'OK':
        return f"ASTQUAL={meta.get('ASTQUAL')}"
    if meta.get('PHQUAL') != 'OK':
        return f"PHQUAL={meta.get('PHQUAL')}"
    if not float(meta.get('FWHM', 99)) <= FWHM_MAX:
        return f"FWHM={meta.get('FWHM')} > {FWHM_MAX}"
    if not float(meta.get('DMAGZERO', 99)) <= DMAGZERO_MAX:
        return f"DMAGZERO={meta.get('DMAGZERO')} > {DMAGZERO_MAX}"
    return None


def frame_wcs(meta):
    """The final WCS of the frame, as dophot left it in the ECSV"""
    from astropy.io import fits
    from astropy.wcs import WCS
    hdr = fits.Header()
    hdr['NAXIS'] = 2
    hdr['NAXIS1'] = int(meta['IMGAXIS1'])
    hdr['NAXIS2'] = int(meta['IMGAXIS2'])
    for key, value in meta.items():
        if _WCS_KEY.match(str(key)):
            hdr[key] = value
    return WCS(hdr)


class _Smooth:
    """Bilinear interpolation of a smooth quantity given on a regular grid
    (clamped at the grid edges); separable, much cheaper than map_coordinates"""

    def __init__(self, gy, gx, values):
        from scipy.interpolate import RectBivariateSpline
        gy, gx, values = np.asarray(gy, float), np.asarray(gx, float), np.asarray(values, float)
        if gy.size == 1:                # the spline needs two knots per axis
            gy, values = np.array([gy[0], gy[0] + 1]), np.repeat(values, 2, axis=0)
        if gx.size == 1:
            gx, values = np.array([gx[0], gx[0] + 1]), np.repeat(values, 2, axis=1)
        self.gy, self.gx = gy, gx
        self.spline = RectBivariateSpline(gy, gx, values, kx=1, ky=1)

    def grid(self, y, x):
        """On the grid of rows y x columns x"""
        return self.spline(np.clip(y, self.gy[0], self.gy[-1]),
                           np.clip(x, self.gx[0], self.gx[-1])).astype(np.float32)

    def at(self, y, x):
        """At the points (y, x)"""
        return self.spline.ev(np.clip(y, self.gy[0], self.gy[-1]),
                              np.clip(x, self.gx[0], self.gx[-1]))


def _box_centers(n):
    return (np.arange(-(-n // BOX)) + 0.5) * BOX - 0.5


def _background(data, good):
    """Sky background: clipped median in BOX x BOX boxes (every 4th pixel),
    smoothed; a _Smooth over the frame"""
    from scipy.ndimage import median_filter
    import warnings
    ny, nx = data.shape
    by, bx = -(-ny // BOX), -(-nx // BOX)
    pad = np.full((by * BOX, bx * BOX), np.nan, dtype=np.float32)
    pad[:ny, :nx] = np.where(good, data, np.nan)
    boxes = pad.reshape(by, BOX, bx, BOX).transpose(0, 2, 1, 3).reshape(by, bx, -1)[..., ::4]
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for _ in range(2):
            med = np.nanmedian(boxes, axis=2, keepdims=True)
            mad = np.nanmedian(np.abs(boxes - med), axis=2, keepdims=True)
            boxes = np.where(np.abs(boxes - med) < 3 * 1.4826 * mad, boxes, np.nan)
        grid = np.nanmedian(boxes, axis=2)
    grid = np.where(np.isfinite(grid), grid, np.nanmedian(grid))
    return _Smooth(_box_centers(ny), _box_centers(nx), median_filter(grid, size=3, mode='nearest'))


def _zeropoint(meta, mag_ref, shape):
    """Z(x, y) of dophot's model at zero color for an instrumental magnitude
    mag_ref, a _Smooth over the frame (pixels are 0-based, the model's
    X/Y_IMAGE 1-based)"""
    from pyrt.core.fotfit import fotfit
    terms, values = [], []
    for chunk in str(meta['RESPONSE']).split(','):
        if '=' not in chunk:
            continue
        term, value = chunk.split('=', 1)
        if term[0] == 'F' and term[1:].isdigit():   # flux weights: nothing at zero color
            continue
        try:
            values.append(float(value))
            terms.append(term)
        except ValueError:                          # FILTER=, SCHEMA=
            continue
    ffit = fotfit()
    ffit.fixterm(terms, values=values)

    cy, cx = _box_centers(shape[0]), _box_centers(shape[1])
    y, x = np.meshgrid(cy + 1, cx + 1, indexing='ij')
    x, y = x.ravel(), y.ravel()
    m = np.full(x.size, mag_ref)
    z = ffit.model(np.array(ffit.fitvalues),
                   (m, 0, (x - meta['CTRX']) / 1024, (y - meta['CTRY']) / 1024,
                    0, 0, 0, 0, 0, 0, 0, x, y, meta['AIRMASS'])) - m
    return _Smooth(cy, cx, z.reshape(cy.size, cx.size))


def _candidate_tiles(wcs, shape):
    """Cells the frame touches, and their neighbours (a tile reaches past its cell)"""
    import astropy.units as u
    hp = _healpix()
    ny, nx = shape
    y, x = np.meshgrid(np.linspace(0, ny - 1, 33), np.linspace(0, nx - 1, 33), indexing='ij')
    ra, dec = wcs.all_pix2world(x.ravel(), y.ravel(), 0)
    ok = np.isfinite(ra) & np.isfinite(dec)
    cells = np.unique(hp.lonlat_to_healpix(ra[ok] * u.deg, dec[ok] * u.deg))
    near = hp.neighbours(cells).ravel()
    return np.unique(np.concatenate([cells, near[near >= 0]]))


def _resample(tile_hdr, fwcs, flux, sigma_scale):
    """The frame on the tile: flux per tile pixel (bilinear, NaN wherever a bad
    or outside pixel takes part) and its noise; (y0, x0, flux, sigma) or None.
    flux is per frame pixel with bad pixels NaN, sigma_scale a _Smooth of its
    noise."""
    from astropy.wcs import WCS
    from scipy.ndimage import map_coordinates
    twcs = WCS(tile_hdr)
    n = tile_hdr['NAXIS1']
    ny, nx = flux.shape

    # where the frame lands on the tile: its outline, mapped through the sky
    t = np.linspace(0, 1, 65)
    ox = np.concatenate([t * (nx - 1), np.full_like(t, nx - 1), t[::-1] * (nx - 1), np.zeros_like(t)])
    oy = np.concatenate([np.zeros_like(t), t * (ny - 1), np.full_like(t, ny - 1), t[::-1] * (ny - 1)])
    ra, dec = fwcs.all_pix2world(ox, oy, 0)
    tx, ty = twcs.wcs_world2pix(ra, dec, 0)
    ok = np.isfinite(tx) & np.isfinite(ty)
    if not ok.any():
        return None
    x0, x1 = max(int(np.floor(tx[ok].min())) - 2, 0), min(int(np.ceil(tx[ok].max())) + 3, n)
    y0, y1 = max(int(np.floor(ty[ok].min())) - 2, 0), min(int(np.ceil(ty[ok].max())) + 3, n)
    if x0 >= x1 or y0 >= y1:
        return None

    # exact transformation on a coarse grid, interpolated in between
    gy = np.arange(y0, y1 + GRID, GRID, dtype=float)
    gx = np.arange(x0, x1 + GRID, GRID, dtype=float)
    cy, cx = np.meshgrid(gy, gx, indexing='ij')
    ra, dec = twcs.wcs_pix2world(cx.ravel(), cy.ravel(), 0)
    fx, fy = fwcs.all_world2pix(ra, dec, 0, quiet=True)
    fx, fy = fx.reshape(cx.shape), fy.reshape(cx.shape)
    if not ((fx >= -GRID) & (fx <= nx - 1 + GRID) & (fy >= -GRID) & (fy <= ny - 1 + GRID)).any():
        return None
    ty, tx = np.arange(y0, y1, dtype=float), np.arange(x0, x1, dtype=float)
    px, py = _Smooth(gy, gx, fx).grid(ty, tx), _Smooth(gy, gx, fy).grid(ty, tx)

    # frame pixels per tile pixel, and the noise per tile pixel, both smooth
    dxdx, dxdy = np.gradient(fx, GRID, axis=1), np.gradient(fx, GRID, axis=0)
    dydx, dydy = np.gradient(fy, GRID, axis=1), np.gradient(fy, GRID, axis=0)
    area = np.abs(dxdx * dydy - dxdy * dydx)
    sigma = _Smooth(gy, gx, area * sigma_scale.at(fy, fx)).grid(ty, tx)
    area = _Smooth(gy, gx, area).grid(ty, tx)

    f = map_coordinates(flux, [py, px], order=1, mode='constant', cval=np.nan, prefilter=False)
    return y0, x0, f * area, sigma


def _add_to_tile(path, hdr, frame, y0, x0, wf, w):
    """Add the frame's w*flux and w at (y0, x0) of the tile, under a lock"""
    from astropy.io import fits
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix('.lock'), 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            with fits.open(path) as hdul:
                frames = list(hdul['FRAMES'].data['NAME'])
                if frame in frames:
                    return False
                total = hdul[0].data.astype(np.float32)
                weight = hdul['WEIGHT'].data.astype(np.float32)
                hdr = hdul[0].header
        else:
            frames = []
            n = hdr['NAXIS1']
            total = np.zeros((n, n), dtype=np.float32)
            weight = np.zeros((n, n), dtype=np.float32)
        ny, nx = w.shape
        total[y0:y0 + ny, x0:x0 + nx] += wf
        weight[y0:y0 + ny, x0:x0 + nx] += w
        frames.append(frame)
        hdr['NFRAMES'] = len(frames)
        hdul = fits.HDUList([
            fits.PrimaryHDU(total, header=hdr),
            fits.ImageHDU(weight, name='WEIGHT'),
            fits.BinTableHDU.from_columns(
                [fits.Column(name='NAME', format='20A', array=np.array(frames))], name='FRAMES'),
        ])
        tmp = path.with_name(path.name + '.tmp')
        hdul.writeto(tmp, overwrite=True)
        os.replace(tmp, path)
    return True


def _read_ecsv(ecsv_path):
    """The ECSV's metadata and the median MAG_AUTO, without parsing the whole
    table (astropy takes seconds for the ~50k rows of a wide field)"""
    import pandas
    from astropy.io.misc import yaml
    header, nhead = [], 0
    with open(ecsv_path) as f:
        for line in f:
            if not line.startswith('#'):
                break
            nhead += 1
            if not line.startswith('# %ECSV'):
                header.append(line[2:])
    head = yaml.load(''.join(header))
    names = [c['name'] for c in head['datatype']]
    # the first non-comment line is the row of column names
    col = pandas.read_csv(ecsv_path, sep=' ', skiprows=nhead + 1, header=None,
                          usecols=[names.index('MAG_AUTO')]).iloc[:, 0]
    return dict(head.get('meta', {})), float(np.nanmedian(col.to_numpy(dtype=float)))


def add_frame(fits_path, ecsv_path, mapdir):
    """Add a calibrated frame (with its dophot ECSV) to the map; number of
    tiles it went to, or the reason it did not go"""
    from astropy.io import fits
    from scipy.ndimage import binary_dilation
    from pyrt.cli.phcat import get_saturation

    meta, mag_ref = _read_ecsv(str(ecsv_path))
    reason = frame_gate(meta)
    if reason:
        return reason
    band = frame_band(meta)
    frame = str(meta['FITSFILE'])[:19]

    data = fits.getdata(str(fits_path)).astype(np.float32)
    good = np.isfinite(data) & (data < 0.9 * get_saturation(str(fits_path)))
    good = ~binary_dilation(~good, iterations=2)

    ny, nx = data.shape
    rows, cols = np.arange(ny, dtype=float), np.arange(nx, dtype=float)
    data -= _background(data, good).grid(rows, cols)
    # cold pixels: nothing on the sky is that far below the background
    good &= data > -5 * float(meta['BGSIGMA'])
    zp = _zeropoint(meta, mag_ref, data.shape)
    data *= 10 ** (0.4 * (ZP - zp.grid(rows, cols)))
    data[~good] = np.nan
    sigma_scale = _Smooth(zp.gy, zp.gx, float(meta['BGSIGMA']) *
                          10 ** (0.4 * (ZP - zp.grid(zp.gy, zp.gx))))

    fwcs = frame_wcs(meta)
    added = 0
    for ipix in _candidate_tiles(fwcs, data.shape):
        hdr = tile_header(int(ipix))
        res = _resample(hdr, fwcs, data, sigma_scale)
        if res is None:
            continue
        y0, x0, f, s = res
        use = np.isfinite(f) & (s > 0)
        if not use.any():
            continue
        w = np.where(use, 1 / np.where(use, s, 1) ** 2, 0).astype(np.float32)
        if _add_to_tile(tile_path(mapdir, band, int(ipix)), hdr, frame, y0, x0,
                        np.where(use, w * f, 0).astype(np.float32), w):
            added += 1
    return added


# --- export --------------------------------------------------------------------

def export(mapdir, outdir, bands):
    """sum/weight of tiles changed since their last export; number written"""
    from astropy.io import fits
    written = 0
    for band in bands:
        for path in sorted((Path(mapdir) / band).glob('*/*.fits')):
            out = Path(outdir) / band / path.parent.name / path.name
            if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
                continue
            with fits.open(path) as hdul:
                total, weight = hdul[0].data, hdul['WEIGHT'].data
                hdr = hdul[0].header.copy()
            with np.errstate(all='ignore'):
                mean = np.where(weight > 0, total / weight, np.nan).astype(np.float32)
            hdr['BUNIT'] = ('flux', 'per pixel, m(AB) = 25 - 2.5 log10(flux)')
            hdr['FILTER'] = band
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + '.tmp')
            fits.PrimaryHDU(mean, header=hdr).writeto(tmp, overwrite=True)
            os.replace(tmp, out)
            written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description='Sky map tiles for Hipsgen',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('add', help='add calibrated frames (FITS with its dophot ECSV)')
    p.add_argument('--dir', required=True, help='map directory')
    p.add_argument('pairs', nargs='+', metavar='FITS ECSV')
    p = sub.add_parser('export', help='write sum/weight of the changed tiles for Hipsgen')
    p.add_argument('--dir', required=True, help='map directory')
    p.add_argument('--out', required=True, help='Hipsgen input directory')
    p.add_argument('--band', action='append', choices=sorted(BANDS.values()))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    if args.cmd == 'add':
        if len(args.pairs) % 2:
            sys.exit('give FITS ECSV pairs')
        for fits_path, ecsv_path in zip(args.pairs[::2], args.pairs[1::2]):
            logger.info(f"{Path(fits_path).name}: {add_frame(fits_path, ecsv_path, args.dir)}")
    else:
        n = export(args.dir, args.out, args.band or sorted(BANDS.values()))
        logger.info(f"{n} tiles exported")


if __name__ == '__main__':
    main()
