"""Every frame asarina sums carries its own COADDS (readouts in all), the keyword pyrt
reads first (pyrt.utils.camera_config.readouts); the series header it copies from carries
the whole series' COADDS, which must not survive."""
import types

import astropy.io.fits as fits
from astropy.wcs import WCS

from pyrt.utils.camera_config import readouts
from asarina.pipeline.kinphot import subframe_header
from asarina.pipeline.kinstep import mark_accumulated
from asarina.pipeline.framestep import bin_header


def raw(coadds=96, acc=1, mode='Kinetic Series'):
    h = fits.Header()
    h['ACQMODE'], h['ACCNUM'], h['KINNUM'] = mode, acc, 96
    if coadds:
        h['COADDS'] = coadds
    h['CTIME'], h['USEC'], h['EXPTIME'], h['FILTER'] = 1795000000, 0, 120.0, 'N'
    return h


def wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval, w.wcs.crpix, w.wcs.cdelt = [25.8, 39.1], [512, 512], [-3e-4, 3e-4]
    return w.to_header()


def test_kinphot_subframe():
    for coadds in (96, None):            # since 2026-11-03 / older (a mean then)
        h = subframe_header(raw(coadds), wcs(), None, 1795000000.5, 5.0, 8, 4, 'ref', 'raw.fits')
        assert readouts(h) == 4
    h = subframe_header(raw(192, acc=2), wcs(), None, 1795000000.5, 5.0, 8, 4, 'ref', 'raw.fits')
    assert readouts(h) == 8


def test_kinstep_bin():
    h = subframe_header(raw(), wcs(), None, 1795000000.5, 15.0, 0, 12, 'ref', 'raw.fits')
    mark_accumulated(h, 12)
    assert readouts(h) == 12 and h['ACCNUM'] == 12 and h['ACQMODE'] == 'Accumulate'


def frame(h, n):
    return types.SimpleNamespace(hdr=h, t0=1795000000.0 + 130 * n, exptime=120.0,
                                 stem=f'20261124-1059{n:02d}-RA', camera='C0')


def test_framestep_bin():
    skel = wcs()
    # 6 summed D50 kinetic series frames: 6 x 96 readouts (2026-11-24 ACCNUM=6 alone: no stars)
    fr = [frame(raw(), n) for n in range(6)]
    assert readouts(bin_header(skel, fr, 6, fr[0].hdr)) == 576
    # SBT/FRAM single frames: as before, k
    plain = fits.Header()
    plain['ACQMODE'], plain['ACCNUM'], plain['CTIME'], plain['EXPTIME'] = 'Single Scan', 1, 1795000000, 60.0
    fr = [frame(plain, n) for n in range(3)]
    assert readouts(bin_header(skel, fr, 3, fr[0].hdr)) == 3
    # D50 kinetic frames before 2026-11-03 (16-bit means, no COADDS): one readout each, as before
    fr = [frame(raw(None), n) for n in range(4)]
    assert readouts(bin_header(skel, fr, 4, fr[0].hdr)) == 4
