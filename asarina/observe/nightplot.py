#!/usr/bin/env python3
"""
Night conditions plot for telescope status webpage.

Produces a three-panel PNG showing atmospheric extinction (k_r),
sky surface brightness (mag/arcsec²), and PSF FWHM (arcsec) for one
observing night, with both actual measurements and model predictions.

Usage
-----
    # Current night from RTS2_STAT_DIR, save to night.png
    rtspy-observe-nightplot --output /var/www/telescope/night.png

    # Specific stat file
    rtspy-observe-nightplot 20260529.ecsv --output night.png

    # Display interactively (no --output)
    rtspy-observe-nightplot 20260529.ecsv
"""

import os
import sys
import argparse
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Filter presentation
# ---------------------------------------------------------------------------

FILTER_STYLE = {
    'Sloan_g': dict(color='#4477AA', label='g', zorder=3),
    'Sloan_r': dict(color='#228833', label='r', zorder=3),
    'Sloan_i': dict(color='#EE7722', label='i', zorder=3),
    'Sloan_z': dict(color='#CC3311', label='z', zorder=3),
    'N':       dict(color='#777777', label='N', zorder=3),
}
_FALLBACK_COLORS = ['#AA3377', '#BBBBBB', '#CCAA00']


def _filter_style(f):
    return FILTER_STYLE.get(f, dict(color=_FALLBACK_COLORS[hash(f) % 3], label=f, zorder=3))


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _jd_to_ut(jd):
    """Convert JD to UT hours within a 24-hour window centred on midnight."""
    # fractional day since noon: put 12:00 at left edge so night spans 12–36
    frac = (jd + 0.5) % 1.0   # 0 = midnight, 0.5 = noon
    return (frac + 0.5) % 1.0 * 24.0  # hours: noon=12, midnight=0/24


def _jd_to_nighthour(jd):
    """
    Return hours from noon UTC (12 = noon, 24 = midnight, 36 = next noon).

    JD integers fall at noon UTC, so (jd % 1.0) is the fractional day since
    the previous noon.  Multiplying by 24 and adding 12 maps noon→12,
    midnight→24, and the following morning hours to 24–36.
    """
    return (jd % 1.0) * 24.0 + 12.0


def _jd_night_label(jd):
    """Return 'YYYY-MM-DD' of the evening (afternoon) date."""
    dt = datetime.fromtimestamp((_jd_to_unix(jd) - 43200), tz=timezone.utc)
    return dt.strftime('%Y-%m-%d')


def _jd_to_unix(jd):
    return (jd - 2440587.5) * 86400.0


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def _k_r(zp_1s, airmass, filt, filter_params):
    """Derive common extinction coefficient from one measurement."""
    p = filter_params.get(filt)
    if p is None or airmass <= 0:
        return np.nan
    return (p['Z0'] - zp_1s) / airmass - p['beta']


def _sky_mag(zp_1s, bgnoise_1s, gain, plate_scale_arcsec):
    """Sky surface brightness in mag/arcsec².  Higher = darker sky."""
    if bgnoise_1s <= 0 or plate_scale_arcsec is None:
        return np.nan
    sky_adu_per_s = bgnoise_1s ** 2 * gain          # ADU/s/pixel via Poisson
    mag_per_pix   = zp_1s - 2.5 * np.log10(sky_adu_per_s)
    return mag_per_pix + 2.5 * np.log10(plate_scale_arcsec ** 2)


# Reference conditions for zenith sky normalisation
_REF_MOON_DIST = 90.0   # deg — "standard" distance from moon
_REF_SUN_DIST  = 90.0   # deg


def _rolling_smooth(x, y, w_half=3):
    """Simple box-smoothed curve; returns (x_sorted, y_smooth)."""
    idx  = np.argsort(x)
    xs   = np.array(x)[idx]
    ys   = np.array(y)[idx]
    out  = np.full_like(ys, np.nan)
    for i in range(len(ys)):
        lo, hi = max(0, i - w_half), min(len(ys), i + w_half + 1)
        vals   = ys[lo:hi]
        out[i] = np.nanmedian(vals)
    return xs, out


# ---------------------------------------------------------------------------
# Core plot function
# ---------------------------------------------------------------------------

def make_night_plot(
    df: pd.DataFrame,
    cfg=None,
    output: Optional[str] = None,
    title: Optional[str] = None,
    predict: bool = True,
    figsize=(13, 8),
) -> 'matplotlib.figure.Figure':
    """
    Generate a three-panel night conditions plot.

    Parameters
    ----------
    df      : stat DataFrame for one night (from read_stat)
    cfg     : CameraConfig (None → built-in defaults)
    output  : save to this path (None → return figure without saving)
    title   : plot title (auto-derived from data if None)
    predict : overlay model predictions (ML bgnoise prior)
    figsize : figure size in inches

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use('Agg' if output else matplotlib.get_backend())
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D

    if cfg is None:
        from asarina.observe.camera import CameraConfig
        cfg = CameraConfig.load(None)

    gain        = cfg.gain
    plate_scale = getattr(cfg, 'plate_scale', None)
    fp          = cfg.filter_params
    sl          = cfg.sanity_limits

    # ------------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------------
    df = df.copy()
    df = df[df['exptime'] > 0].reset_index(drop=True)

    # Sanity-filter zp_1s
    def _sane(row):
        lim = sl.get(row['filter'], {})
        return lim.get('zp_min', -99) <= row['zp_1s'] <= lim.get('zp_max', 99)

    sane_mask = df.apply(_sane, axis=1)

    df['t']       = df['jd'].apply(_jd_to_nighthour)   # hours since noon UTC
    df['k_r']     = [_k_r(r['zp_1s'], r['airmass'], r['filter'], fp)
                      if sane_mask[i] else np.nan
                      for i, r in df.iterrows()]
    df['sky_mag'] = [_sky_mag(r['zp_1s'], r['bgnoise_1s'], gain, plate_scale)
                      if sane_mask[i] else np.nan
                      for i, r in df.iterrows()]
    df['fwhm_as'] = df['fwhm'] * plate_scale if plate_scale else df['fwhm']

    # Zenith r-band sky normalisation (requires ML model)
    # For each observation:
    #   bg_norm = bgnoise_1s * (model_ref / model_actual)
    # where model_ref = predict(airmass=1, Sloan_r, moon_dist=90°, zp_r_zenith=Z0_r−k_r)
    # This removes airmass, filter, and moon-distance geometry, leaving intrinsic sky state.
    if predict and not plate_scale:
        print('WARNING: plate_scale not set in telescope config — sky normalisation disabled',
              file=sys.stderr)
    if predict and plate_scale:
        try:
            from asarina.observe.bg_predict import predict_background
            Z0_r = fp.get('Sloan_r', {}).get('Z0', 21.94)

            bg_actual_list = []
            bg_ref_list    = []
            sky_norm_list  = []
            sky_pred_list  = []

            for i, r in df.iterrows():
                if not sane_mask[i] or r['bgnoise_1s'] <= 0 or np.isnan(r['k_r']):
                    bg_actual_list.append(np.nan)
                    bg_ref_list.append(np.nan)
                    sky_norm_list.append(np.nan)
                    sky_pred_list.append(np.nan)
                    continue

                zp_r_zen = Z0_r - r['k_r']   # r-band zp at airmass=1

                bg_actual = predict_background(
                    r['jd'], r['sun_alt'], r['moon_alt'],
                    r['airmass'], r['filter'], r['zp_1s'],
                    moon_dist=r.get('moon_dist'), sun_dist=r.get('sun_dist'),
                    model_file=cfg.model_file,
                )
                bg_ref = predict_background(
                    r['jd'], r['sun_alt'], r['moon_alt'],
                    1.0, 'Sloan_r', zp_r_zen,
                    moon_dist=_REF_MOON_DIST, sun_dist=_REF_SUN_DIST,
                    model_file=cfg.model_file,
                )

                if bg_actual > 0 and bg_ref > 0:
                    bg_norm = r['bgnoise_1s'] * (bg_ref / bg_actual)
                    sky_norm_list.append(_sky_mag(zp_r_zen, bg_norm, gain, plate_scale))
                    sky_pred_list.append(_sky_mag(zp_r_zen, bg_ref,  gain, plate_scale))
                else:
                    sky_norm_list.append(np.nan)
                    sky_pred_list.append(np.nan)

                bg_actual_list.append(bg_actual)
                bg_ref_list.append(bg_ref)

            df['sky_r_zen']      = sky_norm_list
            df['sky_r_zen_pred'] = sky_pred_list
        except Exception as e:
            import traceback
            print(f'WARNING: sky normalisation/prediction disabled: {e}', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            predict = False

    filters = [f for f in ['Sloan_g', 'Sloan_r', 'Sloan_i', 'Sloan_z', 'N']
               if f in df['filter'].values]
    extra   = [f for f in df['filter'].unique() if f not in filters]
    filters += extra

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        3, 1, figsize=figsize, sharex=True,
        gridspec_kw={'height_ratios': [2, 2, 1], 'hspace': 0.08},
    )
    ax_kr, ax_sky, ax_fwhm = axes

    t_all = df['t'].dropna()
    t_lo  = max(12.0, t_all.min() - 0.3)
    t_hi  = min(36.0, t_all.max() + 0.3)

    # Shade twilight (sun_alt > -12°) in light yellow
    for ax in axes:
        twilight = df[df['sun_alt'] > -12.0]
        if len(twilight):
            for t_seg in _contiguous_segments(twilight['t'].values, gap=0.1):
                ax.axvspan(t_seg[0], t_seg[-1], color='#FFFACC', alpha=0.6, zorder=0)
        ax.set_facecolor('#F8F8F8')
        ax.grid(True, color='white', linewidth=1.0, zorder=1)
        for spine in ax.spines.values():
            spine.set_color('#CCCCCC')

    # ------------------------------------------------------------------
    # Panel 1: k_r (atmospheric extinction)
    # ------------------------------------------------------------------
    legend_handles = []
    for filt in filters:
        sub = df[(df['filter'] == filt) & df['k_r'].notna()]
        if sub.empty:
            continue
        st = _filter_style(filt)
        ax_kr.scatter(sub['t'], sub['k_r'], s=18, color=st['color'],
                      alpha=0.8, zorder=st['zorder'])
        legend_handles.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=st['color'],
                   markersize=7, label=st['label'])
        )

    # Single smooth trend across all filters — the atmosphere's transparency
    all_sane = df[df['k_r'].notna()]
    if len(all_sane) >= 6:
        xs, ys = _rolling_smooth(all_sane['t'].values, all_sane['k_r'].values, w_half=4)
        ax_kr.plot(xs, ys, color='#222222', lw=2.0, alpha=0.7, zorder=4)

    ax_kr.set_ylabel('Extinction  k_r  [mag/airmass]', fontsize=10)
    ax_kr.invert_yaxis()    # lower k_r (clearer sky) at top
    # Auto-range: clip at 5th/99th percentile with a small margin
    kr_vals = all_sane['k_r'].dropna()
    if len(kr_vals) > 5:
        lo, hi = kr_vals.quantile(0.01), kr_vals.quantile(0.99)
        margin  = max(0.05, (hi - lo) * 0.15)
        ax_kr.set_ylim(hi + margin, lo - margin)   # inverted
    ax_kr.legend(handles=legend_handles, ncol=len(filters),
                 loc='upper right', fontsize=9, framealpha=0.9)
    ax_kr.text(0.01, 0.97, '← more transparent', transform=ax_kr.transAxes,
               fontsize=8, va='top', color='#666666')

    # ------------------------------------------------------------------
    # Panel 2: Sky surface brightness — normalised to zenith r-band at 90° from Moon
    # ------------------------------------------------------------------
    use_normalised = predict and plate_scale and 'sky_r_zen' in df.columns

    if use_normalised:
        sky_col  = 'sky_r_zen'
        sky_label = 'Sky brightness  r-band zenith  [mag/arcsec²]'
    elif plate_scale:
        sky_col  = 'sky_mag'
        sky_label = 'Sky brightness  [mag/arcsec²]'
    else:
        sky_col  = 'bgnoise_1s'
        sky_label = 'Sky noise  bgnoise_1s  [ADU]'

    for filt in filters:
        sub = df[(df['filter'] == filt) & df[sky_col].notna()]
        if sub.empty:
            continue
        st = _filter_style(filt)
        ax_sky.scatter(sub['t'], sub[sky_col], s=18, color=st['color'],
                       alpha=0.8, zorder=st['zorder'])

    # Prediction: single curve at reference conditions (same for all filters)
    if use_normalised:
        pred_sub = df[df['sky_r_zen_pred'].notna()]
        if not pred_sub.empty:
            ax_sky.scatter(pred_sub['t'], pred_sub['sky_r_zen_pred'], s=8,
                           color='#888888', alpha=0.45, marker='x', zorder=2)
            xs, ys = _rolling_smooth(pred_sub['t'].values,
                                     pred_sub['sky_r_zen_pred'].values, w_half=4)
            ax_sky.plot(xs, ys, color='#222222', lw=1.5, alpha=0.6, zorder=4)

    ax_sky.set_ylabel(sky_label, fontsize=10)
    if plate_scale:
        ax_sky.invert_yaxis()
        ax_sky.text(0.01, 0.97, '← brighter sky', transform=ax_sky.transAxes,
                    fontsize=8, va='top', color='#666666')
    # Clip y-axis to sane percentile range
    sky_vals = df[sky_col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sky_vals) > 5:
        lo, hi = sky_vals.quantile(0.01), sky_vals.quantile(0.99)
        margin  = max(0.3, (hi - lo) * 0.15)
        if plate_scale:
            ax_sky.set_ylim(hi + margin, lo - margin)   # inverted
        else:
            ax_sky.set_ylim(lo - margin, hi + margin)
    if use_normalised:
        ax_sky.scatter([], [], s=8, color='#888888', marker='x',
                       label='ML prediction', alpha=0.5)
        ax_sky.scatter([], [], s=18, color='#888888',
                       label='measured (normalised)', alpha=0.8)
        ax_sky.legend(fontsize=8, loc='upper right', framealpha=0.9)
        ax_sky.text(0.01, 0.03, 'normalised to airmass=1, 90° from Moon',
                    transform=ax_sky.transAxes, fontsize=7, va='bottom', color='#888888')

    # ------------------------------------------------------------------
    # Panel 3: FWHM / seeing
    # ------------------------------------------------------------------
    fwhm_label = 'FWHM  [arcsec]' if plate_scale else 'FWHM  [pixels]'
    for filt in filters:
        sub = df[(df['filter'] == filt) & df['fwhm_as'].notna()]
        if sub.empty:
            continue
        st = _filter_style(filt)
        ax_fwhm.scatter(sub['t'], sub['fwhm_as'], s=14, color=st['color'],
                        alpha=0.8, zorder=st['zorder'])

    ax_fwhm.set_ylabel(fwhm_label, fontsize=10)
    ax_fwhm.set_xlabel('UT  [hours]', fontsize=10)

    # X-axis ticks: every hour
    tick_h = np.arange(np.ceil(t_lo), np.floor(t_hi) + 1)
    ax_fwhm.set_xticks(tick_h)
    ax_fwhm.set_xticklabels([f'{int(h % 24):02d}:00' for h in tick_h], fontsize=9)
    ax_fwhm.set_xlim(t_lo, t_hi)

    # ------------------------------------------------------------------
    # Title and finish
    # ------------------------------------------------------------------
    if title is None:
        night_label = _jd_night_label(df['jd'].median())
        tel = 'D50/C0'
        title = f'{tel}  —  night {night_label}'

    n_obs   = len(df)
    n_sane  = int(sane_mask.sum())
    updated = datetime.utcnow().strftime('%H:%M UTC')
    fig.suptitle(f'{title}     ({n_sane}/{n_obs} obs)     updated {updated}',
                 fontsize=12, y=0.995)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=120, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Helper: contiguous time segments (for twilight shading)
# ---------------------------------------------------------------------------

def _contiguous_segments(t_arr, gap=0.3):
    """Split sorted time array into contiguous groups (no gap > `gap` hours)."""
    if len(t_arr) == 0:
        return []
    t_sorted = np.sort(t_arr)
    breaks   = np.where(np.diff(t_sorted) > gap)[0] + 1
    return np.split(t_sorted, breaks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Night conditions plot for telescope status webpage',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('stat_file', nargs='?', default=None,
                        help='Stat ECSV file or directory (default: current night '
                             'from RTS2_STAT_DIR)')
    parser.add_argument('--output', '-o', default=None, metavar='FILE',
                        help='Save plot to this file (PNG/PDF/SVG). '
                             'If omitted, display interactively.')
    parser.add_argument('--camera', default=None, metavar='NAME',
                        help='Camera section in /etc/asarina/config (default: $ASARINA_CAMERA or C0)')
    parser.add_argument('--title', default=None,
                        help='Override plot title')
    parser.add_argument('--no-predict', action='store_true',
                        help='Skip ML background predictions (faster)')
    parser.add_argument('--night', default=None, metavar='YYYYMMDD',
                        help='Night to plot (default: current night)')
    args = parser.parse_args()

    from asarina.observe.stat import read_stat, night_from_jd, _STAT_DIR
    from asarina.observe.camera import CameraConfig

    cfg = CameraConfig.load(args.camera)

    # Resolve stat source
    if args.stat_file:
        source = args.stat_file
    else:
        stat_dir = _STAT_DIR
        if args.night:
            source = str(Path(stat_dir) / f'{args.night}.ecsv')
        else:
            jd_now  = 2440587.5 + _time.time() / 86400.0
            night   = night_from_jd(jd_now)
            source  = str(Path(stat_dir) / f'{night}.ecsv')

    if not Path(source).exists():
        print(f'ERROR: {source} not found', file=sys.stderr)
        sys.exit(1)

    df = read_stat(source)
    if df.empty:
        print(f'ERROR: no data in {source}', file=sys.stderr)
        sys.exit(1)

    print(f'Plotting {len(df)} observations from {source}')

    fig = make_night_plot(
        df, cfg,
        output=args.output,
        title=args.title,
        predict=not args.no_predict,
    )

    if args.output:
        print(f'Saved → {args.output}')
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == '__main__':
    main()
