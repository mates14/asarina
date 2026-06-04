#!/usr/bin/env python3
"""
Residual diagnostics for the bgnoise background noise model.

Plots (pred - actual) / actual residuals against time and key observing
parameters to reveal systematic model errors such as slow trends from
dust accumulation, airmass bias, or moon-phase dependence.

Usage
-----
    asarina-observe-diagnose ~/phdb/stat/*.ecsv --camera C2
    asarina-observe-diagnose ~/phdb/stat/*.ecsv --camera C2 --output residuals.png
"""

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from asarina.observe.stat import read_stat
from asarina.observe.bg_predict import predict_background, moon_illumination, _DEFAULT_MODEL


def _jd_to_datetime(jd):
    unix = (np.asarray(jd) - 2440587.5) * 86400.0
    return pd.to_datetime(unix, unit='s', utc=True)


def _binned_median(x, y, n_bins=30):
    """Return (bin_centres, medians) with at least 5 points per bin."""
    edges = np.linspace(np.nanmin(x), np.nanmax(x), n_bins + 1)
    centres, medians = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi) & np.isfinite(y)
        if mask.sum() >= 5:
            centres.append((lo + hi) / 2)
            medians.append(np.median(y[mask]))
    return np.array(centres), np.array(medians)


def _linear_trend(x, y):
    """Return (slope, intercept) from a least-squares fit."""
    finite = np.isfinite(x) & np.isfinite(y)
    return np.polyfit(x[finite], y[finite], 1)


def compute_residuals(data: pd.DataFrame, model_file: str) -> pd.Series:
    moon_dist = data['moon_dist'].values if 'moon_dist' in data.columns else None
    sun_dist  = data['sun_dist'].values  if 'sun_dist'  in data.columns else None
    pred = predict_background(
        data['jd'].values, data['sun_alt'].values, data['moon_alt'].values,
        data['airmass'].values, data['filter'].values, data['zp_1s'].values,
        moon_dist=moon_dist, sun_dist=sun_dist,
        model_file=model_file,
    )
    return (pred - data['bgnoise_1s'].values) / data['bgnoise_1s'].values


def _scatter_with_trend(ax, x, resid, xlabel, n_bins=25, color='#4477AA'):
    finite = np.isfinite(x) & np.isfinite(resid)
    ax.scatter(x[finite], resid[finite] * 100, s=2, alpha=0.25,
               color=color, rasterized=True)
    cx, cy = _binned_median(x[finite], resid[finite] * 100, n_bins=n_bins)
    if len(cx) >= 2:
        ax.plot(cx, cy, color='#CC3311', lw=1.5, zorder=5)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Residual (%)')
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%+.0f%%'))


def plot_diagnostics(data: pd.DataFrame, model_file: str,
                     camera: str, output: str = None) -> None:
    resid = compute_residuals(data, model_file)
    dates = _jd_to_datetime(data['jd'].values)
    moon_illum = moon_illumination(data['jd'].values)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f'bgnoise model residuals — camera {camera}  '
                 f'({len(data):,} obs,  MAE {np.median(np.abs(resid))*100:.1f}%)',
                 fontsize=11)

    # --- Time trend (main diagnostic) ---
    ax = axes[0, 0]
    finite = np.isfinite(resid)
    ax.scatter(dates[finite], resid[finite] * 100, s=2, alpha=0.2,
               color='#4477AA', rasterized=True)

    # binned monthly median
    jd = data['jd'].values
    cx, cy = _binned_median(jd[finite], resid[finite] * 100, n_bins=40)
    cx_dates = _jd_to_datetime(cx)
    if len(cx) >= 2:
        ax.plot(cx_dates, cy, color='#CC3311', lw=1.5, zorder=5, label='monthly median')
        slope, intercept = _linear_trend(jd[finite], resid[finite] * 100)
        slope_per_month = slope * 30
        jd_range = np.array([jd[finite].min(), jd[finite].max()])
        ax.plot(_jd_to_datetime(jd_range), np.polyval([slope, intercept], jd_range),
                color='#EE7722', lw=1.2, ls='--', zorder=6,
                label=f'trend {slope_per_month:+.2f}%/month')
        ax.legend(fontsize=8, loc='upper left')

    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)
    ax.set_xlabel('Date (UTC)')
    ax.set_ylabel('Residual (%)')
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%+.0f%%'))
    ax.set_title('Time trend')

    # --- Moon illumination ---
    ax = axes[0, 1]
    _scatter_with_trend(ax, moon_illum, resid, 'Moon illumination')
    ax.set_title('Moon illumination')

    # --- Moon distance ---
    ax = axes[0, 2]
    moon_dist = data['moon_dist'].values if 'moon_dist' in data.columns else np.full(len(data), np.nan)
    _scatter_with_trend(ax, moon_dist, resid, 'Moon distance (deg)')
    ax.set_title('Moon distance')

    # --- Airmass ---
    ax = axes[1, 0]
    _scatter_with_trend(ax, data['airmass'].values, resid, 'Airmass')
    ax.set_title('Airmass')

    # --- Zeropoint (transparency proxy) ---
    ax = axes[1, 1]
    _scatter_with_trend(ax, data['zp_1s'].values, resid, 'ZP₁ₛ (mag)')
    ax.set_title('Zeropoint (transparency)')

    # --- Per-filter violin ---
    ax = axes[1, 2]
    filters = sorted(data['filter'].unique())
    parts = ax.violinplot(
        [resid[data['filter'] == f] * 100 for f in filters],
        positions=range(len(filters)),
        showmedians=True, showextrema=False,
    )
    for pc in parts['bodies']:
        pc.set_facecolor('#4477AA')
        pc.set_alpha(0.5)
    parts['cmedians'].set_color('#CC3311')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xticks(range(len(filters)))
    ax.set_xticklabels(filters, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Residual (%)')
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%+.0f%%'))
    ax.set_title('Per filter')

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Saved → {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Residual diagnostics for the bgnoise model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('stat_files', nargs='+', metavar='FILE',
                        help='Per-night stat ECSV files (e.g. ~/phdb/stat/*.ecsv)')
    parser.add_argument('--camera', metavar='NAME', default='C0',
                        help='Filter stat records to this camera')
    parser.add_argument('--model', metavar='FILE', default=_DEFAULT_MODEL,
                        help='Trained model .pkl file')
    parser.add_argument('--output', metavar='FILE', default=None,
                        help='Save figure to this path instead of displaying')
    args = parser.parse_args()

    frames = [read_stat(f) for f in args.stat_files]
    data = pd.concat(frames, ignore_index=True).sort_values('jd').reset_index(drop=True)

    if 'camera' in data.columns and data['camera'].str.len().gt(0).any():
        before = len(data)
        data = data[data['camera'] == args.camera].reset_index(drop=True)
        print(f"Camera {args.camera}: {len(data):,} of {before:,} records")
    else:
        print(f"Camera {args.camera}: no camera column — using all {len(data):,} records")

    data = data[
        (data['exptime'] > 0) & (data['bgnoise'] > 0) &
        (data['airmass'] > 0) & (data['jd'] > 2400000) &
        (data['zp_1s'] > 5) & (data['zp_1s'] < 30)
    ].reset_index(drop=True)

    if not args.output:
        matplotlib.use('TkAgg')

    plot_diagnostics(data, args.model, camera=args.camera, output=args.output)


if __name__ == '__main__':
    main()
