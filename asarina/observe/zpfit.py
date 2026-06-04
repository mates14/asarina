#!/usr/bin/env python3
"""
Two-stage robust zeropoint fitter.

Stage 1: Fit g, r, i simultaneously (Rayleigh + aerosol; no H2O) to
         determine Zg0, Zr0, Zi0, β_g, β_i via sigma-clipped least squares
         on quasi-simultaneous cross-filter pairs.

Stage 2: Use the k_r(t) time series from Stage 1 to fit z-band (Z0_z, β_z)
         and N-band (Z0_N, β_N) independently, absorbing H2O variability
         into the scatter rather than letting it corrupt the gri fit.

Usage (CLI)
-----------
    rtspy-observe-zpfit --stat stat.txt
    rtspy-observe-zpfit --stat stat.txt --config telescope.yaml --update-config
    rtspy-observe-zpfit --stat stat.txt --output fitted.txt

Usage (library)
---------------
    from asarina.observe.zpfit import fit_zeropoints
    result = fit_zeropoints('stat.txt', sanity_limits=cfg.sanity_limits)
    print(result['filter_params'])   # → update cfg.filter_params

Integration
-----------
    After a fit, pass the returned filter_params into CameraConfig and
    re-run rtspy-observe-train to retrain the background noise model with
    the updated ZP calibration.
"""

import argparse
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from asarina.observe.zp_predict import SANITY_LIMITS as _DEFAULT_SANITY

# ---------------------------------------------------------------------------
# Default fitting parameters
# ---------------------------------------------------------------------------
MAX_TIME_DIFF   = 0.01    # days (~14 min) — atmospheric stability window
MAX_AIRMASS_DIFF = 0.10   # same atmospheric column (same pointing)
SIGMA_THRESHOLD = 3.5     # sigma-clipping threshold for Stage 1
MAX_ITERATIONS  = 3       # usually converges in 2-3 iterations


# ---------------------------------------------------------------------------
# Core fitter
# ---------------------------------------------------------------------------

def fit_zeropoints(
    stat_file: str,
    sanity_limits: Optional[Dict] = None,
    max_time_diff: float  = MAX_TIME_DIFF,
    max_airmass_diff: float = MAX_AIRMASS_DIFF,
    sigma_threshold: float = SIGMA_THRESHOLD,
    max_iterations: int   = MAX_ITERATIONS,
    verbose: bool = True,
) -> Dict:
    """
    Two-stage robust fit of zeropoint constants from a stat.txt file.

    Parameters
    ----------
    stat_file        : path to pipeline stat.txt
    sanity_limits    : per-filter {zp_min, zp_max} pre-screening bounds;
                       defaults to SANITY_LIMITS from zp_predict.py
    max_time_diff    : max JD separation for quasi-simultaneous pairs [days]
    max_airmass_diff : max airmass separation for pairs
    sigma_threshold  : sigma-clipping threshold for Stage 1 (g,r,i)
    max_iterations   : maximum sigma-clipping iterations
    verbose          : print progress and results

    Returns
    -------
    dict with keys:
        'filter_params'  : {filter → {'Z0': float, 'beta': float}}
                           (None for filters with insufficient data)
        'rms'            : Stage 1 RMS residual [mag]
        'n_pairs'        : {'gr': used/total, 'ir': ..., 'zr': ..., 'Nr': ...}
        'n_obs'          : {'before': total, 'after': after sanity filter}
    """
    limits = sanity_limits if sanity_limits is not None else _DEFAULT_SANITY

    # ------------------------------------------------------------------
    # Load and sanity-filter
    # ------------------------------------------------------------------
    from asarina.observe.stat import read_stat
    data = read_stat(stat_file)
    if 'exposure' in data.columns and 'exptime' not in data.columns:
        data = data.rename(columns={'exposure': 'exptime'})
        data['zp_1s'] = data['zeropoint'] - 2.5 * np.log10(data['exptime'].clip(lower=1e-3))
    data = data[data['exptime'] > 0].copy()
    data['zp_norm'] = data['zp_1s']

    n_before = len(data)
    if verbose:
        print("=" * 70)
        print("PRODUCTION ZEROPOINT FITTING - TWO-STAGE ROBUST METHOD")
        print("=" * 70)
        print(f"\nLoaded {n_before:,} observations from {stat_file}")
        print("\nSANITY FILTERING:")

    n_rejected_per_filter = {}
    for filt, lim in limits.items():
        filt_mask = data['filter'] == filt
        n_filt = filt_mask.sum()
        if n_filt == 0:
            continue
        bad = filt_mask & (
            (data['zp_norm'] < lim['zp_min']) | (data['zp_norm'] > lim['zp_max'])
        )
        n_bad = bad.sum()
        n_rejected_per_filter[filt] = n_bad
        if verbose and n_bad > 0:
            print(f"  {filt}: {n_bad}/{n_filt} rejected "
                  f"(range [{lim['zp_min']:.1f}, {lim['zp_max']:.1f}])")
        data = data[~bad]

    n_after = len(data)
    n_total_rej = n_before - n_after
    if verbose:
        print(f"\nTotal rejected: {n_total_rej}/{n_before} ({100*n_total_rej/n_before:.1f}%)")
        print(f"Remaining: {n_after:,} observations\n")

    # Split by filter
    fdata = {f: data[data['filter'] == f].copy().reset_index(drop=True)
             for f in ['Sloan_g', 'Sloan_r', 'Sloan_i', 'Sloan_z', 'N']}

    if verbose:
        counts = {f: len(fdata[f]) for f in fdata}
        print("Observations after filtering: " +
              ", ".join(f"{f.replace('Sloan_','').replace('N','N')}={counts[f]}"
                        for f in fdata))
        print()

    # ------------------------------------------------------------------
    # Pair builder
    # ------------------------------------------------------------------
    def _create_pairs(d1, d2):
        if len(d1) == 0 or len(d2) == 0:
            return pd.DataFrame()
        jd2 = d2['jd'].values
        X2  = d2['airmass'].values
        coords2 = np.column_stack([jd2 / max_time_diff, X2 / max_airmass_diff])
        tree = KDTree(coords2)
        pairs = []
        for i, row in d1.iterrows():
            q = [row['jd'] / max_time_diff, row['airmass'] / max_airmass_diff]
            for j in tree.query_ball_point(q, r=np.sqrt(2)):
                if (abs(row['jd'] - jd2[j]) <= max_time_diff and
                        abs(row['airmass'] - X2[j]) <= max_airmass_diff):
                    pairs.append({
                        'jd':  0.5 * (row['jd'] + jd2[j]),
                        'X1':  row['airmass'],  'zp1': row['zp_norm'],
                        'X2':  X2[j],           'zp2': d2['zp_norm'].iloc[j],
                    })
        return pd.DataFrame(pairs)

    # ------------------------------------------------------------------
    # Stage 1: fit g, r, i
    # ------------------------------------------------------------------
    if verbose:
        print("=" * 70)
        print("STAGE 1: FITTING CORE BANDS (g, r, i)")
        print("=" * 70)

    gr_pairs = _create_pairs(fdata['Sloan_g'], fdata['Sloan_r'])
    ir_pairs = _create_pairs(fdata['Sloan_i'], fdata['Sloan_r'])

    if len(gr_pairs) == 0 and len(ir_pairs) == 0:
        raise RuntimeError("No g-r or i-r pairs found — cannot fit Stage 1.")

    n_gr0, n_ir0 = len(gr_pairs), len(ir_pairs)
    if verbose:
        print(f"\nPairs: g-r={n_gr0}, i-r={n_ir0}")
        print(f"Fitting with {sigma_threshold}σ clipping (max {max_iterations} iter)...\n")

    gr_pairs['valid'] = True
    ir_pairs['valid'] = True

    Zg0 = Zr0 = Zi0 = beta_g = beta_i = None
    rms = np.nan
    n_gr_used = n_ir_used = 0

    for it in range(max_iterations):
        gr_v = gr_pairs[gr_pairs['valid']]
        ir_v = ir_pairs[ir_pairs['valid']]
        n_gr_v, n_ir_v = len(gr_v), len(ir_v)
        n_v = n_gr_v + n_ir_v

        A = np.zeros((n_v, 5))
        b = np.zeros(n_v)

        # g-r: Xr*Zg0 - Xg*Zr0 - β_g*Xg*Xr = Xr*zp_g - Xg*zp_r
        A[:n_gr_v, 0] =  gr_v['X2'].values
        A[:n_gr_v, 1] = -gr_v['X1'].values
        A[:n_gr_v, 3] = -gr_v['X1'].values * gr_v['X2'].values
        b[:n_gr_v]    = (gr_v['X2'].values * gr_v['zp1'].values
                         - gr_v['X1'].values * gr_v['zp2'].values)

        # i-r: Xr*Zi0 - Xi*Zr0 - β_i*Xi*Xr = Xr*zp_i - Xi*zp_r
        A[n_gr_v:, 1] = -ir_v['X1'].values
        A[n_gr_v:, 2] =  ir_v['X2'].values
        A[n_gr_v:, 4] = -ir_v['X1'].values * ir_v['X2'].values
        b[n_gr_v:]    = (ir_v['X2'].values * ir_v['zp1'].values
                         - ir_v['X1'].values * ir_v['zp2'].values)

        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        Zg0, Zr0, Zi0, beta_g, beta_i = params

        resid = b - A @ params
        rms = np.sqrt(np.mean(resid**2))
        thresh = sigma_threshold * rms
        outliers = np.abs(resid) > thresh
        n_out = outliers.sum()

        if verbose:
            print(f"  Iter {it+1}: {n_v} pairs, RMS={rms:.6f}, outliers={n_out}")

        if n_out == 0:
            if verbose:
                print("  Converged.")
            break

        gr_pairs.loc[gr_pairs['valid'], 'valid'] = ~outliers[:n_gr_v]
        ir_pairs.loc[ir_pairs['valid'], 'valid'] = ~outliers[n_gr_v:]

    n_gr_used = gr_pairs['valid'].sum()
    n_ir_used = ir_pairs['valid'].sum()

    if verbose:
        print(f"\nFITTED (g,r,i):")
        print(f"  Zg0 = {Zg0:.6f} mag")
        print(f"  Zr0 = {Zr0:.6f} mag")
        print(f"  Zi0 = {Zi0:.6f} mag")
        print(f"  β_g = {beta_g:.6f} mag/airmass")
        print(f"  β_i = {beta_i:.6f} mag/airmass")
        print(f"  Pairs used: g-r={n_gr_used}/{n_gr0}  i-r={n_ir_used}/{n_ir0}")
        print(f"  Final RMS: {rms:.6f}\n")

    # ------------------------------------------------------------------
    # Stage 2: k_r time series → fit z and N
    # ------------------------------------------------------------------
    gr_v = gr_pairs[gr_pairs['valid']]
    ir_v = ir_pairs[ir_pairs['valid']]
    gr_v = gr_v.copy(); gr_v['k_r'] = (Zr0 - gr_v['zp2']) / gr_v['X2']
    ir_v = ir_v.copy(); ir_v['k_r'] = (Zr0 - ir_v['zp2']) / ir_v['X2']

    all_kr = pd.concat([
        pd.DataFrame({'jd': gr_v['jd'], 'k_r': gr_v['k_r']}),
        pd.DataFrame({'jd': ir_v['jd'], 'k_r': ir_v['k_r']}),
    ]).sort_values('jd').reset_index(drop=True)

    if verbose:
        print("=" * 70)
        print("STAGE 2: FITTING z AND N")
        print("=" * 70)
        print(f"\n  k_r from g-r: {gr_v['k_r'].median():.4f} ± {gr_v['k_r'].std():.4f} mag/airmass")
        print(f"  k_r from i-r: {ir_v['k_r'].median():.4f} ± {ir_v['k_r'].std():.4f} mag/airmass")

    def _fit_secondary(band_data, band_name):
        pairs = _create_pairs(band_data, fdata['Sloan_r'])
        n_pairs = len(pairs)
        if n_pairs == 0:
            if verbose:
                print(f"\n  {band_name}: no pairs found")
            return None, None, 0

        k_r_interp = np.interp(pairs['jd'].values,
                                all_kr['jd'].values, all_kr['k_r'].values)
        pairs = pairs.copy()
        pairs['k_r_est'] = k_r_interp

        A = np.column_stack([np.ones(n_pairs), -pairs['X1'].values])
        b = pairs['zp1'].values + pairs['k_r_est'].values * pairs['X1'].values
        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        Z0, beta = params

        if verbose:
            k_z_fit = pairs['k_r_est'] + beta
            k_z_obs = (Z0 - pairs['zp1']) / pairs['X1']
            k_H2O   = k_z_obs - k_z_fit
            print(f"\n  {band_name}:  Z0={Z0:.6f}  β={beta:.6f}  "
                  f"scatter(H2O)={k_H2O.std():.4f} mag/airmass  ({n_pairs} pairs)")

        return Z0, beta, n_pairs

    Zz0, beta_z, n_zr = _fit_secondary(fdata['Sloan_z'], 'z')
    Zn0, beta_N, n_Nr = _fit_secondary(fdata['N'], 'N')

    # ------------------------------------------------------------------
    # Assemble results
    # ------------------------------------------------------------------
    filter_params = {
        'Sloan_r': {'Z0': float(Zr0), 'beta': 0.0},
        'Sloan_g': {'Z0': float(Zg0), 'beta': float(beta_g)},
        'Sloan_i': {'Z0': float(Zi0), 'beta': float(beta_i)},
    }
    if Zz0 is not None:
        filter_params['Sloan_z'] = {'Z0': float(Zz0), 'beta': float(beta_z)}
    if Zn0 is not None:
        filter_params['N'] = {'Z0': float(Zn0), 'beta': float(beta_N)}

    stats = {
        'n_obs_before': n_before,
        'n_obs_after':  n_after,
        'rms':          float(rms),
        'n_pairs': {
            'gr': (int(n_gr_used), int(n_gr0)),
            'ir': (int(n_ir_used), int(n_ir0)),
            'zr': (int(n_zr), int(n_zr)),
            'Nr': (int(n_Nr), int(n_Nr)),
        },
        'jd_range': (float(data['jd'].min()), float(data['jd'].max())),
    }

    if verbose:
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        for f, p in filter_params.items():
            print(f"  {f:<12}  Z0={p['Z0']:.6f}  beta={p['beta']:+.6f}")

    return {'filter_params': filter_params, **stats}


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(result: Dict, path: str) -> None:
    """Write a human-readable fit report to a text file."""
    fp = result['filter_params']
    s  = result

    lines = [
        "=" * 70,
        "PRODUCTION ZEROPOINT FIT - TWO-STAGE ROBUST METHOD",
        "=" * 70,
        "",
        f"Time span: JD {s['jd_range'][0]:.4f} – {s['jd_range'][1]:.4f}",
        f"Observations: {s['n_obs_before']:,} → {s['n_obs_after']:,} after sanity filter",
        f"Stage 1 RMS: {s['rms']:.6f} mag",
        "",
        "FITTED PARAMETERS:",
    ]
    for f, p in fp.items():
        lines.append(f"  {f:<12}  Z0={p['Z0']:.6f}  beta={p['beta']:+.6f}")
    lines += [
        "",
        "PAIRS USED:",
        f"  g-r: {s['n_pairs']['gr'][0]}/{s['n_pairs']['gr'][1]}",
        f"  i-r: {s['n_pairs']['ir'][0]}/{s['n_pairs']['ir'][1]}",
        f"  z-r: {s['n_pairs']['zr'][0]}",
        f"  N-r: {s['n_pairs']['Nr'][0]}",
        "",
        "NOTES:",
        "  Z0 encodes instrument + absolute calibration (drifts with instrument changes).",
        "  beta encodes per-filter atmospheric extinction offset (stable; atmosphere only).",
        "  z-band scatter includes unmodelled H2O variability.",
        "  N-band accuracy limited by variable effective wavelength.",
    ]

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Two-stage robust zeropoint fitter — outputs filter_params for telescope config',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--stat', required=True, metavar='FILE',
                        help='Path to pipeline stat.txt')
    parser.add_argument('--camera', default=None, metavar='NAME',
                        help='Camera section in /etc/asarina/config (default: $ASARINA_CAMERA or C0)')
    parser.add_argument('--output', default=None, metavar='FILE',
                        help='Save text report to this file (default: no file)')
    parser.add_argument('--max-time-diff', type=float, default=MAX_TIME_DIFF,
                        metavar='DAYS', help='Max JD separation for quasi-simultaneous pairs')
    parser.add_argument('--max-airmass-diff', type=float, default=MAX_AIRMASS_DIFF,
                        metavar='DX', help='Max airmass separation for pairs')
    parser.add_argument('--sigma', type=float, default=SIGMA_THRESHOLD,
                        metavar='SIGMA', help='Sigma-clipping threshold for Stage 1')
    args = parser.parse_args()

    from asarina.observe.camera import CameraConfig
    cfg = CameraConfig.load(args.camera)

    result = fit_zeropoints(
        args.stat,
        sanity_limits=cfg.sanity_limits,
        max_time_diff=args.max_time_diff,
        max_airmass_diff=args.max_airmass_diff,
        sigma_threshold=args.sigma,
        verbose=True,
    )

    if args.output:
        write_report(result, args.output)
        print(f"\nReport written to {args.output}")

    print("\nFitted filter_params — update FILTER_PARAMS in asarina/observe/zp_predict.py:")
    for filt, p in result['filter_params'].items():
        print(f"  '{filt}': {{'Z0': {p['Z0']:.6f}, 'beta': {p['beta']:+.6f}}},")
    print("Then retrain the background model:")
    print(f"  asarina-observe-train --stat {args.stat}")


if __name__ == '__main__':
    main()
