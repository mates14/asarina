#!/usr/bin/env python3
"""
Real-time zeropoint predictor for multi-filter photometry.

Physical model:  zp_1s = Z0_f - k_f * X
                 k_f(t) = k_r(t) + beta_f

k_r(t) is a common time-varying extinction that all filters share.
beta_f is a static per-filter offset from the r-band reference.
Z0_f   is the instrument+atmosphere zeropoint at airmass=0.

Given recent observations from any mix of filters, this module estimates
k_r(t) via a weighted linear time trend, then converts back to the
requested filter's zeropoint.
"""

import numpy as np
from typing import Union, List, Optional, Dict, Any

# ---------------------------------------------------------------------------
# Calibrated constants from production fit (fitted_parameters_production.txt)
# ---------------------------------------------------------------------------
FILTER_PARAMS: Dict[str, Dict[str, float]] = {
    'Sloan_g': {'Z0': 21.919920, 'beta':  0.099949},
    'Sloan_r': {'Z0': 21.937985, 'beta':  0.000000},  # reference band
    'Sloan_i': {'Z0': 21.394691, 'beta': -0.067270},
    'Sloan_z': {'Z0': 20.365853, 'beta': -0.056017},
    'N':       {'Z0': 22.986458, 'beta': -0.012176},
}

# Valid zp_norm ranges per filter (from zpfit.py SANITY_LIMITS).
# Observations outside [zp_min, zp_max] are discarded before fitting.
SANITY_LIMITS: Dict[str, Dict[str, float]] = {
    'Sloan_g': {'zp_min': 16.7, 'zp_max': 22.1},
    'Sloan_r': {'zp_min': 16.8, 'zp_max': 22.0},
    'Sloan_i': {'zp_min': 16.3, 'zp_max': 21.5},
    'Sloan_z': {'zp_min': 15.5, 'zp_max': 20.7},
    'N':       {'zp_min': 18.0, 'zp_max': 23.2},
}

# Default weighting parameters (can be overridden per call)
DEFAULT_SAME_FILTER_WEIGHT = 2.0   # multiplier for same-filter observations
DEFAULT_TIME_DECAY_MIN     = 5.0   # exponential decay timescale [minutes]
DEFAULT_WINDOW_MIN         = 15.0  # look-back window [minutes]


# ---------------------------------------------------------------------------

def predict_zeropoint(
    observations: Any,
    target_filters: Optional[Union[str, List[str]]] = None,
    window_minutes: float = DEFAULT_WINDOW_MIN,
    reference_time: Optional[float] = None,
    target_airmass: Optional[Union[float, Dict[str, float]]] = None,
    filter_weight: float = DEFAULT_SAME_FILTER_WEIGHT,
    time_decay_minutes: float = DEFAULT_TIME_DECAY_MIN,
    min_slope_points: int = 3,
    filter_params: Optional[Dict] = None,
    sanity_limits: Optional[Dict] = None,
    sigma_clip: float = 3.0,
) -> Dict[str, Dict]:
    """
    Predict the 1-second normalised zeropoint for one or more filters.

    Parameters
    ----------
    observations : sequence of (jd, filter_name, zp_1s, error)
                   or          (jd, filter_name, zp_1s, error, airmass)
        Recent photometric observations.
        - jd          : Julian Date of the observation
        - filter_name : string matching a key in filter_params
        - zp_1s       : 1-second normalised zeropoint
                        = measured_zp - 2.5 * log10(exposure_seconds)
        - error       : uncertainty on zp_1s (used for weighting; approximate ok)
        - airmass     : (optional) enables exact cross-filter calibration;
                        without it an X≈1 approximation is used

    target_filters : str | list[str] | None
        Which filter(s) to predict. None → all filters seen in the window.

    window_minutes : float
        Look-back window length in minutes (default 15).

    reference_time : float (JD) | None
        Prediction epoch. Defaults to the latest observation time in the input.

    target_airmass : float | dict {filter→float} | None
        Airmass to evaluate the output zp_pred at.
        None → last observed airmass for that filter, or 1.0 if not available.

    filter_weight : float
        Weight multiplier for same-filter observations (default 2.0).

    time_decay_minutes : float
        Exponential decay timescale: obs taken `t` min ago get weight ∝ exp(-t/τ).
        Default 5 min.

    min_slope_points : int
        Minimum observations required to fit a linear slope; fewer → weighted mean.

    filter_params : dict | None
        Override default calibration constants.

    sanity_limits : dict | None
        Per-filter {zp_min, zp_max} to pre-screen corrupt observations.
        Defaults to SANITY_LIMITS. Pass {} to disable.

    sigma_clip : float
        Iterative sigma-clipping threshold in k_r space (default 3.0σ).
        Set to 0 or inf to disable.

    Returns
    -------
    dict  { filter_name →
        {
          'zp_pred'      : float   predicted 1s-zeropoint at target_airmass
          'k_f_pred'     : float   predicted extinction for this filter [mag/airmass]
          'k_r_pred'     : float   predicted r-band reference extinction [mag/airmass]
          'uncertainty'  : float   1-σ uncertainty on zp_pred [mag]
          'n_obs_direct' : int     observations from this filter used
          'n_obs_total'  : int     total observations used (all filters)
          'airmass_used' : float   airmass applied when computing zp_pred
          'slope_per_min': float   linear trend in k_r [mag airmass⁻¹ min⁻¹]
        }
    }
    """
    params  = filter_params if filter_params is not None else FILTER_PARAMS
    slimits = sanity_limits if sanity_limits is not None else SANITY_LIMITS

    # ------------------------------------------------------------------
    # 1. Parse input
    # ------------------------------------------------------------------
    rows = []
    for obs in observations:
        obs = tuple(obs)
        if len(obs) >= 5:
            jd, filt, zp, err, X = (
                float(obs[0]), str(obs[1]), float(obs[2]),
                float(obs[3]), float(obs[4])
            )
        elif len(obs) == 4:
            jd, filt, zp, err = float(obs[0]), str(obs[1]), float(obs[2]), float(obs[3])
            X = None
        else:
            continue
        rows.append({'jd': jd, 'filter': filt, 'zp': zp,
                     'err': max(abs(err), 1e-3), 'X': X})

    if not rows:
        return {}

    # ------------------------------------------------------------------
    # 1b. Sanity-filter: drop obviously bogus zp_norm values
    # ------------------------------------------------------------------
    if slimits:
        rows = [
            r for r in rows
            if r['filter'] not in slimits
            or (slimits[r['filter']]['zp_min'] <= r['zp'] <= slimits[r['filter']]['zp_max'])
        ]
    if not rows:
        return {}

    has_airmass = any(r['X'] is not None for r in rows)

    # ------------------------------------------------------------------
    # 2. Apply time window
    # ------------------------------------------------------------------
    jd_max = max(r['jd'] for r in rows)
    t_ref = float(reference_time) if reference_time is not None else jd_max
    t_start = t_ref - window_minutes / 1440.0

    window = [r for r in rows if t_start <= r['jd'] <= t_ref + 0.5 / 1440.0]
    if not window:
        return {}

    # ------------------------------------------------------------------
    # 3. Resolve target filter list
    # ------------------------------------------------------------------
    observed_known = {r['filter'] for r in window if r['filter'] in params}
    if target_filters is None:
        target_list = sorted(observed_known)
    elif isinstance(target_filters, str):
        target_list = [target_filters]
    else:
        target_list = list(target_filters)

    # ------------------------------------------------------------------
    # 4. Convert each observation to k_r
    #
    #    With airmass:   k_r = (Z0_f − zp_f) / X − beta_f        [exact]
    #    Without airmass: k_r ≈  Z0_f − zp_f − beta_f            [X ≈ 1]
    # ------------------------------------------------------------------
    kr_pts = []
    for r in window:
        filt = r['filter']
        if filt not in params:
            continue
        Z0, beta = params[filt]['Z0'], params[filt]['beta']

        if has_airmass and r['X'] is not None:
            X = r['X']
            k_r = (Z0 - r['zp']) / X - beta
            sigma = r['err'] / X           # error propagation through ÷X
        else:
            k_r = Z0 - r['zp'] - beta      # approximate (X ≈ 1)
            sigma = r['err']

        dt_min = (t_ref - r['jd']) * 1440.0
        time_w = np.exp(-dt_min / time_decay_minutes)

        kr_pts.append({
            'jd': r['jd'],
            'filter': filt,
            'k_r': k_r,
            'sigma': max(sigma, 1e-3),
            'time_w': time_w,
        })

    if not kr_pts:
        return {}

    # ------------------------------------------------------------------
    # 4b. Sigma-clipping in k_r space (filter-independent)
    # ------------------------------------------------------------------
    if sigma_clip > 0 and sigma_clip < np.inf and len(kr_pts) >= 4:
        k_all = np.array([p['k_r'] for p in kr_pts])
        for _ in range(3):
            med = np.median(k_all)
            mad = np.median(np.abs(k_all - med)) * 1.4826  # robust σ
            if mad < 1e-6:
                break
            mask = np.abs(k_all - med) <= sigma_clip * mad
            if mask.sum() < 3:
                break
            kr_pts = [p for p, m in zip(kr_pts, mask) if m]
            k_all  = np.array([p['k_r'] for p in kr_pts])

    if not kr_pts:
        return {}

    t_arr = np.array([(p['jd'] - t_ref) * 1440.0 for p in kr_pts])   # minutes from t_ref
    k_arr = np.array([p['k_r'] for p in kr_pts])

    # ------------------------------------------------------------------
    # 5. Per-target prediction
    # ------------------------------------------------------------------
    results: Dict[str, Dict] = {}

    for target in target_list:
        if target not in params:
            results[target] = {'error': f'Unknown filter: {target}'}
            continue

        # Weights: filter_pref × time_decay × inverse-variance
        w_arr = np.array([
            (filter_weight if p['filter'] == target else 1.0)
            * p['time_w']
            / p['sigma'] ** 2
            for p in kr_pts
        ])
        W = w_arr.sum()
        n = len(kr_pts)

        # Weighted linear fit  k_r(t) = a + b·t  (t in minutes, origin at t_ref)
        if n >= min_slope_points:
            Wt  = (w_arr * t_arr).sum()
            Wt2 = (w_arr * t_arr ** 2).sum()
            Wk  = (w_arr * k_arr).sum()
            Wkt = (w_arr * k_arr * t_arr).sum()

            denom = W * Wt2 - Wt ** 2
            if abs(denom) > 1e-12 * max(W * Wt2, 1.0):
                a = (Wk * Wt2 - Wkt * Wt) / denom   # intercept at t_ref
                b = (W * Wkt - Wt * Wk) / denom      # slope [mag/airmass/min]
            else:
                a = Wk / W
                b = 0.0
        else:
            a = (w_arr * k_arr).sum() / W
            b = 0.0

        k_r_pred = a   # prediction at t = t_ref

        # ------------------------------------------------------------------
        # Uncertainty: weighted RMSE of residuals, scaled by effective √N
        # ------------------------------------------------------------------
        resid = k_arr - (a + b * t_arr)
        if n > 2:
            s2 = (w_arr * resid ** 2).sum() / W     # weighted mean squared residual
            n_eff = W ** 2 / (w_arr ** 2).sum()     # effective sample count
            sigma_kr = np.sqrt(s2 / max(n_eff, 1.0))
        elif n == 2:
            sigma_kr = np.abs(resid).mean()
        else:
            sigma_kr = np.abs(k_arr).mean() * 0.05 + 0.05

        # ------------------------------------------------------------------
        # Convert back to zp space
        # ------------------------------------------------------------------
        Z0_t  = params[target]['Z0']
        beta_t = params[target]['beta']
        k_f_pred = k_r_pred + beta_t

        # Determine output airmass
        if isinstance(target_airmass, dict):
            X_out = float(target_airmass.get(target, 1.0))
        elif target_airmass is not None:
            X_out = float(target_airmass)
        else:
            direct = sorted(
                [r for r in window if r['filter'] == target and r['X'] is not None],
                key=lambda r: r['jd'],
            )
            X_out = direct[-1]['X'] if direct else 1.0

        zp_pred = Z0_t - k_f_pred * X_out
        zp_unc  = sigma_kr * X_out

        results[target] = {
            'zp_pred':       zp_pred,
            'k_f_pred':      k_f_pred,
            'k_r_pred':      k_r_pred,
            'uncertainty':   zp_unc,
            'n_obs_direct':  sum(1 for r in window if r['filter'] == target),
            'n_obs_total':   n,
            'airmass_used':  X_out,
            'slope_per_min': b,
        }

    return results
