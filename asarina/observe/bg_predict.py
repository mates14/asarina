#!/usr/bin/env python3
"""
Machine-learned background noise predictor.

Predicts the 1-second background RMS noise from observing conditions using
a gradient boosting regressor trained on per-night ECSV stat data.

Target:
  bgnoise_1s = bgnoise / sqrt(exposure)    [background-limited; read noise negligible]

Features:
  sun_alt          -- solar altitude (deg); drives twilight sky brightness
  moon_alt         -- lunar altitude (deg); above-horizon scattered moonlight
  moon_illumination-- fraction of lunar disk illuminated (0=new, 1=full), derived from JD
  night_fraction   -- position in night centred on midnight: jd - floor(jd+0.5),
                      range [-0.5, +0.5]; distinguishes evening from morning twilight
                      at the same sun_alt, and captures airglow buildup, light pollution
                      tapering after midnight, and zodiacal light near dawn/dusk
  airmass          -- observing airmass; sky brightness ∝ airmass
  filter           -- ordinal-encoded by wavelength (g < r < i < z < N)
  zp_1s            -- 1-second normalised zeropoint; encodes atmospheric transparency
  moon_dist        -- angular distance moon→target (deg); scattered moonlight (NaN = unknown)
  sun_dist         -- angular distance sun→target (deg); mostly relevant in deep twilight (NaN = unknown)

HistGradientBoostingRegressor handles NaN features natively, so moon_dist/sun_dist
can be omitted at prediction time when the target position is not known.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import time
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold

_DEFAULT_MODEL = os.environ.get(
    'RTS2_BGNOISE_MODEL',
    str(Path(__file__).parent / 'bgnoise_model.pkl'),
)

# ---------------------------------------------------------------------------
# Derived time features from JD
# ---------------------------------------------------------------------------
_NEW_MOON_JD  = 2451549.5     # Jan 6 2000 18:14 UTC (known new moon)
_SYNODIC_DAYS = 29.530589     # mean lunar synodic period

def moon_illumination(jd):
    """Approximate fraction of lunar disk illuminated (0 = new, 1 = full)."""
    jd = np.asarray(jd, dtype=float)
    age = (jd - _NEW_MOON_JD) % _SYNODIC_DAYS          # days since new moon
    phase = 2 * np.pi * age / _SYNODIC_DAYS             # phase angle [rad]
    return (1.0 - np.cos(phase)) / 2.0                  # 0..1

def night_fraction(jd):
    """
    Position within the day centred on midnight: range [−0.5, +0.5].

    0 = local solar midnight (approximately).
    Negative = evening (before midnight): twilight winding down, residual
               ionospheric glow, light pollution at full activity.
    Positive = morning (after midnight): airglow building, zodiacal light,
               light pollution tapering, approaching dawn twilight.

    Distinguishes evening from morning twilight at the same sun_alt, and
    captures time-of-night trends in airglow and light pollution that
    sun_alt alone cannot encode.
    """
    jd = np.asarray(jd, dtype=float)
    return jd - np.floor(jd + 0.5)

# ---------------------------------------------------------------------------
# Filter encoding (ordinal, roughly by central wavelength)
# ---------------------------------------------------------------------------
FILTER_ORDER = ['Sloan_g', 'Sloan_r', 'Sloan_i', 'Sloan_z', 'N']

def _encode_filter(f):
    try:
        return float(FILTER_ORDER.index(str(f)))
    except ValueError:
        return float(len(FILTER_ORDER))

# ---------------------------------------------------------------------------
# Feature matrix construction (vectorised)
# ---------------------------------------------------------------------------
FEATURE_NAMES = ['sun_alt', 'moon_alt', 'moon_illum', 'night_frac', 'airmass', 'filter_enc', 'zp_1s',
                 'moon_dist', 'sun_dist']

def _build_X(jd, sun_alt, moon_alt, airmass, filter_name, zp_1s,
             moon_dist=None, sun_dist=None):
    """Assemble feature matrix from equal-length arrays.

    moon_dist and sun_dist are optional (NaN when not available).
    HistGradientBoostingRegressor handles NaN natively.
    """
    jd         = np.asarray(jd,      dtype=float)
    sun_alt    = np.asarray(sun_alt,  dtype=float)
    moon_alt   = np.asarray(moon_alt, dtype=float)
    airmass    = np.asarray(airmass,  dtype=float)
    zp_1s      = np.asarray(zp_1s,   dtype=float)
    filter_enc = np.array([_encode_filter(f) for f in np.atleast_1d(filter_name)])
    n          = len(jd)

    if moon_dist is None:
        moon_dist_arr = np.full(n, np.nan)
    else:
        moon_dist_arr = np.broadcast_to(np.atleast_1d(np.asarray(moon_dist, dtype=float)), (n,))

    if sun_dist is None:
        sun_dist_arr = np.full(n, np.nan)
    else:
        sun_dist_arr = np.broadcast_to(np.atleast_1d(np.asarray(sun_dist, dtype=float)), (n,))

    return np.column_stack([
        sun_alt,
        moon_alt,
        moon_illumination(jd),
        night_fraction(jd),
        airmass,
        filter_enc,
        zp_1s,
        moon_dist_arr,
        sun_dist_arr,
    ])

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    stat_file: str = 'stat.txt',
    model_file: str = _DEFAULT_MODEL,
    verbose: bool = True,
) -> HistGradientBoostingRegressor:
    """
    Train a gradient boosting model on stat.txt and save it.

    The model predicts log(bgnoise_1s) to handle the wide dynamic range.
    Moon illumination is derived from the JD column so no external data needed.

    Parameters
    ----------
    stat_file  : path to the stat.txt observations file
    model_file : where to write the trained model (joblib pickle)
    verbose    : print diagnostics and cross-validation results

    Returns
    -------
    trained GradientBoostingRegressor (also saved to model_file)
    """
    from asarina.observe.stat import read_stat
    data = read_stat(stat_file)
    # accept both 'exptime' (new ECSV) and legacy 'exposure' column name
    if 'exposure' in data.columns and 'exptime' not in data.columns:
        data = data.rename(columns={'exposure': 'exptime'})
        data['zp_1s']      = data['zeropoint'] - 2.5 * np.log10(data['exptime'].clip(lower=1e-3))
        data['bgnoise_1s'] = data['bgnoise']   / np.sqrt(data['exptime'].clip(lower=1e-3))
    data = data[
        (data['exptime'] > 0) & (data['bgnoise'] > 0) &
        (data['airmass'] > 0) & (data['jd'] > 2400000)
    ].copy()

    # Log target for regression (covers ~1 dex of dynamic range)
    y = np.log(data['bgnoise_1s'].values)

    moon_dist = data['moon_dist'].values if 'moon_dist' in data.columns else None
    sun_dist  = data['sun_dist'].values  if 'sun_dist'  in data.columns else None

    X = _build_X(
        data['jd'], data['sun_alt'], data['moon_alt'],
        data['airmass'], data['filter'], data['zp_1s'],  # type: ignore[arg-type]
        moon_dist=moon_dist, sun_dist=sun_dist,
    )

    if verbose:
        print(f"Training on {len(data):,} observations")
        print(f"bgnoise_1s: {data['bgnoise_1s'].quantile(.05):.2f} (p5) – "
              f"{data['bgnoise_1s'].median():.2f} (median) – "
              f"{data['bgnoise_1s'].quantile(.95):.2f} (p95)")
        filt_counts = data['filter'].value_counts()
        print("Filter counts:", dict(filt_counts))
        print()

    # HistGradientBoostingRegressor: histogram-based algorithm (like LightGBM),
    # 10-20x faster than GradientBoostingRegressor on large datasets.
    # verbose=1 prints a progress line every iteration with remaining-time estimate.
    model = HistGradientBoostingRegressor(
        max_iter=400,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=30,
        random_state=42,
        verbose=1 if verbose else 0,
    )

    if verbose:
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        print(f"Running 5-fold cross-validation (5 parallel fits)...")
        t0 = time.time()
        cv = cross_val_score(model, X, y, cv=kf,
                             scoring='neg_mean_absolute_error', n_jobs=-1)
        mae_log = -cv.mean()
        pct_err = 100.0 * (np.exp(mae_log) - 1.0)
        print(f"CV done in {time.time()-t0:.0f}s  "
              f"MAE(log): {mae_log:.4f}  ≈ {pct_err:.1f}% relative error on bgnoise_1s")
        print()
        print("Fitting final model (verbose: one line per iteration):")

    t0 = time.time()
    model.fit(X, y)
    if verbose:
        print(f"Training done in {time.time()-t0:.0f}s")
        print()

    if verbose:
        from sklearn.inspection import permutation_importance
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), size=min(10000, len(X)), replace=False)
        print("Computing permutation importances on 10k-sample subset...")
        pi = permutation_importance(model, X[idx], y[idx],
                                    n_repeats=5, random_state=0, n_jobs=-1)
        print("Feature importances (permutation, mean ± std):")
        order = np.argsort(pi.importances_mean)[::-1]
        for i in order:
            print(f"  {FEATURE_NAMES[i]:<16} {pi.importances_mean[i]:.4f} ± {pi.importances_std[i]:.4f}")
        print()

    payload = {'model': model}
    joblib.dump(payload, model_file)
    if verbose:
        print(f"Model saved → {model_file}")

    return model

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

_cache: dict = {}

def predict_background(
    jd,
    sun_alt,
    moon_alt,
    airmass,
    filter_name,
    zp_1s,
    moon_dist=None,
    sun_dist=None,
    model_file: str = _DEFAULT_MODEL,
) -> float | np.ndarray:
    """
    Predict 1-second background RMS noise for given observing conditions.

    Parameters
    ----------
    jd          : float or array — Julian Date (used to derive moon illumination)
    sun_alt     : float or array — solar altitude (deg)
    moon_alt    : float or array — lunar altitude (deg)
    airmass     : float or array — observing airmass
    filter_name : str or array   — filter name (e.g. 'Sloan_r')
    zp_1s       : float or array — 1-second normalised zeropoint (mag)
    moon_dist   : float or array — moon→target angular distance (deg); None = unknown (NaN)
    sun_dist    : float or array — sun→target angular distance (deg); None = unknown (NaN)
    model_file  : path to saved model (trained by train_model())

    Returns
    -------
    float or np.ndarray — predicted bgnoise_1s in ADU, normalised to 1-second exposure.
    """
    global _cache
    if model_file not in _cache:
        _cache[model_file] = joblib.load(model_file)['model']
    model = _cache[model_file]

    scalar = np.ndim(jd) == 0
    n = 1 if scalar else len(np.atleast_1d(jd))

    def _broadcast(v):
        return np.broadcast_to(np.atleast_1d(v), (n,))

    X = _build_X(
        _broadcast(jd), _broadcast(sun_alt), _broadcast(moon_alt),
        _broadcast(airmass), _broadcast(filter_name), _broadcast(zp_1s),
        moon_dist=None if moon_dist is None else _broadcast(moon_dist),
        sun_dist=None  if sun_dist  is None else _broadcast(sun_dist),
    )
    result = np.exp(model.predict(X))
    return float(result[0]) if scalar else result


# ---------------------------------------------------------------------------
# Real-time residual correction
# ---------------------------------------------------------------------------

# Cross-filter sky residuals transfer poorly — unlike zeropoints, the sky SED
# is a mix of components with very different spectra (OH airglow, moonlight,
# Rayleigh, twilight) whose ratio varies with conditions.  Knowing the sky is
# 20% bright in g says almost nothing reliable about z.  The ML model already
# handles filter differences; the real-time correction should be driven by
# same-filter observations.  Cross-filter observations get a small weight only
# for the weak aerosol/scattering component that is approximately achromatic.
RESIDUAL_TRANSFER_WEIGHT = {
    'Sloan_g': 0.15,
    'Sloan_r': 0.15,
    'Sloan_i': 0.10,
    'Sloan_z': 0.05,   # OH + H2O dominated — essentially no cross-filter signal
    'N':       0.10,
}

# Time decay for residual weighting (same philosophy as predict_zeropoint)
RESIDUAL_TIME_DECAY_MIN = 7.0   # slightly longer than ZP decay; sky changes slower


def predict_background_realtime(
    jd,
    sun_alt,
    moon_alt,
    airmass,
    filter_name,
    zp_1s,
    observations,
    moon_dist=None,
    sun_dist=None,
    window_minutes: float = 15.0,
    time_decay_minutes: float = RESIDUAL_TIME_DECAY_MIN,
    same_filter_weight: float = 2.0,
    model_file: str = _DEFAULT_MODEL,
) -> dict:
    """
    Predict 1-second background noise using the ML model as a prior and
    correcting it with recent actual sky measurements from any filter.

    The correction captures the real-time aerosol/transparency state that
    the static ML model cannot know: if the sky is currently 30% brighter
    than the model predicts (across all available filters), apply that
    correction to the target filter prediction too.

    Unlike zeropoints, sky brightness does NOT transfer reliably between
    filters: the sky SED is a mixture of OH airglow, moonlight, Rayleigh
    scattering, and twilight whose ratio changes with conditions, so sky in
    g says very little about sky in z.  The ML model handles the cross-filter
    question; the real-time correction here is driven almost entirely by
    same-filter observations.  Cross-filter observations carry a very small
    weight for the weak achromatic aerosol component only.

    Parameters
    ----------
    jd, sun_alt, moon_alt, airmass, filter_name, zp_1s
        Target observing conditions (same as predict_background).
    observations : sequence of tuples (jd, filter, bgnoise_1s, sun_alt, moon_alt, airmass, zp_1s[, moon_dist, sun_dist])
        Recent actual background measurements. Elements 8 and 9 (moon_dist, sun_dist)
        are optional; omit or pass NaN if unavailable.
        bgnoise_1s = measured bgnoise / sqrt(exposure).
    moon_dist   : angular distance moon→target for the prediction point (deg); None = NaN
    sun_dist    : angular distance sun→target for the prediction point (deg); None = NaN
    window_minutes : look-back window for recent measurements.
    time_decay_minutes : exponential age weighting of recent obs.
    same_filter_weight : extra weight for observations in the same filter.
    model_file : trained model file.

    Returns
    -------
    dict with keys:
        'bgnoise_1s'      : corrected prediction
        'bgnoise_1s_prior': ML-only prediction (no correction)
        'log_correction'  : log-space additive correction applied
        'n_obs'           : number of recent observations used
        'n_obs_direct'    : same-filter observations used
    """
    # ML prior for the target
    prior = predict_background(jd, sun_alt, moon_alt, airmass,
                               filter_name, zp_1s,
                               moon_dist=moon_dist, sun_dist=sun_dist,
                               model_file=model_file)

    if not observations:
        return {
            'bgnoise_1s': prior, 'bgnoise_1s_prior': prior,
            'log_correction': 0.0, 'n_obs': 0, 'n_obs_direct': 0,
        }

    t_ref   = float(jd)
    t_start = t_ref - window_minutes / 1440.0

    # Compute log-residual for each recent observation:
    #   log_resid = log(actual) - log(model_prediction)
    # A positive residual means the sky was brighter than the model expected.
    log_resids = []
    weights    = []
    n_direct   = 0

    for obs in observations:
        obs = tuple(obs)
        if len(obs) < 7:
            continue
        o_jd, o_filt, o_bg1s, o_sun, o_moon, o_X, o_zp = (
            float(obs[0]), str(obs[1]), float(obs[2]),
            float(obs[3]), float(obs[4]), float(obs[5]), float(obs[6]),
        )
        o_moon_dist = float(obs[7]) if len(obs) > 7 else None
        o_sun_dist  = float(obs[8]) if len(obs) > 8 else None
        if o_jd < t_start or o_jd > t_ref + 0.5 / 1440.0:
            continue
        if o_bg1s <= 0:
            continue

        model_pred = predict_background(o_jd, o_sun, o_moon, o_X,
                                        o_filt, o_zp,
                                        moon_dist=o_moon_dist, sun_dist=o_sun_dist,
                                        model_file=model_file)
        if model_pred <= 0:
            continue

        log_resid = np.log(o_bg1s) - np.log(model_pred)

        dt_min  = (t_ref - o_jd) * 1440.0
        time_w  = np.exp(-dt_min / time_decay_minutes)
        filt_w  = same_filter_weight if o_filt == str(filter_name) else 1.0
        trans_w = RESIDUAL_TRANSFER_WEIGHT.get(o_filt, 0.5)
        w       = filt_w * trans_w * time_w

        log_resids.append(log_resid)
        weights.append(w)
        if o_filt == str(filter_name):
            n_direct += 1

    if not log_resids:
        return {
            'bgnoise_1s': prior, 'bgnoise_1s_prior': prior,
            'log_correction': 0.0, 'n_obs': 0, 'n_obs_direct': 0,
        }

    w_arr = np.array(weights)
    r_arr = np.array(log_resids)
    log_correction = (w_arr * r_arr).sum() / w_arr.sum()

    corrected = prior * np.exp(log_correction)

    return {
        'bgnoise_1s':       corrected,
        'bgnoise_1s_prior': prior,
        'log_correction':   log_correction,
        'n_obs':            len(log_resids),
        'n_obs_direct':     n_direct,
    }


# ---------------------------------------------------------------------------
# CLI: python bg_predict.py [stat.txt [model.pkl]]
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    stat = sys.argv[1] if len(sys.argv) > 1 else 'stat.txt'
    out  = sys.argv[2] if len(sys.argv) > 2 else _DEFAULT_MODEL
    train_model(stat, out, verbose=True)
