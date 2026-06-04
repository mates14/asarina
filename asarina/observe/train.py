#!/usr/bin/env python3
"""
Training pipeline for the bgnoise background noise predictor.

Usage
-----
    rtspy-observe-train --stat stat.txt --config telescope.yaml
    rtspy-observe-train --stat stat.txt --config telescope.yaml --model out.pkl
    rtspy-observe-train --stat stat.txt  # uses built-in D50/C0 config

What it does
------------
1. Loads telescope config (filter_params, sanity_limits, hardware constants).
2. Trains HistGradientBoostingRegressor on stat.txt → predicts log(bgnoise_1s).
3. Evaluates on a held-out set (most recent 20% by JD) and prints a summary.
4. Saves the model to the path specified in --model or in the config.

Workflow for a new telescope
-----------------------------
1. Collect stat.txt from the photometric pipeline.
2. Run zpfit.py to fit Z0 and beta per filter.
3. Fill in a config YAML (use config_d50_c0.yaml as a template).
4. Run this script to train and evaluate the model.
5. Deploy the config YAML + model file to the telescope.
   Set environment variables:
       export RTS2_OBSERVE_CONFIG=/path/to/telescope.yaml
       export RTS2_STAT_FILE=/path/to/stat.txt
"""

import argparse
import sys
import time

import numpy as np

from asarina.observe.camera import CameraConfig
from asarina.observe.bg_predict import (
    train_model, predict_background,
    _build_X, FEATURE_NAMES,
)


def _held_out_test(stat_file: str, model_file: str, held_frac: float = 0.20) -> None:
    """Evaluate the trained model on the most recent held_frac of stat data."""
    from asarina.observe.stat import read_stat
    data = read_stat(stat_file)
    if 'exposure' in data.columns and 'exptime' not in data.columns:
        data = data.rename(columns={'exposure': 'exptime'})
        data['zp_1s']      = data['zeropoint'] - 2.5 * np.log10(data['exptime'].clip(lower=1e-3))
        data['bgnoise_1s'] = data['bgnoise'] / np.sqrt(data['exptime'].clip(lower=1e-3))
    data = data[
        (data['exptime'] > 0) & (data['bgnoise'] > 0) &
        (data['airmass'] > 0) & (data['jd'] > 2400000)
    ].copy()

    cutoff = data['jd'].quantile(1.0 - held_frac)
    test   = data[data['jd'] >= cutoff].copy()
    if len(test) == 0:
        print("Warning: no held-out test data.")
        return

    print(f"\nHeld-out test (most recent {held_frac*100:.0f}%  —  {len(test):,} obs, "
          f"JD ≥ {cutoff:.2f})")

    moon_dist = test['moon_dist'].values if 'moon_dist' in test.columns else None
    sun_dist  = test['sun_dist'].values  if 'sun_dist'  in test.columns else None

    pred = predict_background(
        test['jd'].values, test['sun_alt'].values, test['moon_alt'].values,
        test['airmass'].values, test['filter'].values, test['zp_1s'].values,
        moon_dist=moon_dist, sun_dist=sun_dist,
        model_file=model_file,
    )

    actual  = test['bgnoise_1s'].values
    rel_err = (pred - actual) / actual

    print(f"  Relative error  (pred−actual)/actual:")
    print(f"    median  {np.median(rel_err)*100:+.1f}%  (bias)")
    print(f"    mean    {np.mean(rel_err)*100:+.1f}%")
    print(f"    MAE     {np.median(np.abs(rel_err))*100:.1f}%")
    print(f"    p90     {np.percentile(np.abs(rel_err), 90)*100:.1f}%")

    # Per-filter breakdown
    print(f"\n  Per-filter MAE:")
    for filt in sorted(test['filter'].unique()):
        mask = test['filter'].values == filt
        if mask.sum() < 10:
            continue
        mae = np.median(np.abs(rel_err[mask])) * 100
        bias = np.median(rel_err[mask]) * 100
        print(f"    {filt:<12} {mae:.1f}%  (bias {bias:+.1f}%,  n={mask.sum():,})")


def main():
    parser = argparse.ArgumentParser(
        description='Train bgnoise background predictor for a telescope',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--stat', required=True, metavar='FILE',
                        help='Path to the pipeline stat.txt training file')
    parser.add_argument('--camera', metavar='NAME', default=None,
                        help='Camera section in /etc/asarina/config (default: $ASARINA_CAMERA or C0)')
    parser.add_argument('--model', metavar='FILE', default=None,
                        help='Output model path (default: from $RTS2_BGNOISE_MODEL or package dir)')
    parser.add_argument('--no-test', action='store_true',
                        help='Skip held-out evaluation after training')
    parser.add_argument('--held-frac', type=float, default=0.20, metavar='FRAC',
                        help='Fraction of most-recent data used for held-out test')
    args = parser.parse_args()

    cfg = CameraConfig.load(args.camera)
    model_file = args.model or cfg.model_file

    print(f"Camera            : {args.camera or 'C0 (default)'}")
    print(f"Training data     : {args.stat}")
    print(f"Model output      : {model_file}")
    print(f"Hardware          : GAIN={cfg.gain}  RN={cfg.readnoise}  APE={cfg.ape}")
    print(f"Filters           : {list(cfg.filter_params.keys())}")
    print()

    t0 = time.time()
    train_model(args.stat, model_file, verbose=True)
    elapsed = time.time() - t0
    print(f"\nTotal training time: {elapsed:.0f}s")

    if not args.no_test:
        _held_out_test(args.stat, model_file, held_frac=args.held_frac)

    print(f"\nDeploy: copy {model_file} to the telescope.")
    print(f"  Set   export RTS2_BGNOISE_MODEL=/path/to/bgnoise_model.pkl")
    print(f"  Set   export RTS2_STAT_FILE=/path/to/stat.txt")
    print(f"  Run   rts2-observe -m <mag> -s <snr>")


if __name__ == '__main__':
    main()
