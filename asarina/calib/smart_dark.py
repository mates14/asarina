"""Per-pixel analytical dark correction for cameras with a dark current model.

The model file is a .npy array of shape (rows, cols, 3) where the three planes
are coefficients A, B, C such that:

    dark_current(T) = A * B^T + C

Temperature T is in degrees Celsius.  The correction is applied as:

    corrected = image - dark_current(T)

The optimum T is found by minimising image noise (sum of absolute pixel
differences along columns), allowing a small deviation from the measured
CCD temperature to compensate for sensor non-uniformity or thermistor offset.

This model was developed for the Makak all-sky zenith camera (mi0315) but
is camera-agnostic — any camera with a characterised dark response can use it.
"""

import numpy as np
from scipy.optimize import minimize_scalar


def apply_dark_correction(image, calibration, temperature):
    A, B, C = calibration[:, :, 0], calibration[:, :, 1], calibration[:, :, 2]
    return image - (A * np.power(B, temperature) + C)


def _image_noise(image):
    """Sum of absolute column-wise pixel differences — minimised by optimal T."""
    return np.sum(np.abs(np.diff(image, axis=1)))


def _optimize_temperature(image, calibration, initial_temp, temp_range=10):
    result = minimize_scalar(
        lambda t: _image_noise(apply_dark_correction(image, calibration, t)),
        method='brent',
        bracket=(initial_temp - temp_range / 2, initial_temp + temp_range / 2),
    )
    if not result.success or abs(result.x - initial_temp) > temp_range:
        return initial_temp
    return result.x


def smart_dark(image, calibration_path, initial_temp=20.0):
    """Apply smart dark correction.

    Parameters
    ----------
    image : ndarray (float64)
        Raw image data.
    calibration_path : str
        Path to the .npy calibration file with shape (rows, cols, 3).
    initial_temp : float
        Initial temperature guess in °C (default 20.0).

    Returns
    -------
    corrected_image : ndarray
    optimal_temp : float
        Temperature (°C) that minimised residual image noise.
    """
    calibration = np.load(calibration_path)
    optimal_temp = _optimize_temperature(image, calibration, initial_temp)
    return apply_dark_correction(image, calibration, optimal_temp), optimal_temp


def image_bgsigma(data):
    """Estimate background noise as median of row-wise median absolute differences.

    Robust to stars and gradients; used both for dark quality assessment and as
    the BGSIGMA header keyword written into calibrated science frames.
    """
    ndiff = []
    for i in range(len(data) - 1):
        d = np.nanmedian(np.abs(np.float32(data[i]) - np.float32(data[i + 1])))
        if not np.isnan(d):
            ndiff.append(d)
    return float(np.nanmedian(ndiff)) if ndiff else float('nan')
