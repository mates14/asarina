"""Per-camera configuration for the observe package.

Hardware constants are read from /etc/asarina/config (or the cascade defined
by asarina.config).  Photometric calibration constants (filter_params,
sanity_limits) are baked into zp_predict.py — update them there after
running zpfit and then retrain the background model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from asarina.observe.expcalc import ExposureCalculator, GAIN, RN, APE
from asarina.observe.zp_predict import FILTER_PARAMS, SANITY_LIMITS

_DEFAULT_CAMERA = os.environ.get('ASARINA_CAMERA', 'C0')

_FILTER_TOKEN: Dict[str, str] = {
    'Sloan_g': 'g', 'Sloan_r': 'r', 'Sloan_i': 'i', 'Sloan_z': 'z', 'N': 'N',
}


@dataclass
class CameraConfig:
    """Hardware and photometric calibration parameters for one camera.

    Attributes
    ----------
    gain         : CCD gain [e-/ADU]
    readnoise    : effective readout noise [electrons RMS]
    ape          : aperture growth curve parameter (fitted from photon noise data)
    default_fwhm : typical PSF FWHM [pixels]
    pixel_scale  : arcsec/pixel; None = unknown
    filter_params: {filter → {Z0, beta}} — from zp_predict.FILTER_PARAMS
    sanity_limits: {filter → {zp_min, zp_max}} — from zp_predict.SANITY_LIMITS
    filter_token : {filter_name → camera filter wheel token}
    model_file   : path to trained bgnoise_model.pkl
    """

    gain:          float = GAIN
    readnoise:     float = RN
    ape:           float = APE
    default_fwhm:  float = 3.0
    pixel_scale:   Optional[float] = None

    filter_params:  Dict = field(default_factory=lambda: dict(FILTER_PARAMS))
    sanity_limits:  Dict = field(default_factory=lambda: dict(SANITY_LIMITS))
    filter_token:   Dict = field(default_factory=lambda: dict(_FILTER_TOKEN))
    model_file:     str  = field(default_factory=lambda: os.environ.get(
        'RTS2_BGNOISE_MODEL',
        str(Path(__file__).parent / 'bgnoise_model.pkl'),
    ))

    def __post_init__(self):
        self._calc = ExposureCalculator(
            gain=self.gain, readnoise=self.readnoise, ape=self.ape,
        )

    @classmethod
    def load(cls, camera: Optional[str] = None,
             config_file: Optional[str] = None) -> 'CameraConfig':
        """Load hardware constants for camera from /etc/asarina/config."""
        from asarina.config import load_config
        name = camera or _DEFAULT_CAMERA
        d = load_config(config_file=config_file, camera=name)

        cfg = cls.__new__(cls)
        cfg.gain         = float(d.get('gain',         GAIN))
        cfg.readnoise    = float(d.get('readnoise',    RN))
        cfg.ape          = float(d.get('ape',          APE))
        cfg.default_fwhm = float(d.get('default_fwhm', 3.0))
        cfg.pixel_scale  = d.get('pixel_scale')   # already float or None from _coerce
        cfg.filter_params = dict(FILTER_PARAMS)
        cfg.sanity_limits = dict(SANITY_LIMITS)
        cfg.filter_token  = dict(_FILTER_TOKEN)
        cfg.model_file   = os.environ.get(
            'RTS2_BGNOISE_MODEL',
            str(Path(__file__).parent / 'bgnoise_model.pkl'),
        )
        cfg._calc = ExposureCalculator(
            gain=cfg.gain, readnoise=cfg.readnoise, ape=cfg.ape,
        )
        return cfg

    # ------------------------------------------------------------------
    # Exposure calculator delegation
    # ------------------------------------------------------------------

    def sky_1s_from_bgnoise(self, bgnoise_1s: float) -> float:
        return self._calc.sky_1s_from_bgnoise(bgnoise_1s)

    def calculate_exptime(self, magnitude, magerror, fwhm, zp_1s, sky_1s) -> float:
        return self._calc.calculate_exptime(magnitude, magerror, fwhm, zp_1s - 10.0, sky_1s)

    def predict_performance(self, magnitude, exptime, fwhm, zp_1s, sky_1s):
        return self._calc.predict_performance(magnitude, exptime, fwhm, zp_1s - 10.0, sky_1s)
