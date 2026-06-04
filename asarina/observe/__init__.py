"""asarina.observe — real-time observation planning and photometric condition modelling.

Key functions:
    predict_zeropoint()   — estimate current 1s-ZP from recent pipeline output
    predict_background()  — estimate current 1s sky background noise (ML model)
    CameraConfig.load()   — hardware constants from /etc/asarina/config

Data paths (override via environment variables):
    RTS2_STAT_DIR       — directory containing per-night stat ECSV files
    RTS2_BGNOISE_MODEL  — path to the trained background noise model
    ASARINA_CAMERA      — default camera name (section in /etc/asarina/config)
"""

from asarina.observe.zp_predict import predict_zeropoint, FILTER_PARAMS, SANITY_LIMITS
from asarina.observe.bg_predict import (
    predict_background, predict_background_realtime, train_model,
    moon_illumination, night_fraction,
)
from asarina.observe.camera import CameraConfig
from asarina.observe.zpfit import fit_zeropoints
from asarina.observe.apecalfit import fit_ape
from asarina.observe.stat import (
    read_stat, write_stat_record, record_from_ecsv,
    night_from_jd, load_recent as stat_load_recent,
)

__all__ = [
    'predict_zeropoint',
    'FILTER_PARAMS',
    'SANITY_LIMITS',
    'predict_background',
    'predict_background_realtime',
    'train_model',
    'moon_illumination',
    'night_fraction',
    'CameraConfig',
    'fit_zeropoints',
    'fit_ape',
    'read_stat',
    'write_stat_record',
    'record_from_ecsv',
    'night_from_jd',
]
