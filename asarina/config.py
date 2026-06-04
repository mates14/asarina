"""Per-camera configuration file support for asarina tools.

Config files are read in cascade order; later files override earlier ones:
  1. /etc/asarina/config   — system-wide (shared by all users, including root)
  2. ~/.config/asarina/config — per-user override
  --config FILE replaces the cascade and reads only the named file.

  [DEFAULT]        — baseline defaults applied to every camera
  [<camera_name>]  — overrides for a specific camera (matched against CCD_NAME
                     from the FITS header, case-insensitive)

Recognised keys (underscores or dashes interchangeable):
  sbt_window_patch  bool   false
  sip               int    1
  passes            int    2
  pixel_scale       float
  makak             bool   false
  dophot_model      str
  dophot_catalog    str
  dophot_maglim     float
  dophot_enlarge    float
  dophot_terms      str
  dophot_idlimit    int
  dophot_max_stars  int    1000
  smart_dark        str
  stat_dir          str
  ssh_key           str

Example /etc/asarina/config:
  [C2]
  sbt_window_patch = true
  sip = 2
  passes = 3
  dophot_terms = .p5,.r5,.l
"""

import argparse
import configparser
import os
import sys
from typing import Optional

DEFAULT_CONFIG = """\
[DEFAULT]
sip = 1
passes = 2
dophot_max_stars = 1000
"""

# Cascade: system-wide first, then per-user (later entries override earlier).
SYSTEM_CONFIG_FILE = '/etc/asarina/config'
USER_CONFIG_FILE   = '~/.config/asarina/config'

_BOOL_KEYS  = frozenset({'sbt_window_patch', 'makak'})
_INT_KEYS   = frozenset({'sip', 'passes', 'dophot_idlimit', 'dophot_max_stars'})
_FLOAT_KEYS = frozenset({'pixel_scale', 'dophot_maglim', 'dophot_enlarge',
                          'gain', 'readnoise', 'ape', 'default_fwhm'})


def _norm(key: str) -> str:
    return key.lower().replace('-', '_')


def load_config(config_file: Optional[str] = None,
                camera: Optional[str] = None) -> dict:
    """Return merged config dict: built-in defaults → config file(s) → camera section.

    If config_file is given, only that file is read.
    Otherwise reads the default cascade (system → user), each overriding the previous.
    """
    cp = configparser.ConfigParser(default_section=None)
    cp.read_string(DEFAULT_CONFIG)

    if config_file is not None:
        path = os.path.expanduser(config_file)
        if os.path.exists(path):
            cp.read(path)
    else:
        for p in (SYSTEM_CONFIG_FILE, os.path.expanduser(USER_CONFIG_FILE)):
            if os.path.exists(p):
                cp.read(p)

    result = {}
    if 'DEFAULT' in cp:
        result.update({_norm(k): v for k, v in cp['DEFAULT'].items()})

    if camera is not None:
        for candidate in (camera, camera.upper(), camera.lower()):
            if candidate in cp:
                result.update({_norm(k): v for k, v in cp[candidate].items()})
                break

    return _coerce(result)


def _coerce(cfg: dict) -> dict:
    out = {}
    for k, v in cfg.items():
        if k in _BOOL_KEYS:
            out[k] = str(v).lower() in ('true', '1', 'yes')
        elif k in _INT_KEYS:
            try:
                out[k] = int(v)
            except (ValueError, TypeError):
                out[k] = None
        elif k in _FLOAT_KEYS:
            try:
                out[k] = float(v)
            except (ValueError, TypeError):
                out[k] = None
        else:
            out[k] = v if v else None
    return out


def detect_camera(fits_file: str) -> Optional[str]:
    """Return CCD_NAME from the FITS header, or None on any failure."""
    try:
        from astropy.io import fits
        with fits.open(fits_file) as hdul:
            val = hdul[0].header.get('CCD_NAME')
            if val:
                return val.strip()
    except Exception:
        pass
    return None


def pre_parse(argv=None):
    """First-pass parse to extract --config/--camera and auto-detect camera from FITS.

    Returns (config_file, camera, remaining_args).
    remaining_args has --config and --camera stripped out; pass it to the
    main argparse parser.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    pre.add_argument('--camera', default=None)
    conf_args, remaining = pre.parse_known_args(
        argv if argv is not None else sys.argv[1:]
    )

    camera = conf_args.camera
    if camera is None:
        for arg in remaining:
            if not arg.startswith('-') and arg.lower().endswith(('.fits', '.fit')):
                camera = detect_camera(arg)
                if camera:
                    break

    return conf_args.config, camera, remaining


def as_argparse_defaults(config: dict) -> dict:
    """Filter None values; return dict suitable for parser.set_defaults()."""
    return {k: v for k, v in config.items() if v is not None}
