#!/usr/bin/env python3
"""
Exposure time calculator based on empirical calibration data.

Two modes:
1. Reference image: --refimg <ecsv-file> (extracts conditions from real observation)
2. 1s normalized: --magzero <val> --sky <val> (uses pre-calculated 1s values)

Usage examples:
python exposure_calculator.py 18.5 --refimg reference.ecsv --snr 10
python exposure_calculator.py 18.5 --refimg reference.ecsv 2.0 --snr 10  # override FWHM
python exposure_calculator.py 18.5 2.1 --magzero 30.2 --sky 5.5 --snr 10

Fitted parameter: ape = 6.146 ± 0.002
RMS residual: 0.047 in log(magerror) (~5% accuracy)
"""

import numpy as np
from scipy.optimize import fsolve
import argparse
from astropy.table import Table

# Fitted parameters
APE = 6.146      # Fitted transition parameter
GAIN = 0.81      # CCD gain  
ZERO = 10        # Zero offset
RN = 8.0         # Effective readout noise (electrons)

def sbl(A, B, N, x):
    """Smooth transition function between photon and background noise regimes."""
    return A*x + (B-A)*(abs(N)*np.sqrt(1.0 + x*x/(N*N)) + x)/2.0

def break_magnitude(bgsigma, fwhm):
    """Calculate transition magnitude where photon noise equals background noise."""
    return -2.5*np.log10(APE * np.pi/4 * fwhm*fwhm * (bgsigma*GAIN)**2) + ZERO

def log_magerror(magnitude, bgsigma, fwhm):
    """Predict log10(magnitude error) for given magnitude, background, and seeing."""
    break_mag = break_magnitude(bgsigma, fwhm)
    return sbl(0.2, 0.4, 2.5, magnitude - break_mag) + 0.2*break_mag - 2

def sky_brightness_from_bgsigma(bgsigma_ref, exptime_ref):
    """Convert reference bgsigma to sky brightness (photons/s/pixel)."""
    return (GAIN**2 * bgsigma_ref**2 - GAIN**2 * RN**2) / exptime_ref

def bgsigma_from_sky_brightness(sky_brightness, exptime):
    """Calculate bgsigma for given exposure time and sky brightness."""
    return np.sqrt(sky_brightness * exptime / GAIN**2 + RN**2)

def read_reference_image(refimg_path):
    """Extract reference conditions from ecsv file header."""
    try:
        table = Table.read(refimg_path, format='ascii.ecsv')
        header = table.meta
        
        # Extract required header keywords
        exptime_ref = header.get('EXPTIME', None)
        bgsigma_ref = header.get('BGSIGMA', None) 
        magzero_ref = header.get('MAGZERO', None) # - 10
        fwhm_ref = header.get('FWHM', None)
        
        # Check for missing values
        missing = []
        if exptime_ref is None: missing.append('EXPTIME')
        if bgsigma_ref is None: missing.append('BGSIGMA')
        if magzero_ref is None: missing.append('MAGZERO')
        if fwhm_ref is None: missing.append('FWHM')
        
        if missing:
            raise ValueError(f"Missing required header keywords: {', '.join(missing)}")
        
        # Convert to 1s normalized values
        sky_1s = sky_brightness_from_bgsigma(bgsigma_ref, exptime_ref)
        magzero_1s = magzero_ref - 2.5*np.log10(exptime_ref)
        
        return magzero_1s, sky_1s, fwhm_ref, exptime_ref, bgsigma_ref, magzero_ref
        
    except UnicodeDecodeError:
        raise ValueError(f"{refimg_path} appears to be a binary FITS image; --refimg expects an ECSV catalogue file (e.g. *-dft.ecsv)")
    except Exception as e:
        raise ValueError(f"Error reading reference catalogue {refimg_path}: {e}")

def calculate_exptime(target_magnitude, target_magerror, fwhm, magzero_1s, sky_1s):
    """
    Calculate required exposure time.
    
    Parameters:
    -----------
    target_magnitude : float
        Target object magnitude (catalog/measured)
    target_magerror : float
        Required magnitude error (not log10!)
    fwhm : float
        Seeing FWHM (pixels)
    magzero_1s : float
        Magnitude zeropoint normalized to 1 second exposure
    sky_1s : float
        Sky brightness in photons/s/pixel
        
    Returns:
    --------
    exptime : float
        Required exposure time (seconds)
    """
    target_log_magerror = np.log10(target_magerror)
    
    def equation(log_exptime):
        exptime_new = 10**log_exptime
        
        # Calculate background noise for this exposure time
        bgsigma_new = bgsigma_from_sky_brightness(sky_1s, exptime_new)
        
        # Zeropoint for new exposure time
        magzero_new = magzero_1s + 2.5*log_exptime
        
        # Convert target magnitude to magnitude without zeropoint
        mag_relative_to_zero = target_magnitude - magzero_new
        
        predicted_log_magerror = log_magerror(mag_relative_to_zero, bgsigma_new, fwhm)
        return predicted_log_magerror - target_log_magerror
    
    try:
        log_exptime_solution = fsolve(equation, np.log10(300))[0]  # Start at 300s
        return 10**log_exptime_solution
    except:
        return np.nan

def predict_performance(target_magnitude, exptime, fwhm, magzero_1s, sky_1s):
    """Predict magnitude error for given exposure time and conditions."""
    bgsigma_new = bgsigma_from_sky_brightness(sky_1s, exptime)
    magzero_new = magzero_1s + 2.5*np.log10(exptime)
    
    mag_relative_to_zero = target_magnitude - magzero_new
    predicted_log_magerror = log_magerror(mag_relative_to_zero, bgsigma_new, fwhm)
    predicted_magerror = 10**predicted_log_magerror
    predicted_snr = magerror_to_snr(predicted_magerror)
    
    return predicted_magerror, predicted_snr

def snr_to_magerror(snr):
    """Convert SNR to magnitude error: magerr = 1/(SNR * ln(10)/2.5)"""
    return 1.0 / (snr * np.log(10) / 2.5)

def magerror_to_snr(magerror):
    """Convert magnitude error to SNR."""
    return 1.0 / (magerror * np.log(10) / 2.5)

def main():
    parser = argparse.ArgumentParser(description='Calculate exposure time for target photometry')
    parser.add_argument('magnitude', type=float, help='Target object magnitude (catalog/measured)')
    parser.add_argument('fwhm', type=float, nargs='?', help='Seeing FWHM (pixels), optional with --refimg')
    
    # Input mode selection
    parser.add_argument('--refimg', type=str, help='Reference ECSV catalogue file (e.g. *-dft.ecsv), NOT a FITS image')
    parser.add_argument('--magzero', type=float, help='Magnitude zeropoint for 1s exposure')
    parser.add_argument('--sky', type=float, help='Sky brightness (photons/s/pixel)')
    
    # Target specification
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument('--magerror', type=float, help='Target magnitude error')
    target_group.add_argument('--snr', type=float, help='Target SNR')
    
    args = parser.parse_args()
    
    # Validate input combinations
    if args.refimg and (args.magzero is not None or args.sky is not None):
        parser.error("Cannot use --refimg with --magzero or --sky")
    if not args.refimg and (args.magzero is None or args.sky is None):
        parser.error("Must provide either --refimg OR both --magzero and --sky")
    if not args.refimg and args.fwhm is None:
        parser.error("FWHM is required when using --magzero and --sky")
    
    # Extract reference conditions
    if args.refimg:
        magzero_1s, sky_1s, fwhm_ref, exptime_ref, bgsigma_ref, magzero_ref = read_reference_image(args.refimg)
        print(f"Reference image: {args.refimg}")
        print(f"  Original: {exptime_ref}s, σ_bg={bgsigma_ref:.1f}, magzero={magzero_ref:.2f}, FWHM={fwhm_ref:.1f}px")
        print(f"  1s normalized: magzero={magzero_1s:.2f}, sky={sky_1s:.1f} ph/s/px")
        
        # Use FWHM from command line if provided, otherwise use reference FWHM
        if args.fwhm is not None:
            fwhm = args.fwhm
            if abs(fwhm - fwhm_ref) > 0.1:  # Only note if significantly different
                print(f"  Using command line FWHM={fwhm:.1f}px (differs from reference {fwhm_ref:.1f}px)")
        else:
            fwhm = fwhm_ref
            print(f"  Using reference FWHM={fwhm:.1f}px")
    else:
        magzero_1s = args.magzero
        sky_1s = args.sky
        fwhm = args.fwhm
        print(f"1s normalized conditions: magzero={magzero_1s:.2f}, sky={sky_1s:.1f} ph/s/px, FWHM={fwhm:.1f}px")
    
    # Determine target
    if args.snr:
        target_magerror = snr_to_magerror(args.snr)
        print(f"Target SNR {args.snr} corresponds to magnitude error {target_magerror:.4f}")
    else:
        target_magerror = args.magerror
        print(f"Target magnitude error: {target_magerror:.4f}")
    
    # Calculate required exposure time
    # the original calculation needs the old zeropoint definition z-10
    exptime = calculate_exptime(args.magnitude, target_magerror, fwhm, magzero_1s - 10.0, sky_1s)
    
    print(f"\nTarget magnitude: {args.magnitude:.2f}")
    print(f"Required exposure time: {exptime:.1f} seconds")
    
    # Verification (again old definition of magzero)
    pred_magerror, pred_snr = predict_performance(args.magnitude, exptime, fwhm, magzero_1s - 10, sky_1s)
    
    print(f"\nVerification:")
    print(f"Predicted mag error: {pred_magerror:.4f}")
    print(f"Predicted SNR: {pred_snr:.1f}")
    
    # Show final observing conditions
    final_magzero = magzero_1s + 2.5*np.log10(exptime)
    final_bgsigma = bgsigma_from_sky_brightness(sky_1s, exptime)
    print(f"Final conditions: magzero={final_magzero:.2f}, bgsigma={final_bgsigma:.1f}")

class ExposureCalculator:
    """
    Exposure time calculator with per-instance telescope parameters.

    Use this when working with multiple cameras that have different GAIN, RN,
    or APE values.  The module-level functions use the hardcoded constants
    above (D50/C0 defaults); this class lets you override them per telescope.

    Parameters
    ----------
    gain      : CCD gain [e-/ADU]
    readnoise : effective readout noise [electrons RMS]
    ape       : aperture growth curve parameter (fitted from photon noise data)
    """

    def __init__(self, gain: float = GAIN, readnoise: float = RN, ape: float = APE):
        self.gain      = gain
        self.readnoise = readnoise
        self.ape       = ape

    def bgsigma_from_sky_brightness(self, sky_brightness: float, exptime: float) -> float:
        return np.sqrt(sky_brightness * exptime / self.gain**2 + self.readnoise**2)

    def sky_brightness_from_bgsigma(self, bgsigma_ref: float, exptime_ref: float) -> float:
        return (self.gain**2 * bgsigma_ref**2 - self.gain**2 * self.readnoise**2) / exptime_ref

    def sky_1s_from_bgnoise(self, bgnoise_1s: float) -> float:
        """bgnoise_1s [ADU] → sky_1s [photons/s/pixel]."""
        return max(self.gain**2 * bgnoise_1s**2 - self.gain**2 * self.readnoise**2, 0.5)

    def _break_magnitude(self, bgsigma: float, fwhm: float) -> float:
        return -2.5 * np.log10(self.ape * np.pi / 4 * fwhm**2 * (bgsigma * self.gain)**2) + ZERO

    def _log_magerror(self, magnitude: float, bgsigma: float, fwhm: float) -> float:
        bm = self._break_magnitude(bgsigma, fwhm)
        return sbl(0.2, 0.4, 2.5, magnitude - bm) + 0.2 * bm - 2

    def calculate_exptime(self, target_magnitude: float, target_magerror: float,
                          fwhm: float, magzero_1s: float, sky_1s: float) -> float:
        """Required exposure time (seconds); magzero_1s is already zp_1s − 10."""
        target_log_magerror = np.log10(target_magerror)

        def equation(log_exptime):
            t = 10 ** log_exptime
            bgsigma = self.bgsigma_from_sky_brightness(sky_1s, t)
            magzero = magzero_1s + 2.5 * log_exptime
            mag_rel = target_magnitude - magzero
            return self._log_magerror(mag_rel, bgsigma, fwhm) - target_log_magerror

        try:
            return float(10 ** fsolve(equation, np.log10(300))[0])
        except Exception:
            return np.nan

    def predict_performance(self, target_magnitude: float, exptime: float,
                            fwhm: float, magzero_1s: float, sky_1s: float):
        """Return (magerror, snr) for the given exposure."""
        bgsigma  = self.bgsigma_from_sky_brightness(sky_1s, exptime)
        magzero  = magzero_1s + 2.5 * np.log10(exptime)
        mag_rel  = target_magnitude - magzero
        log_magerr = self._log_magerror(mag_rel, bgsigma, fwhm)
        magerr   = 10 ** log_magerr
        snr      = magerror_to_snr(magerr)
        return magerr, snr


if __name__ == '__main__':
    main()

# Example usage:
# python exposure_calculator.py 18.5 --refimg reference.ecsv --snr 10
# python exposure_calculator.py 18.5 2.1 --magzero 30.2 --sky 5.5 --snr 10
