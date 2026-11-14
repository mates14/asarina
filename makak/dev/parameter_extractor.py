#!/usr/bin/env python3
"""
Photometric Parameter Extractor
Extracts robust polynomial fitting parameters for detailed analysis of photometric data.
Outputs A, B, C coefficients, RMS values, and astronomical context.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon
from astropy import units as u
import warnings
warnings.filterwarnings('ignore')

class ParameterExtractor:
    def __init__(self, 
                 window_minutes=5,
                 observatory_location=None):
        """
        Parameters:
        - window_minutes: Window size for polynomial fitting
        - observatory_location: EarthLocation object for astronomical calculations
                               If None, will use a default location (you should specify your observatory)
        """
        self.window_minutes = window_minutes
        
        # Default observatory location - PLEASE CHANGE THIS TO YOUR OBSERVATORY
        if observatory_location is None:
            # Example: Paranal Observatory coordinates
            # Replace with your observatory coordinates
            self.observatory = EarthLocation(
                lat=49.9093889 * u.deg,    # Replace with your latitude
                lon=14.7813631 * u.deg,    # Replace with your longitude  
                height=500 * u.m        # Replace with your altitude
            )
            print("Warning: Using default observatory location. Please set your coordinates!")
            print(f"Current default: lat={self.observatory.lat}, lon={self.observatory.lon}")
        else:
            self.observatory = observatory_location
    
    def load_photometric_data(self, filename):
        """Load photometric data file."""
        columns = ['timestamp', 'jd', 'jd2', 'filter', 'exptime', 'duration', 
                  'nstars', 'zeropoint', 'zp_error', 'extinction', 'airmass',
                  'mag_limit', 'mag_limit2', 'sky_brightness', 'ccd_temp', 'status']
        
        df = pd.read_csv(filename, sep=r'\s+', names=columns, comment='#')
        astropy_times = Time(df['jd'], format='jd')
        df['datetime'] = pd.to_datetime(astropy_times.datetime)
        df['astropy_time'] = astropy_times
        
        return df
    
    def load_cloudmeter_data(self, filename):
        """Load cloudmeter data."""
        try:
            cloud_df = pd.read_csv(filename, sep=r'\s+', names=['timestamp', 'cloud_temp'])
            cloud_df['datetime'] = pd.to_datetime(cloud_df['timestamp'], unit='s')
            # Convert to astropy Time for astronomical calculations
            cloud_df['astropy_time'] = Time(cloud_df['datetime'])
            return cloud_df
        except:
            return None
    
    def parabolic_function(self, x, A, B, C):
        """Parabolic function: A*x^2 + B*x + C"""
        return A * x**2 + B * x + C
    
    def robust_parabolic_fit(self, times, values, max_iterations=3):
        """
        Robust parabolic fitting with iterative outlier rejection.
        
        Returns:
        - fit_params: [A, B, C] coefficients
        - rms: RMS of the fit residuals  
        - n_good: Number of points used in final fit
        - n_total: Total number of input points
        """
        if len(values) < 4:
            return None, np.nan, 0, len(values)
        
        # Normalize time to center of interval (in hours from center)
        t_center = np.mean(times)
        t_norm = (times - t_center) / 3600.0
        
        mask = np.isfinite(values)
        
        for iteration in range(max_iterations):
            if np.sum(mask) < 4:
                break
                
            try:
                # Fit parabola to current good points
                popt, _ = curve_fit(self.parabolic_function, t_norm[mask], values[mask])
                
                # Calculate residuals for all points
                fitted = self.parabolic_function(t_norm, *popt)
                residuals = values - fitted
                
                # Robust outlier detection using MAD
                mad = stats.median_abs_deviation(residuals[mask])
                if mad == 0:  # All residuals are identical
                    break
                threshold = 3.0 * mad
                
                # Update mask to exclude outliers
                new_mask = mask & (np.abs(residuals) <= threshold)
                
                # Stop if no points were rejected
                if np.array_equal(mask, new_mask):
                    break
                    
                mask = new_mask
                
            except (RuntimeError, ValueError):
                return None, np.nan, 0, len(values)
        
        if np.sum(mask) < 4:
            return None, np.nan, 0, len(values)
        
        # Final fit with good points
        try:
            popt, _ = curve_fit(self.parabolic_function, t_norm[mask], values[mask])
            fitted = self.parabolic_function(t_norm[mask], *popt)
            rms = np.sqrt(np.mean((values[mask] - fitted)**2))
            
            # Parameters: A (curvature), B (trend per hour), C (value at center)
            A, B, C = popt
            
            return [A, B, C], rms, np.sum(mask), len(values)
            
        except (RuntimeError, ValueError):
            return None, np.nan, 0, len(values)
    
    def calculate_astronomical_parameters(self, time_astropy):
        """Calculate sun altitude, moon altitude, and moon phase."""
        
        # Create AltAz frame for the observatory and time
        altaz_frame = AltAz(obstime=time_astropy, location=self.observatory)
        
        # Get sun position
        sun = get_sun(time_astropy)
        sun_altaz = sun.transform_to(altaz_frame)
        sun_altitude = sun_altaz.alt.degree
        
        # Get moon position
        moon = get_moon(time_astropy)
        moon_altaz = moon.transform_to(altaz_frame)
        moon_altitude = moon_altaz.alt.degree
        
        # Calculate moon phase (0 = new moon, 1 = full moon)
        # This is an approximation based on sun-moon angular separation
        sun_moon_sep = sun.separation(moon).degree
        # Phase calculation: 0 at 180° separation (full), 1 at 0° separation (new)
        moon_phase = (1 - np.cos(np.radians(sun_moon_sep))) / 2
        
        return sun_altitude, moon_altitude, moon_phase
    
    def extract_parameters(self, phot_df, cloud_df=None, output_interval=5):
        """
        Extract robust fitting parameters for all data types.
        
        Returns DataFrame with all parameters for each time interval.
        """
        
        # Filter obviously bad photometric data
        good_mask = (
            np.isfinite(phot_df['zeropoint']) & 
            np.isfinite(phot_df['sky_brightness']) &
            (phot_df['zp_error'] < 0.5) &
            (phot_df['zeropoint'] > 3) & (phot_df['zeropoint'] < 15)
        )
        
        df_clean = phot_df[good_mask].copy().sort_values('datetime')
        
        if len(df_clean) < 10:
            print("Insufficient good data points")
            return None
        
        # Create output time grid
        start_time = df_clean['datetime'].min()
        end_time = df_clean['datetime'].max()
        output_times = pd.date_range(start_time, end_time, freq=f'{output_interval}min')
        
        # Initialize output lists
        results = []
        window_seconds = self.window_minutes * 60
        
        print(f"Processing {len(output_times)} time intervals...")
        
        for i, out_time in enumerate(output_times):
            if i % 50 == 0:
                print(f"  Processing interval {i+1}/{len(output_times)}")
                
            out_time_np = out_time.to_numpy()
            
            # Create result dictionary for this time interval
            result = {
                'datetime': out_time,
                'jd': Time(out_time).jd
            }
            
            # === PHOTOMETRIC DATA ANALYSIS ===
            # Define window around output time for photometric data
            times_phot = df_clean['datetime'].values
            time_diffs_phot = np.abs(times_phot - out_time_np) / np.timedelta64(1, 's')
            window_mask_phot = time_diffs_phot <= window_seconds / 2
            
            if np.sum(window_mask_phot) >= 4:
                # Zeropoint analysis
                window_zp = df_clean.loc[window_mask_phot, 'zeropoint'].values
                zp_params, zp_rms, zp_n_good, zp_n_total = self.robust_parabolic_fit(
                    time_diffs_phot[window_mask_phot], window_zp
                )
                
                if zp_params is not None:
                    result.update({
                        'zp_A': zp_params[0], 'zp_B': zp_params[1], 'zp_C': zp_params[2],
                        'zp_rms': zp_rms, 'zp_n_good': zp_n_good, 'zp_n_total': zp_n_total
                    })
                else:
                    result.update({
                        'zp_A': np.nan, 'zp_B': np.nan, 'zp_C': np.nan,
                        'zp_rms': np.nan, 'zp_n_good': 0, 'zp_n_total': zp_n_total
                    })
                
                # Sky brightness analysis
                window_sky = df_clean.loc[window_mask_phot, 'sky_brightness'].values
                sky_params, sky_rms, sky_n_good, sky_n_total = self.robust_parabolic_fit(
                    time_diffs_phot[window_mask_phot], window_sky
                )
                
                if sky_params is not None:
                    result.update({
                        'sky_A': sky_params[0], 'sky_B': sky_params[1], 'sky_C': sky_params[2],
                        'sky_rms': sky_rms, 'sky_n_good': sky_n_good, 'sky_n_total': sky_n_total
                    })
                else:
                    result.update({
                        'sky_A': np.nan, 'sky_B': np.nan, 'sky_C': np.nan,
                        'sky_rms': np.nan, 'sky_n_good': 0, 'sky_n_total': sky_n_total
                    })
            else:
                # No photometric data in window
                result.update({
                    'zp_A': np.nan, 'zp_B': np.nan, 'zp_C': np.nan,
                    'zp_rms': np.nan, 'zp_n_good': 0, 'zp_n_total': 0,
                    'sky_A': np.nan, 'sky_B': np.nan, 'sky_C': np.nan,
                    'sky_rms': np.nan, 'sky_n_good': 0, 'sky_n_total': 0
                })
            
            # === CLOUDMETER DATA ANALYSIS ===
            if cloud_df is not None:
                times_cloud = cloud_df['datetime'].values
                time_diffs_cloud = np.abs(times_cloud - out_time_np) / np.timedelta64(1, 's')
                window_mask_cloud = time_diffs_cloud <= window_seconds / 2
                
                if np.sum(window_mask_cloud) >= 4:
                    window_cloud = cloud_df.loc[window_mask_cloud, 'cloud_temp'].values
                    cloud_params, cloud_rms, cloud_n_good, cloud_n_total = self.robust_parabolic_fit(
                        time_diffs_cloud[window_mask_cloud], window_cloud
                    )
                    
                    if cloud_params is not None:
                        result.update({
                            'cloud_A': cloud_params[0], 'cloud_B': cloud_params[1], 'cloud_C': cloud_params[2],
                            'cloud_rms': cloud_rms, 'cloud_n_good': cloud_n_good, 'cloud_n_total': cloud_n_total
                        })
                    else:
                        result.update({
                            'cloud_A': np.nan, 'cloud_B': np.nan, 'cloud_C': np.nan,
                            'cloud_rms': np.nan, 'cloud_n_good': 0, 'cloud_n_total': cloud_n_total
                        })
                else:
                    result.update({
                        'cloud_A': np.nan, 'cloud_B': np.nan, 'cloud_C': np.nan,
                        'cloud_rms': np.nan, 'cloud_n_good': 0, 'cloud_n_total': 0
                    })
            else:
                result.update({
                    'cloud_A': np.nan, 'cloud_B': np.nan, 'cloud_C': np.nan,
                    'cloud_rms': np.nan, 'cloud_n_good': 0, 'cloud_n_total': 0
                })
            
            # === ASTRONOMICAL CALCULATIONS ===
            try:
                time_astropy = Time(out_time)
                sun_alt, moon_alt, moon_phase = self.calculate_astronomical_parameters(time_astropy)
                result.update({
                    'sun_altitude': sun_alt,
                    'moon_altitude': moon_alt,
                    'moon_phase': moon_phase
                })
            except Exception as e:
                print(f"Warning: Could not calculate astronomical parameters for {out_time}: {e}")
                result.update({
                    'sun_altitude': np.nan,
                    'moon_altitude': np.nan,
                    'moon_phase': np.nan
                })
            
            results.append(result)
        
        return pd.DataFrame(results)

def save_parameters_file(df, filename):
    """Save parameters to a well-formatted text file."""
    
    with open(filename, 'w') as f:
        # Write header with column descriptions
        f.write("# Photometric Parameter Analysis\n")
        f.write("# Generated by Parameter Extractor\n")
        f.write("#\n")
        f.write("# Column descriptions:\n")
        f.write("# datetime: Time of analysis window center\n")
        f.write("# jd: Julian Date\n")
        f.write("# zp_A, zp_B, zp_C: Zeropoint parabolic fit coefficients (A*t^2 + B*t + C)\n")
        f.write("#   A: curvature (mag/hr^2), B: trend (mag/hr), C: value at center (mag)\n")
        f.write("# zp_rms: RMS of zeropoint fit (mag)\n")
        f.write("# zp_n_good/zp_n_total: Number of good/total points in fit\n")
        f.write("# sky_*: Same parameters for sky brightness\n")
        f.write("# cloud_*: Same parameters for cloudmeter data\n")
        f.write("# sun_altitude: Sun altitude (degrees, negative at night)\n")
        f.write("# moon_altitude: Moon altitude (degrees)\n")
        f.write("# moon_phase: Moon phase (0=new, 0.5=half, 1=full)\n")
        f.write("#\n")
        
        # Write column headers
        columns = [
            'datetime', 'jd',
            'zp_A', 'zp_B', 'zp_C', 'zp_rms', 'zp_n_good', 'zp_n_total',
            'sky_A', 'sky_B', 'sky_C', 'sky_rms', 'sky_n_good', 'sky_n_total',
            'cloud_A', 'cloud_B', 'cloud_C', 'cloud_rms', 'cloud_n_good', 'cloud_n_total',
            'sun_altitude', 'moon_altitude', 'moon_phase'
        ]
        
        # Format header line
        header_line = " ".join(f"{col:>12}" for col in columns)
        f.write(header_line + "\n")
        
        # Write data
        for _, row in df.iterrows():
            values = []
            for col in columns:
                if col == 'datetime':
                    values.append(f"{row[col].strftime('%Y-%m-%d_%H:%M:%S'):>12}")
                elif col in ['zp_n_good', 'zp_n_total', 'sky_n_good', 'sky_n_total', 'cloud_n_good', 'cloud_n_total']:
                    values.append(f"{int(row[col]) if not np.isnan(row[col]) else 0:>12}")
                elif col == 'jd':
                    values.append(f"{row[col]:>12.6f}")
                else:
                    if np.isnan(row[col]):
                        values.append(f"{'NaN':>12}")
                    else:
                        values.append(f"{row[col]:>12.6f}")
            
            f.write(" ".join(values) + "\n")

# Command line usage
if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("Usage: python parameter_extractor.py <photometric_data_file> [cloudmeter_data_file]")
        print("Example: python parameter_extractor.py mr20220322.dat cloud20220322.dat")
        print("\nNote: Please edit the script to set your observatory coordinates!")
        sys.exit(1)
    
    phot_file = sys.argv[1]
    cloud_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(phot_file):
        print(f"Error: Photometric data file '{phot_file}' not found.")
        sys.exit(1)
    
    if cloud_file and not os.path.exists(cloud_file):
        print(f"Warning: Cloudmeter data file '{cloud_file}' not found. Proceeding without cloudmeter data.")
        cloud_file = None
    
    # Initialize extractor
    print("Initializing parameter extractor...")
    print("WARNING: Using default observatory coordinates. Please edit the script to set your location!")
    
    extractor = ParameterExtractor(window_minutes=5)
    
    try:
        # Load data
        print(f"Loading photometric data from {phot_file}...")
        phot_df = extractor.load_photometric_data(phot_file)
        print(f"Loaded {len(phot_df)} photometric data points")
        
        cloud_df = None
        if cloud_file:
            print(f"Loading cloudmeter data from {cloud_file}...")
            cloud_df = extractor.load_cloudmeter_data(cloud_file)
            if cloud_df is not None:
                print(f"Loaded {len(cloud_df)} cloudmeter data points")
        
        # Extract parameters
        print("Extracting robust polynomial parameters...")
        params_df = extractor.extract_parameters(phot_df, cloud_df, output_interval=5)
        
        if params_df is None:
            print("Error: Insufficient data for analysis")
            sys.exit(1)
        
        # Save results
        base_name = os.path.splitext(phot_file)[0]
        output_file = f"{base_name}_parameters.txt"
        
        print(f"Saving parameters to {output_file}...")
        save_parameters_file(params_df, output_file)
        
        # Also save as CSV for easier analysis
        # csv_file = f"{base_name}_parameters.csv"
        # params_df.to_csv(csv_file, index=False)
        
        # Print summary
        print(f"\n=== Parameter Extraction Summary ===")
        print(f"Analysis period: {params_df['datetime'].min()} to {params_df['datetime'].max()}")
        print(f"Total intervals: {len(params_df)}")
        
        # Statistics for key parameters
        good_zp = params_df[params_df['zp_rms'] <= 0.02]
        print(f"Intervals with zp_rms ≤ 0.02 mag: {len(good_zp)} ({len(good_zp)/len(params_df)*100:.1f}%)")
        
        if not params_df['zp_C'].isna().all():
            print(f"Zeropoint range: {params_df['zp_C'].min():.3f} to {params_df['zp_C'].max():.3f} mag")
            print(f"Zeropoint mean: {params_df['zp_C'].mean():.3f} ± {params_df['zp_C'].std():.3f} mag")
        
        if not params_df['sun_altitude'].isna().all():
            print(f"Sun altitude range: {params_df['sun_altitude'].min():.1f}° to {params_df['sun_altitude'].max():.1f}°")
        
        print(f"\nOutput files:")
        print(f"  {output_file} (formatted text)")
#        print(f"  {csv_file} (CSV format)")
        
        print(f"\nParameters ready for analysis!")
        print("You can now examine the A, B, C coefficients and RMS values to understand")
        print("the behavior of your photometric system under different conditions.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
