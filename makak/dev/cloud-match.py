import sys
from datetime import datetime, timezone
from astropy.time import Time
import numpy as np

def parse_photometry_line(line):
    """Parse a line from the photometry file."""
    parts = line.strip().split()
    try:
        jd = float(parts[1])  # JD timestamp
        zp = float(parts[7])   # Zeropoint
        zp_err = float(parts[8])  # Zeropoint error
        bgnoise = float(parts[13])  # bg noise
        ccdtemp = float(parts[14])  # ccd temperature
        return jd, zp, zp_err, bgnoise, ccdtemp
    except (IndexError, ValueError):
        return None

def parse_cloud_line(line):
    """Parse a CLOUD line from the meteo log."""
    if not line.startswith('CLOUD '):
        return None
    
    parts = line.strip().split()
    try:
        # Parse timestamp
        date_str = f"{parts[1]} {parts[2]}"
        dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')
        dt = dt.replace(tzinfo=timezone.utc)
        
        # Convert to JD using astropy
        t = Time(dt)
        jd = t.jd
        
        # Get IR readings (7-13th columns after the timestamp)
        ir_readings = [float(x) for x in parts[6:13]]
        
        return jd, ir_readings
    except (IndexError, ValueError):
        return None

def find_nearest_cloud(cloud_data, target_jd, max_time_diff=3/1440.0):  # max 3 minute difference
    """Find the nearest cloud measurement to a given JD timestamp."""
    if not cloud_data:
        return None
    
    cloud_times = np.array([x[0] for x in cloud_data])
    idx = np.argmin(np.abs(cloud_times - target_jd))
    
    if abs(cloud_times[idx] - target_jd) > max_time_diff:
        return None
        
    return cloud_data[idx]

def process_files(phot_file, meteo_file, output_file):
    """Process both files and write matched measurements."""
    # Read all cloud measurements first
    cloud_data = []
    with open(meteo_file, 'r') as f:
        for line in f:
            result = parse_cloud_line(line)
            if result:
                cloud_data.append(result)
    
    # Process photometry and match with cloud data
    with open(phot_file, 'r') as f_in, open(output_file, 'w') as f_out:
        # Write header
        f_out.write("# JD ZeroPoint ZP_Error IR1 IR2 IR3 IR4 IR5 IR6 IR7\n")
        
        for line in f_in:
            phot_result = parse_photometry_line(line)
            if not phot_result:
                continue
                
            jd, zp, zp_err, bg, temp = phot_result
            cloud_match = find_nearest_cloud(cloud_data, jd)
            
            if cloud_match:
                cloud_jd, ir_readings = cloud_match
                output = f"{jd:.6f} {zp:.3f} {zp_err:.3f} {bg:.3f} {temp:.3f} " + \
                        " ".join(f"{x:.3f}" for x in ir_readings)
                f_out.write(output + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py photometry.dat meteo.log output.txt")
        sys.exit(1)
        
    process_files(sys.argv[1], sys.argv[2], sys.argv[3])
