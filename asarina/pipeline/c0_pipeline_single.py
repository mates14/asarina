#!/usr/bin/env python3

import argparse
import time
import subprocess
import logging
import os
import json
import socket
from pathlib import Path
from typing import Optional, List, Dict, Any
import tempfile
import shutil
from datetime import datetime
import threading
import queue
import signal
import sys
import os
import socket

from asarina.pipeline.get_ecsv import PhotometryPipeline
from asarina.pipeline.pipeline_utils import HealthChecker, PngCleaner, setup_logging

# Configure logging at module level
logger = logging.getLogger(__name__)

# Check if systemd is available
try:
    from systemd import daemon, journal
    HAS_SYSTEMD = True
except ImportError:
    HAS_SYSTEMD = False


class DatabaseUploader:
    """Handles uploading ECSV files to remote database."""
    
    def __init__(self,
                 ssh_key: str,
                 zeus_host: str = "zeus.asu.cas.cz",
                 hog_host: str = "hog.asu.cas.cz",
                 remote_script: str = "/home/mates/ecsv/mk-img.py",
                 local_phdb: str = "/home/mates/phdb",
                 remote_ecsv: str = "/home/mates/ecsv"):
        
        self.ssh_key = ssh_key
        self.zeus_host = zeus_host
        self.hog_host = hog_host
        self.remote_script = remote_script
        self.local_phdb = local_phdb
        self.remote_ecsv = remote_ecsv
    
    def upload_ecsv(self, ecsv_path: str, overwrite: bool = True) -> bool:
        """Upload ECSV file to database and trigger processing.
        
        Args:
            ecsv_path: Local path to ECSV file
            overwrite: Whether to overwrite existing database entries
            
        Returns:
            True if successful, False otherwise
        """
        ecsv_path = Path(ecsv_path)
        
        if not ecsv_path.exists():
            logger.error(f"ECSV file does not exist: {ecsv_path}")
            return False
        
        # Skip files with UNK filter
        if "UNK" in ecsv_path.name:
            logger.info(f"Skipping UNK filter file: {ecsv_path.name}")
            return False
        
        try:
            # Calculate remote path
            ecsv_name = ecsv_path.name
            relative_path = ecsv_path.parent.relative_to(self.local_phdb)
            remote_path = f"{self.remote_ecsv}/{relative_path}"
            
            logger.info(f"Uploading {ecsv_name} to {remote_path}")
            
            # Create remote directory
            mkdir_cmd = [
                "ssh", "-i", self.ssh_key, self.zeus_host,
                f"mkdir -p {remote_path}"
            ]
            result = subprocess.run(mkdir_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Failed to create remote directory: {result.stderr}")
                return False
            
            # Upload ECSV file
            scp_cmd = [
                "scp", "-i", self.ssh_key, str(ecsv_path),
                f"{self.zeus_host}:{remote_path}/"
            ]
            result = subprocess.run(scp_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Failed to upload ECSV: {result.stderr}")
                return False
            
            # Trigger database processing
            overwrite_flag = "-f" if overwrite else ""
            process_cmd = [
                "ssh", "-i", self.ssh_key, self.hog_host,
                self.remote_script, overwrite_flag, f"{remote_path}/{ecsv_name}"
            ]
            result = subprocess.run(process_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Failed to process in database: {result.stderr}")
                return False
            
            logger.info(f"Successfully uploaded and processed {ecsv_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading {ecsv_path}: {e}")
            return False


class TransientSearcher:
    """Handles transient detection on processed images."""

    def __init__(self, transients_script: str = "/home/fnovotny/bin/transients.py", 
            transient_user: str = "fnovotny"):
        self.transients_script = transients_script
        self.transient_user = transient_user

    def search_transients(self, ecsv_path: str, fits_path: str) -> bool:
        """Run transient search on ECSV file with corresponding FITS image.
        
        Args:
            ecsv_path: Path to ECSV catalog file
            fits_path: Path to calibrated FITS image file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect("/home/fnovotny/transient_daemon.sock")

            request = json.dumps({
                'ecsv_path': ecsv_path,
                'fits_path': fits_path
            })
            
            sock.send(request.encode())
            response_data = sock.recv(4096).decode()
            response = json.loads(response_data)
            
            sock.close()
            
            if response['success']:
                logger.info(f"Transient search triggered for {Path(ecsv_path).name}")
                return True
            else:
                logger.error(f"Transient search trigger failed: {response.get('error', 'Unknown error')}")
                return False
            
        except Exception as e:
            logger.error(f"Error communicating with transient daemon: {e}")
            return False

class ProcessingPipeline:
    """Complete processing pipeline from image to database."""
    
    def __init__(self,
                 cleanup_ecsv: bool = True,
                 run_transients: bool = False,
                 ssh_key: str = None,
                 **kwargs):

        self.photometry = PhotometryPipeline(**kwargs)
        self.uploader = DatabaseUploader(ssh_key=ssh_key)
        self.transient_searcher = TransientSearcher()
        self.cleanup_ecsv = cleanup_ecsv
        self.run_transients = run_transients
    
    def process_image(self, image_path: str, force: bool = False) -> bool:
        """Process single image through complete pipeline.
        
        Args:
            image_path: Path to input FITS image
            force: Force reprocessing even if ECSV exists
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Run photometry pipeline, keep image if trasients are to be searched
            result = self.photometry.process_image(image_path, force=force, keep_image=self.run_transients)
            if not result:
                logger.error(f"Photometry failed for {image_path}")
                return False
            
            # Handle return value (could be just ecsv_path or tuple)
            if isinstance(result, tuple):
                ecsv_path, fits_path = result
            else:
                # Backward compatibility - just ECSV path returned
                ecsv_path = result
                fits_path = None
            
            # Upload to database
            upload_success = self.uploader.upload_ecsv(ecsv_path, overwrite=force)
            if not upload_success:
                logger.error(f"Database upload failed for {ecsv_path}")
                return False
            
            # Run transient search if enabled
            if self.run_transients and fits_path:
                transient_success = self.transient_searcher.search_transients(ecsv_path, fits_path)
                if not transient_success:
                    logger.warning(f"Transient search failed for {ecsv_path}")
                    # Don't return False - transient search failure shouldn't stop pipeline
                
                # Clean up the FITS file after transient search
                try:
                    Path(fits_path).unlink()
                    logger.debug(f"Cleaned up temporary FITS file: {fits_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up FITS file {fits_path}: {e}")
            
            # Clean up ECSV file if requested
            if self.cleanup_ecsv:
                Path(ecsv_path).unlink()
                logger.info(f"Cleaned up {ecsv_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline error for {image_path}: {e}")
            return False


class FileMonitor:
    """Monitors filesystem for new images to process."""
    
    def __init__(self, 
                 search_root: str = "/images",
                 camera_pattern: str = "C0",
                 file_pattern: str = "2*.fits",
                 exclude_patterns: List[str] = None,
                 poll_interval: float = 1.0):
        
        self.search_root = Path(search_root)
        self.camera_pattern = camera_pattern
        self.file_pattern = file_pattern
        self.exclude_patterns = exclude_patterns or ["dark", "flat", "bad", "queue"]
        self.poll_interval = poll_interval
        self.last_file = None
        self.running = False

    def find_latest_file(self) -> Optional[str]:
        """Find the most recent file matching criteria."""
        try:
            # Find latest year directory
            year_dirs = list(self.search_root.glob("2???"))
            if not year_dirs:
                logger.debug(f"No year directories found in {self.search_root}")
                return None
            latest_year = max(year_dirs, key=lambda p: p.name)
            logger.debug(f"Latest year directory: {latest_year}")
            
            # Find latest date directory in that year
            date_dirs = list(latest_year.glob("20??????"))
            if not date_dirs:
                logger.debug(f"No date directories found in {latest_year}")
                return None
            latest_date = max(date_dirs, key=lambda p: p.name)
            logger.debug(f"Latest date directory: {latest_date}")
            
            # Find latest file in camera subdirectory
            camera_path = latest_date / self.camera_pattern
            fits_files = []
            
            if camera_path.exists():
                # Direct camera directory exists - search it and subdirectories
                fits_files.extend(camera_path.rglob(self.file_pattern))
                logger.debug(f"Searching in {camera_path} and subdirectories")
            else:
                # No direct camera directory - search all subdirectories for camera pattern
                logger.debug(f"Camera directory {camera_path} does not exist, searching subdirectories")
                for subdir in latest_date.iterdir():
                    if subdir.is_dir():
                        camera_subpath = subdir / self.camera_pattern
                        if camera_subpath.exists():
                            fits_files.extend(camera_subpath.rglob(self.file_pattern))
                            logger.debug(f"Found camera directory: {camera_subpath}")
            
            if not fits_files:
                logger.debug(f"No FITS files found matching pattern '{self.file_pattern}'")
                return None
            
            logger.debug(f"Found {len(fits_files)} FITS files in {camera_path}")
            
            # Filter out excluded patterns
            filtered_files = []
            for f in fits_files:
                if not any(pattern in str(f) for pattern in self.exclude_patterns):
                    filtered_files.append(f)
                else:
                    logger.debug(f"Excluded file: {f}")
            
            if not filtered_files:
                logger.debug("No files remaining after filtering")
                return None
            
            logger.debug(f"After filtering: {len(filtered_files)} files")
            
            # Return most recent by modification time
            latest_file = max(filtered_files, key=lambda p: p.stat().st_mtime)
            return str(latest_file)
            
        except Exception as e:
            logger.error(f"Error finding latest file: {e}")
            return None
    
    def monitor(self, process_queue: queue.Queue):
        """Monitor for new files and add them to processing queue."""
        self.running = True
        logger.info("Starting file monitoring...")
        
        while self.running:
            try:
                latest_file = self.find_latest_file()
                
                if latest_file and latest_file != self.last_file:
                    logger.info(f"New file detected: {latest_file}")
                    process_queue.put(latest_file)
                    self.last_file = latest_file
                
                time.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(self.poll_interval)
    
    def stop(self):
        """Stop monitoring."""
        self.running = False


class AutomatedPipeline:
    """Main automated processing system with systemd integration."""
    
    def __init__(self, use_systemd=False, png_cleanup_days=7, cleanup_interval_hours=6, **kwargs):
        self.pipeline = ProcessingPipeline(**kwargs)
        self.monitor = FileMonitor()
        self.process_queue = queue.Queue()
        self.health_checker = HealthChecker()
        self.png_cleaner = PngCleaner(max_age_days=png_cleanup_days)
        self.use_systemd = use_systemd and HAS_SYSTEMD
        self.cleanup_interval = cleanup_interval_hours * 3600  # Convert to seconds
        self.last_cleanup = 0
        self.running = False
        self.logger = logging.getLogger(__name__)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Setup systemd watchdog if available
        if self.use_systemd:
            self.watchdog_interval = self._get_watchdog_interval()
        else:
            self.watchdog_interval = None
    
    def _get_watchdog_interval(self) -> Optional[float]:
        """Get systemd watchdog interval."""
        if not self.use_systemd:
            return None
        
        watchdog_usec = os.environ.get('WATCHDOG_USEC')
        if watchdog_usec:
            # Return half the watchdog timeout for safe margin
            return int(watchdog_usec) / 2000000.0  # Convert microseconds to seconds
        return None
    
    def _notify_systemd_ready(self):
        """Notify systemd that service is ready."""
        if self.use_systemd:
            daemon.notify('READY=1')
            self.logger.info("Notified systemd: service ready")
    
    def _notify_systemd_status(self, status: str):
        """Update systemd status."""
        if self.use_systemd:
            daemon.notify(f'STATUS={status}')
    
    def _send_watchdog_ping(self):
        """Send watchdog ping to systemd."""
        if self.use_systemd:
            daemon.notify('WATCHDOG=1')
    
    def _check_and_run_cleanup(self):
        """Check if it's time to run PNG cleanup and do it if needed."""
        current_time = time.time()
        
        if current_time - self.last_cleanup > self.cleanup_interval:
            self.logger.info("Running scheduled PNG cleanup...")
            self._notify_systemd_status("Cleaning up old PNG files...")
            
            cleanup_stats = self.png_cleaner.cleanup_old_pngs()
            self.last_cleanup = current_time
            
            if 'error' in cleanup_stats:
                self.logger.error(f"PNG cleanup failed: {cleanup_stats['error']}")
            elif cleanup_stats['deleted_files'] > 0:
                self.logger.info(f"PNG cleanup completed: {cleanup_stats}")
            
            return cleanup_stats
        
        return None
        """Send watchdog ping to systemd."""
        if self.use_systemd:
            daemon.notify('WATCHDOG=1')
    
    def _watchdog_worker(self):
        """Worker thread for systemd watchdog pings."""
        if not self.watchdog_interval:
            return
        
        while self.running:
            time.sleep(self.watchdog_interval)
            if self.running and self.health_checker.is_healthy():
                self._send_watchdog_ping()
            elif self.running:
                self.logger.warning("Service unhealthy, skipping watchdog ping")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Shutdown signal {signum} received, stopping...")
        self._notify_systemd_status("Shutting down...")
        self.stop()
    
    def _processor_worker(self):
        """Worker thread that processes images from the queue."""
        while self.running:
            try:
                # Get image path from queue with timeout
                image_path = self.process_queue.get(timeout=1.0)
                
                self.logger.info(f"Processing {image_path}")
                self._notify_systemd_status(f"Processing {Path(image_path).name}")
                
                success = self.pipeline.process_image(image_path)
                
                if success:
                    self.logger.info(f"Successfully processed {image_path}")
                    self.health_checker.record_activity()
                else:
                    self.logger.error(f"Failed to process {image_path}")
                    self.health_checker.record_error()
                
                self.process_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Processor worker error: {e}")
                self.health_checker.record_error()
    
    def run(self):
        """Run the automated pipeline."""
        self.running = True
        
        # Log startup information
        hostname = socket.gethostname()
        self.logger.info(f"Starting automated pipeline on {hostname}")
        self.logger.info(f"PNG cleanup: every {self.cleanup_interval/3600:.1f} hours, "
                        f"files older than {self.png_cleaner.max_age_seconds/(24*3600)} days")
        
        status = self.health_checker.get_status()
        self.logger.info(f"Service status: {status}")
        
        # Run initial cleanup
        self.logger.info("Running initial PNG cleanup...")
        cleanup_stats = self.png_cleaner.cleanup_old_pngs()
        if 'error' not in cleanup_stats and cleanup_stats['deleted_files'] > 0:
            self.logger.info(f"Initial cleanup: {cleanup_stats}")
        self.last_cleanup = time.time()
        
        # Start watchdog worker if systemd is available
        if self.watchdog_interval:
            watchdog_thread = threading.Thread(target=self._watchdog_worker, daemon=True)
            watchdog_thread.start()
            self.logger.info(f"Started watchdog with {self.watchdog_interval}s interval")
        else:
            # If no systemd watchdog, still run periodic cleanup
            cleanup_thread = threading.Thread(target=self._periodic_cleanup_worker, daemon=True)
            cleanup_thread.start()
            self.logger.info(f"Started cleanup worker with {self.cleanup_interval/3600:.1f}h interval")
        
        # Start processor worker thread
        processor_thread = threading.Thread(target=self._processor_worker, daemon=True)
        processor_thread.start()
        
        # Notify systemd that we're ready
        self._notify_systemd_ready()
        self._notify_systemd_status("Monitoring for new images...")
        
        # Start monitoring in main thread
        try:
            self.monitor.monitor(self.process_queue)
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        finally:
            self.stop()
    
    def _periodic_cleanup_worker(self):
        """Worker thread for periodic cleanup when no systemd watchdog."""
        while self.running:
            time.sleep(self.cleanup_interval)
            if self.running:
                self._check_and_run_cleanup()
    
    def stop(self):
        """Stop the pipeline."""
        self.running = False
        self.monitor.stop()
        
        if self.use_systemd:
            daemon.notify('STOPPING=1')
        
        # Log final statistics
        status = self.health_checker.get_status()
        self.logger.info(f"Final statistics: {status}")
        self.logger.info("Pipeline stopped")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status for monitoring."""
        return self.health_checker.get_status()


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Automated astronomical image processing pipeline"
    )
    parser.add_argument('--search-root', default='/images',
                       help='Root directory to search for images')
    parser.add_argument('--camera-pattern', default='C0',
                       help='Camera subdirectory pattern')
    parser.add_argument('--poll-interval', type=float, default=1.0,
                       help='File monitoring poll interval in seconds')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Keep ECSV files after upload')
    parser.add_argument('--run-transients', action='store_true',
                       help='Run transient search on processed images')
    parser.add_argument('--png-cleanup-days', type=int, default=7,
                       help='Delete PNG files older than this many days (default: 7)')
    parser.add_argument('--cleanup-interval-hours', type=float, default=6,
                       help='How often to run PNG cleanup in hours (default: 6)')
    parser.add_argument('--systemd', action='store_true',
                       help='Enable systemd integration (journal logging, watchdog, notifications)')
    parser.add_argument('--health-check', action='store_true',
                       help='Print health status and exit')
    parser.add_argument('--ssh-key', required=True,
                       help='Path to SSH private key for database upload')
    parser.add_argument('--fast', action='store_true',
                       help='Prefer being less precise and faster')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Setup logging
    setup_logging(use_systemd=args.systemd, verbose=args.verbose)

    if args.verbose:
        logger.debug("Debug logging enabled")

    # Handle health check
    if args.health_check:
        # This could be expanded to check a running instance
        print("Service health check not implemented for standalone execution")
        sys.exit(0)

    # Create and run pipeline
    pipeline = AutomatedPipeline(
        use_systemd=args.systemd,
        png_cleanup_days=args.png_cleanup_days,
        cleanup_interval_hours=args.cleanup_interval_hours,
        cleanup_ecsv=not args.no_cleanup,
        run_transients=args.run_transients,
        ssh_key=args.ssh_key
#        fast=args.fast
    )
    
    # Update monitor settings
    pipeline.monitor.search_root = Path(args.search_root)
    pipeline.monitor.camera_pattern = args.camera_pattern
    pipeline.monitor.poll_interval = args.poll_interval
    
    # Update transient searcher
    # pipeline.pipeline.transient_searcher.transients_script = args.transients_script
    
    logger.info("Starting automated pipeline...")
    if args.systemd:
        logger.info("Systemd integration enabled")
    
    pipeline.run()


if __name__ == "__main__":
    main()
