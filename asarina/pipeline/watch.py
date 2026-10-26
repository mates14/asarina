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
from concurrent.futures import ThreadPoolExecutor

from asarina.pipeline.pipeline_utils import HealthChecker, PngCleaner, DatabaseUploader, setup_logging

# Configure logging at module level
logger = logging.getLogger(__name__)

# Check if systemd is available
try:
    from systemd import daemon, journal
    HAS_SYSTEMD = True
except ImportError:
    HAS_SYSTEMD = False




class FileMonitor:
    """Monitors filesystem for new images to process."""
    
    def __init__(self, 
                 search_root: str = "/images",
                 camera_pattern: str = "C0",
                 file_pattern: str = "2*.fits",
                 exclude_patterns: List[str] = None,
                 poll_interval: float = 1.0,
                 min_age: float = 0.0):
        
        self.search_root = Path(search_root)
        # Seconds a file must sit unmodified before it is taken: where the
        # camera writes straight into the archive (Makak), the newest file may
        # still be growing.
        self.min_age = min_age
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
            
            if self.min_age > 0:
                settled = time.time() - self.min_age
                filtered_files = [f for f in filtered_files if f.stat().st_mtime <= settled]
                if not filtered_files:
                    return None

            # Return most recent by modification time
            latest_file = max(filtered_files, key=lambda p: p.stat().st_mtime)
            logger.debug(f"Latest file: {latest_file}")
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
    """File-watching daemon: detects new images and hands each to asarina-imgproc."""

    def __init__(self, use_systemd=False, png_cleanup_days=7, cleanup_interval_hours=6,
                 max_workers=3, imgproc_base=None, latest_only=False, frame_timeout=None):
        self.imgproc_base = imgproc_base or ['asarina-imgproc']
        # latest_only: with all workers busy, a new frame replaces the one
        # waiting instead of queueing behind it.  For a camera whose cadence
        # outruns the processing (Makak, 10 s frames) the queue would otherwise
        # only grow, and the night's results lag further behind every hour.
        self.latest_only = latest_only
        self.frame_timeout = frame_timeout
        self._slot_lock = threading.Lock()
        self._busy = 0
        self._pending = None
        self.monitor = FileMonitor()
        self.process_queue = queue.Queue()
        self.health_checker = HealthChecker()
        self.png_cleaner = PngCleaner(max_age_days=png_cleanup_days)
        self.use_systemd = use_systemd and HAS_SYSTEMD
        self.cleanup_interval = cleanup_interval_hours * 3600  # Convert to seconds
        self.last_cleanup = 0
        self.running = False
        self.logger = logging.getLogger(__name__)
        
        # Parallel processing
        self.max_workers = max_workers
        self.processing_pool = None
        
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
        """Worker thread that processes images from the queue using thread pool."""
        while self.running:
            try:
                # Get image path from queue with timeout
                image_path = self.process_queue.get(timeout=1.0)
                
                self.logger.info(f"Queuing {Path(image_path).name} (queue size: {self.process_queue.qsize()})")
                self._notify_systemd_status(f"Processing {Path(image_path).name}")
                
                # Submit to processing pool
                if self.latest_only:
                    with self._slot_lock:
                        if self._busy >= self.max_workers:
                            if self._pending is not None:
                                self.logger.info(f"Skipping {Path(self._pending).name}, "
                                                 f"superseded by {Path(image_path).name}")
                            self._pending = image_path
                            self.process_queue.task_done()
                            continue
                        self._busy += 1
                    self.processing_pool.submit(self._run_slot, image_path)
                else:
                    self.processing_pool.submit(self._process_single_image, image_path)
                
                # Mark task as received (actual processing happens in thread pool)
                self.process_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Processor worker error: {e}")
                self.health_checker.record_error()
    
    def _run_slot(self, image_path: str):
        """latest_only worker: process the frame, then whichever one is waiting."""
        while image_path is not None:
            self._process_single_image(image_path)
            with self._slot_lock:
                image_path, self._pending = self._pending, None
                if image_path is None:
                    self._busy -= 1

    def _process_single_image(self, image_path: str):
        """Process a single image by invoking asarina-imgproc (runs in thread pool)."""
        cmd = self.imgproc_base + [image_path]
        t = time.time()
        try:
            # Own session, so that a timeout takes imgproc's children
            # (pyrt-field-solve on a cloudy frame) down with it.
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True,
                                    start_new_session=True)
            try:
                _, stderr = proc.communicate(timeout=self.frame_timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    _, stderr = proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    _, stderr = proc.communicate()
                self.logger.error(f"✗ Timeout {Path(image_path).name} "
                                  f"after {self.frame_timeout:.0f}s")
                self.health_checker.record_error()
                return
            if proc.returncode == 0:
                self.logger.info(f"✓ Completed {Path(image_path).name} ({time.time() - t:.1f}s)")
                self.health_checker.record_activity()
            else:
                self.logger.error(f"✗ Failed {Path(image_path).name} (exit {proc.returncode})")
                for line in stderr.splitlines()[-10:]:
                    self.logger.error(f"  {line}")
                self.health_checker.record_error()
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")
            self.health_checker.record_error()
    
    def run(self):
        """Run the automated pipeline."""
        self.running = True
        
        # Log startup information
        hostname = socket.gethostname()
        self.logger.info(f"Starting automated pipeline on {hostname}")
        self.logger.info(f"Parallel processing: {self.max_workers} workers")
        self.logger.info(f"PNG cleanup: every {self.cleanup_interval/3600:.1f} hours, "
                        f"files older than {self.png_cleaner.max_age_seconds/(24*3600)} days")
        
        status = self.health_checker.get_status()
        self.logger.info(f"Service status: {status}")
        
        # Initialize thread pool for parallel processing
        self.processing_pool = ThreadPoolExecutor(max_workers=self.max_workers, 
                                                 thread_name_prefix="processor")
        
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
        
        # Shutdown processing pool gracefully
        if self.processing_pool:
            self.logger.info("Shutting down processing pool...")
            self.processing_pool.shutdown(wait=True, cancel_futures=True)
        
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
    parser = argparse.ArgumentParser(
        description="Watch for new images and process each with asarina-imgproc"
    )

    # Watcher settings
    parser.add_argument('--search-root', default='/images',
                        help='Root directory to search for images')
    parser.add_argument('--camera-pattern', default='C0',
                        help='Camera subdirectory pattern')
    parser.add_argument('--poll-interval', type=float, default=1.0,
                        help='File monitoring poll interval in seconds')
    parser.add_argument('--png-cleanup-days', type=int, default=7,
                        help='Delete PNG files older than this many days (default: 7)')
    parser.add_argument('--cleanup-interval-hours', type=float, default=6,
                        help='How often to run PNG cleanup in hours (default: 6)')
    parser.add_argument('--max-workers', type=int, default=3,
                        help='Number of parallel imgproc workers (default: 3)')
    parser.add_argument('--systemd', action='store_true',
                        help='Enable systemd integration (journal logging, watchdog, notifications)')
    parser.add_argument('--health-check', action='store_true',
                        help='Print health status and exit')
    parser.add_argument('-v', '--verbose', action='store_true')

    parser.add_argument('--latest-only', action='store_true',
                        help='With all workers busy, a new frame replaces the waiting one '
                             'instead of queueing (cameras whose cadence outruns processing)')
    parser.add_argument('--min-age', type=float, default=0.0, metavar='S',
                        help='Take a file only once unmodified for S seconds (camera '
                             'writing straight into the archive; default 0)')
    parser.add_argument('--frame-timeout', type=float, metavar='S',
                        help='Kill an imgproc run after S seconds (default: no limit)')

    # Forwarded verbatim to asarina-imgproc, and only when given: anything left
    # unset is up to imgproc - its config file (--config, or /etc/asarina/config
    # and ~/.config/asarina/config), then its own defaults.
    fwd = parser.add_argument_group('forwarded to asarina-imgproc')
    fwd.add_argument('--config', metavar='FILE',
                     help='asarina config file for imgproc (replaces the default cascade)')
    fwd.add_argument('--camera', metavar='NAME',
                     help='Config section for imgproc (default: CCD_NAME of each frame)')
    fwd.add_argument('--ssh-key',
                     help='Path to SSH private key for database upload (no upload if omitted)')
    fwd.add_argument('--realtime', action='store_true',
                     help='Pass --realtime to imgproc (web preview, corrwerr, WCS-to-raw)')
    fwd.add_argument('--no-raw-update', action='store_true',
                     help='Pass --no-raw-update to imgproc (archive not writable)')
    fwd.add_argument('--phdb-root')
    fwd.add_argument('--phdb-date-fmt', metavar='FMT')
    fwd.add_argument('--phdb-by-night', action='store_true')
    fwd.add_argument('--png-root')
    fwd.add_argument('--daily-summary', metavar='DIR', dest='daily_summary_dir')
    fwd.add_argument('--stat-dir', metavar='DIR', dest='stat_dir',
                     help='Directory for per-night photometric stat ECSVs')
    fwd.add_argument('--smart-dark', metavar='CALIB.npy', dest='smart_dark_calib')
    fwd.add_argument('--pixel-scale', type=float, metavar='ARCSEC')
    fwd.add_argument('--dophot-model', metavar='FILE')
    fwd.add_argument('--dophot-catalog', metavar='NAME')
    fwd.add_argument('--dophot-maglim', type=float, metavar='N')
    fwd.add_argument('--dophot-enlarge', type=float, metavar='N')
    fwd.add_argument('--dophot-terms', metavar='TERMS')
    fwd.add_argument('--dophot-idlimit', type=int, metavar='N')
    fwd.add_argument('--dophot-max-stars', type=int, metavar='N')
    fwd.add_argument('--dophot-workers', type=int, metavar='N')
    fwd.add_argument('--makak', action='store_true', dest='makak_mode',
                     help='Enable Makak-specific features')
    fwd.add_argument('--sip', type=int, metavar='N')
    fwd.add_argument('--passes', type=int, metavar='N')
    fwd.add_argument('--sbt-window-patch', action='store_true')
    fwd.add_argument('-f', '--force', action='store_true',
                     help='Reprocess images even if ECSV already exists')

    args = parser.parse_args()

    setup_logging(use_systemd=args.systemd, verbose=args.verbose)

    if args.health_check:
        print("Service health check not implemented for standalone execution")
        sys.exit(0)

    # Build the base imgproc command that will be prepended to each image path
    imgproc_base = ['asarina-imgproc']
    for opt, dest in [('--config', 'config'), ('--camera', 'camera'),
                      ('--ssh-key', 'ssh_key'),
                      ('--phdb-root', 'phdb_root'), ('--phdb-date-fmt', 'phdb_date_fmt'),
                      ('--png-root', 'png_root'),
                      ('--daily-summary', 'daily_summary_dir'), ('--stat-dir', 'stat_dir'),
                      ('--smart-dark', 'smart_dark_calib'), ('--pixel-scale', 'pixel_scale'),
                      ('--dophot-model', 'dophot_model'), ('--dophot-catalog', 'dophot_catalog'),
                      ('--dophot-maglim', 'dophot_maglim'), ('--dophot-enlarge', 'dophot_enlarge'),
                      ('--dophot-terms', 'dophot_terms'), ('--dophot-idlimit', 'dophot_idlimit'),
                      ('--dophot-max-stars', 'dophot_max_stars'),
                      ('--dophot-workers', 'dophot_workers'),
                      ('--sip', 'sip'), ('--passes', 'passes')]:
        val = getattr(args, dest)
        if val is not None:
            imgproc_base += [opt, str(val)]
    for opt, dest in [('--realtime', 'realtime'), ('--no-raw-update', 'no_raw_update'),
                      ('--phdb-by-night', 'phdb_by_night'), ('--makak', 'makak_mode'),
                      ('--sbt-window-patch', 'sbt_window_patch'), ('-f', 'force'),
                      ('-v', 'verbose')]:
        if getattr(args, dest):
            imgproc_base.append(opt)

    watcher = AutomatedPipeline(
        use_systemd=args.systemd,
        png_cleanup_days=args.png_cleanup_days,
        cleanup_interval_hours=args.cleanup_interval_hours,
        max_workers=args.max_workers,
        imgproc_base=imgproc_base,
        latest_only=args.latest_only,
        frame_timeout=args.frame_timeout,
    )
    watcher.monitor.search_root = Path(args.search_root)
    watcher.monitor.camera_pattern = args.camera_pattern
    watcher.monitor.poll_interval = args.poll_interval
    watcher.monitor.min_age = args.min_age

    logger.info(f"Starting watcher, imgproc base: {' '.join(imgproc_base)}")
    if args.systemd:
        logger.info("Systemd integration enabled")

    watcher.run()


if __name__ == "__main__":
    main()
