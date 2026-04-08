#!/usr/bin/env python3
"""
Utility classes for the astronomical image processing pipeline.
Shared between different pipeline components.
"""

import json
import socket
import time
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class HealthChecker:
    """Health monitoring for pipeline services."""
    
    def __init__(self):
        self.last_activity = time.time()
        self.processed_count = 0
        self.error_count = 0
        self.start_time = time.time()
    
    def record_activity(self):
        """Record successful activity."""
        self.last_activity = time.time()
        self.processed_count += 1
    
    def record_error(self):
        """Record error."""
        self.error_count += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get current health status."""
        uptime = time.time() - self.start_time
        time_since_activity = time.time() - self.last_activity
        
        return {
            'uptime_seconds': uptime,
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'last_activity_seconds_ago': time_since_activity,
            'status': 'healthy' if time_since_activity < 3600 else 'idle'
        }
    
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        # Consider healthy if we've processed something recently or just started
        time_since_activity = time.time() - self.last_activity
        uptime = time.time() - self.start_time
        
        # Healthy if recent activity OR just started (less than 10 minutes)
        return time_since_activity < 3600 or uptime < 600


class PngCleaner:
    """Manages automatic cleanup of old PNG files."""
    
    def __init__(self, png_root: str = "/home/mates/png", max_age_days: int = 7):
        self.png_root = Path(png_root).expanduser()
        self.max_age_seconds = max_age_days * 24 * 3600
        self.logger = logging.getLogger(__name__)
    
    def cleanup_old_pngs(self) -> Dict[str, Any]:
        """Remove PNG files older than max_age_days.
        
        Returns:
            Dictionary with cleanup statistics
        """
        if not self.png_root.exists():
            return {'deleted_files': 0, 'freed_mb': 0, 'error': 'PNG root directory does not exist'}
        
        deleted_count = 0
        freed_bytes = 0
        current_time = time.time()
        
        try:
            # Find all PNG files recursively
            for png_file in self.png_root.rglob('*.png'):
                try:
                    file_age = current_time - png_file.stat().st_mtime
                    
                    if file_age > self.max_age_seconds:
                        file_size = png_file.stat().st_size
                        png_file.unlink()
                        deleted_count += 1
                        freed_bytes += file_size
                        self.logger.debug(f"Deleted old PNG: {png_file}")
                        
                except Exception as e:
                    self.logger.warning(f"Error deleting {png_file}: {e}")
                    continue
            
            # Clean up empty directories
            self._cleanup_empty_dirs()
            
            freed_mb = freed_bytes / (1024 * 1024)
            
            if deleted_count > 0:
                self.logger.info(f"PNG cleanup: deleted {deleted_count} files, freed {freed_mb:.1f} MB")
            else:
                self.logger.debug("PNG cleanup: no old files to delete")
            
            return {
                'deleted_files': deleted_count,
                'freed_mb': round(freed_mb, 1),
                'max_age_days': self.max_age_seconds / (24 * 3600)
            }
            
        except Exception as e:
            self.logger.error(f"Error during PNG cleanup: {e}")
            return {'deleted_files': 0, 'freed_mb': 0, 'error': str(e)}
    
    def _cleanup_empty_dirs(self):
        """Remove empty directories in PNG tree."""
        try:
            # Walk bottom-up to remove empty directories
            for dirpath in sorted(self.png_root.rglob('*'), key=lambda p: len(p.parts), reverse=True):
                if dirpath.is_dir() and dirpath != self.png_root:
                    try:
                        # Only remove if empty
                        dirpath.rmdir()
                        self.logger.debug(f"Removed empty directory: {dirpath}")
                    except OSError:
                        # Directory not empty, which is fine
                        pass
        except Exception as e:
            self.logger.warning(f"Error cleaning empty directories: {e}")


class TransientSearcher:
    """Notify the transient daemon of a newly processed image via Unix socket."""

    def __init__(self, socket_path: str = "/home/fnovotny/transient_daemon.sock"):
        self.socket_path = socket_path

    def search_transients(self, ecsv_path: str, fits_path: str) -> bool:
        """Send ecsv_path + fits_path to the transient daemon.

        Returns True if the daemon acknowledged successfully.
        Failure is logged but never raises — transient search is best-effort.
        """
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(self.socket_path)
            sock.send(json.dumps({
                'ecsv_path': ecsv_path,
                'fits_path': fits_path,
            }).encode())
            response = json.loads(sock.recv(4096).decode())
            sock.close()
            if response.get('success'):
                logger.info(f"Transient search queued for {Path(ecsv_path).name}")
                return True
            else:
                logger.error(f"Transient daemon error: {response.get('error', 'unknown')}")
                return False
        except Exception as e:
            logger.error(f"Transient daemon unreachable: {e}")
            return False


# Configure logging for systemd support
def setup_logging(use_systemd=False, verbose=False):
    """Setup logging with optional systemd journal support."""
    level = logging.DEBUG if verbose else logging.INFO
    
    try:
        if use_systemd:
            from systemd import journal
            # Use systemd journal for logging
            handler = journal.JournalHandler()
            handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        else:
            raise ImportError  # Fall back to standard logging
    except ImportError:
        # Use standard logging
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
    
    logger = logging.getLogger()
    logger.setLevel(level)
    # Clear any existing handlers
    logger.handlers.clear()
    logger.addHandler(handler)
    
    return logger
