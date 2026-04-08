#!/usr/bin/env python3
# /home/fnovotny/bin/transient_daemon.py
"""
Asynchronous transient detection daemon.
Receives requests via Unix socket, copies files, responds immediately,
then processes transients in background with process limiting.
"""

import socket
import os
import json
import shutil
import subprocess
import logging
import threading
import time
import signal
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

# Configuration
SOCKET_PATH = "/home/fnovotny/transient_daemon.sock"
WORK_DIR = Path("/home/fnovotny/transient_work")
LOG_DIR = Path("/home/fnovotny/logs")
MAX_PARALLEL_PROCESSES = 3  # Limit concurrent transient detections
PIPELINE_SCRIPT = "/home/fnovotny/bin/pipeline_magic.py"

# Setup logging
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'transient_daemon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TransientDaemon:
    def __init__(self):
        self.work_dir = WORK_DIR
        self.work_dir.mkdir(exist_ok=True)

        # Thread pool for background processing
        self.executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL_PROCESSES)
        self.active_jobs = 0
        self.jobs_lock = threading.Lock()

        # Job queue and statistics
        self.job_queue = Queue()
        self.processed_count = 0
        self.failed_count = 0

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        self.running = True

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        self.executor.shutdown(wait=True)
        try:
            os.unlink(SOCKET_PATH)
        except FileNotFoundError:
            pass
        sys.exit(0)

    def _generate_work_id(self) -> str:
        """Generate unique work ID for this job."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"job_{timestamp}_{os.getpid()}"

    def copy_files_for_processing(self, ecsv_path: str, fits_path: str) -> tuple:
        """Copy files to work directory and return new paths."""
        work_id = self._generate_work_id()
        job_dir = self.work_dir / work_id
        job_dir.mkdir(exist_ok=True)

        ecsv_copy = job_dir / Path(ecsv_path).name
        fits_copy = job_dir / Path(fits_path).name

        try:
            shutil.copy2(ecsv_path, ecsv_copy)
            shutil.copy2(fits_path, fits_copy)
            logger.info(f"Copied files for job {work_id}")
            return str(ecsv_copy), str(fits_copy), job_dir
        except Exception as e:
            logger.error(f"Failed to copy files for job {work_id}: {e}")
            # Clean up on failure
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

    def process_transient_background(self, ecsv_path: str, fits_path: str, job_dir: Path):
        """Process transient detection in background thread."""
        job_id = job_dir.name

        with self.jobs_lock:
            self.active_jobs += 1

        try:
            logger.info(f"Starting transient processing for job {job_id}")
            start_time = time.time()

            # Run pipeline_magic.py
            result = subprocess.run([
                PIPELINE_SCRIPT, ecsv_path, fits_path
            ],
            cwd=str(job_dir),
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                logger.info(f"Job {job_id} completed successfully in {elapsed:.1f}s")
                self.processed_count += 1
            else:
                logger.error(f"Job {job_id} failed with exit code {result.returncode}")
                logger.error(f"Stderr: {result.stderr}")
                self.failed_count += 1

            # Log stdout if present
            if result.stdout:
                logger.debug(f"Job {job_id} stdout: {result.stdout}")

        except subprocess.TimeoutExpired:
            logger.error(f"Job {job_id} timed out after 10 minutes")
            self.failed_count += 1
        except Exception as e:
            logger.error(f"Job {job_id} failed with exception: {e}")
            self.failed_count += 1
        finally:
            # Clean up work directory
            try:
                shutil.rmtree(job_dir, ignore_errors=True)
                logger.debug(f"Cleaned up job directory {job_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up job directory {job_dir}: {e}")

            with self.jobs_lock:
                self.active_jobs -= 1
                logger.info(f"Job {job_id} finished. Active jobs: {self.active_jobs}")

    def handle_request(self, conn):
        """Handle incoming socket request."""
        try:
            # Receive request
            data = conn.recv(4096).decode()
            if not data:
                return

            request = json.loads(data)
            ecsv_path = request['ecsv_path']
            fits_path = request['fits_path']

            logger.info(f"Received request: {Path(ecsv_path).name}")

            # Check if we're at capacity
            with self.jobs_lock:
                if self.active_jobs >= MAX_PARALLEL_PROCESSES:
                    response = {
                        'success': False,
                        'error': f'At capacity ({self.active_jobs}/{MAX_PARALLEL_PROCESSES} jobs active)'
                    }
                    conn.send(json.dumps(response).encode())
                    return

            # Copy files immediately
            try:
                ecsv_copy, fits_copy, job_dir = self.copy_files_for_processing(ecsv_path, fits_path)
                logger.info(f"Files copied successfully for job {job_dir.name}")
            except Exception as e:
                logger.error(f"File copy failed: {e}")
                response = {'success': False, 'error': f'File copy failed: {str(e)}'}
                conn.send(json.dumps(response).encode())
                return

            # Respond immediately that files are copied
            response = {
                'success': True,
                'message': 'Files copied, processing started in background',
                'job_id': job_dir.name
            }
            conn.send(json.dumps(response).encode())

            # Submit background job
            self.executor.submit(
                self.process_transient_background,
                ecsv_copy, fits_copy, job_dir
            )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON request: {e}")
            error_response = {'success': False, 'error': 'Invalid JSON'}
            try:
                conn.send(json.dumps(error_response).encode())
            except:
                pass  # Connection may be closed
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            error_response = {'success': False, 'error': str(e)}
            try:
                conn.send(json.dumps(error_response).encode())
            except:
                pass  # Connection may be closed

    def print_status(self):
        """Print periodic status information."""
        while self.running:
            time.sleep(60)  # Status every minute
            with self.jobs_lock:
                logger.info(f"Status: {self.active_jobs} active jobs, "
                           f"{self.processed_count} completed, {self.failed_count} failed")

    def run(self):
        """Main daemon loop."""
        # Remove old socket
        try:
            os.unlink(SOCKET_PATH)
        except FileNotFoundError:
            pass

        # Create socket
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(SOCKET_PATH)
        os.chmod(SOCKET_PATH, 0o666)  # Allow other users to connect
        sock.listen(5)

        logger.info(f"Transient daemon started")
        logger.info(f"Listening on {SOCKET_PATH}")
        logger.info(f"Max parallel processes: {MAX_PARALLEL_PROCESSES}")
        logger.info(f"Work directory: {self.work_dir}")

        # Start status thread
        status_thread = threading.Thread(target=self.print_status, daemon=True)
        status_thread.start()

        try:
            while self.running:
                try:
                    sock.settimeout(1.0)  # Allow checking self.running
                    conn, addr = sock.accept()

                    # Handle request in main thread to avoid socket issues
                    self.handle_request(conn)
                    # Handle request in separate thread to avoid blocking
                    #request_thread = threading.Thread(
                    #    target=self.handle_request,
                    #    args=(conn,),
                    #    daemon=True
                    #)
                    #request_thread.start()

                    # Close connection after starting handler
                    conn.close()

                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        logger.error(f"Socket error: {e}")
                        time.sleep(1)
        finally:
            sock.close()
            logger.info("Daemon stopped")


def main():
    daemon = TransientDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
