"""Cross-process file locking for SQLite database safety.

This module provides a ProcessLock context manager that uses platform-specific
file locking (fcntl.flock on Unix, msvcrt.locking on Windows) to ensure that
multi-statement transactions are protected across multiple Python processes.

Usage:
    with ProcessLock(db_path + ".lock"):
        # Multi-statement operations are protected here
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT ...")
        conn.execute("UPDATE ...")
        conn.commit()

ProcessLock is NOT reentrant within the same thread/process. If you need
reentrant semantics, wrap ProcessLock usage with threading.RLock or similar.
"""

import contextlib
import logging
import os
import platform
import threading
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class ProcessLock:
    """Exclusive cross-process file lock using native OS primitives.

    This lock is safe for multi-process access but NOT reentrant within
    a single process. Attempting to acquire the same lock twice in the same
    thread will block indefinitely.

    On Unix systems, uses fcntl.flock with LOCK_EX (exclusive lock).
    On Windows, uses msvcrt.locking with exclusive lock.

    Attributes:
        lock_file_path: Path to the lock file (typically <db_path>.lock).
    """

    def __init__(self, lock_file_path: str | Path) -> None:
        """Initialize the ProcessLock.

        Args:
            lock_file_path: Path where the lock file will be created/held.
                           Typically <database_path>.lock
        """
        self.lock_file_path = Path(lock_file_path)
        self._fd: int | None = None
        self._lock = threading.Lock()  # Protects fd state during acquire/release

    def __enter__(self) -> "ProcessLock":
        """Acquire the exclusive lock, blocking until available."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release the lock and close the file descriptor."""
        self.release()
        return None

    def acquire(self) -> None:
        """Acquire an exclusive lock on the lock file.

        This method blocks until the lock is available. It creates the lock file
        if it doesn't exist. The lock is held by keeping a file descriptor open
        with an exclusive lock applied.

        Raises:
            IOError: If the lock cannot be created or acquired.
            OSError: On platform-specific locking errors.
        """
        with self._lock:
            if self._fd is not None:
                raise RuntimeError(f"Lock already held on {self.lock_file_path}. ProcessLock is not reentrant.")

            try:
                # Create lock file if needed, ensuring parent directory exists
                self.lock_file_path.parent.mkdir(parents=True, exist_ok=True)

                # Open/create lock file
                # O_CREAT | O_WRONLY: create if doesn't exist, open for writing
                # 0o666: permissions (subject to umask)
                flags = os.O_CREAT | os.O_WRONLY
                self._fd = os.open(str(self.lock_file_path), flags, 0o666)

                if platform.system() == "Windows":
                    import msvcrt

                    try:
                        # msvcrt.locking: (fd, mode, nbytes)
                        # LK_LOCK or LK_NBLCK for exclusive non-blocking,
                        # but we want blocking so we loop with LK_NBLCK and sleep
                        # Actually, msvcrt.locking blocks by default with LK_LOCK
                        msvcrt.locking(self._fd, msvcrt.LK_LOCK, 1)
                    except OSError as e:
                        os.close(self._fd)
                        self._fd = None
                        raise OSError(f"Failed to acquire lock on {self.lock_file_path}: {e}") from e
                else:
                    # Unix: use fcntl.flock
                    import fcntl

                    try:
                        # LOCK_EX: exclusive lock, blocking
                        fcntl.flock(self._fd, fcntl.LOCK_EX)
                    except OSError as e:
                        os.close(self._fd)
                        self._fd = None
                        raise OSError(f"Failed to acquire lock on {self.lock_file_path}: {e}") from e

                LOGGER.debug(f"Acquired cross-process lock on {self.lock_file_path}")

            except Exception as e:
                if self._fd is not None:
                    with contextlib.suppress(Exception):
                        os.close(self._fd)
                    self._fd = None
                LOGGER.error(f"Error acquiring lock: {e}")
                raise

    def release(self) -> None:
        """Release the lock and close the file descriptor.

        This method is idempotent: calling release() multiple times is safe.
        It closes the file descriptor and releases the lock.
        """
        with self._lock:
            if self._fd is None:
                # Already released or never acquired
                return

            try:
                if platform.system() == "Windows":
                    import msvcrt

                    try:
                        # Unlock the file before closing
                        msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
                    except OSError as e:
                        LOGGER.warning(f"Error unlocking on Windows: {e}")
                else:
                    # Unix: fcntl.flock automatically releases when fd is closed
                    # but we can explicitly unlock if desired
                    import fcntl

                    try:
                        fcntl.flock(self._fd, fcntl.LOCK_UN)
                    except OSError as e:
                        LOGGER.warning(f"Error unlocking on Unix: {e}")

                os.close(self._fd)
                LOGGER.debug(f"Released cross-process lock on {self.lock_file_path}")
            except OSError as e:
                LOGGER.warning(f"Error releasing lock on {self.lock_file_path}: {e}")
            finally:
                self._fd = None
