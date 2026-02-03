"""Simple file-based locking for inter-process communication."""

import os
import time
import fcntl
from pathlib import Path
from contextlib import contextmanager


class FileLock:
    """File-based lock for safe inter-process file access."""
    
    def __init__(self, lock_path: str, timeout: float = 10.0):
        self.lock_path = Path(lock_path)
        self.timeout = timeout
    
    @contextmanager
    def acquire(self):
        """Acquire the lock with timeout."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = open(self.lock_path, 'w')
        
        start_time = time.time()
        acquired = False
        
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except (IOError, OSError):
                if time.time() - start_time > self.timeout:
                    lock_file.close()
                    raise TimeoutError(f"Could not acquire lock: {self.lock_path}")
                time.sleep(0.01)
        
        try:
            yield
        finally:
            if acquired:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except:
                    pass
            try:
                lock_file.close()
            except:
                pass
    
    def is_locked(self) -> bool:
        """Check if the lock is currently held."""
        if not self.lock_path.exists():
            return False
        
        try:
            with open(self.lock_path, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                return False
        except (IOError, OSError):
            return True