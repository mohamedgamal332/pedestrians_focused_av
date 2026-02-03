"""Causation logging for Chain-of-Causation reasoning traces."""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import threading


class CausationLogger:
    """Thread-safe logger for causation traces."""
    
    def __init__(self, log_dir: str, filename: str = "causation.csv"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.log_dir / filename
        self._lock = threading.Lock()
        
        # Initialize file with headers if not exists
        if not self.filepath.exists():
            self._write_header()
    
    def _write_header(self):
        """Write CSV header."""
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'datetime', 'request_id', 'causation_text'])
    
    def log(
        self,
        request_id: str,
        causation_text: str,
        timestamp: Optional[float] = None
    ):
        """
        Log a causation entry.
        
        Args:
            request_id: Request identifier
            causation_text: The Chain-of-Causation reasoning text
            timestamp: Unix timestamp (default: current time)
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()
        
        dt_str = datetime.fromtimestamp(timestamp).isoformat()
        
        # Clean causation text (remove newlines for CSV)
        clean_text = causation_text.replace('\n', ' ').replace('\r', '').strip()
        
        with self._lock:
            with open(self.filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, dt_str, request_id, clean_text])
    
    def get_recent(self, n: int = 10) -> list:
        """Get the n most recent causation entries."""
        entries = []
        
        with self._lock:
            if not self.filepath.exists():
                return entries
            
            with open(self.filepath, 'r') as f:
                reader = csv.DictReader(f)
                entries = list(reader)
        
        return entries[-n:] if len(entries) > n else entries
