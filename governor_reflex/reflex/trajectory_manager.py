"""Trajectory management for route injection."""

import xml.etree.ElementTree as ET
import json
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Waypoint:
    """Single trajectory waypoint."""
    x: float
    y: float
    z: float
    pitch: float
    yaw: float
    roll: float
    
    def to_tuple(self) -> Tuple[float, float, float, float, float, float]:
        return (self.x, self.y, self.z, self.pitch, self.yaw, self.roll)


class TrajectoryManager:
    """
    Manage trajectory from Governor and track consumption by Reflex.
    """
    
    def __init__(
        self,
        runtime_dir: str,
        min_buffer_seconds: float = 2.0,
        waypoint_dt: float = 0.1  # 10Hz waypoints
    ):
        self.runtime_dir = Path(runtime_dir)
        self.output_dir = self.runtime_dir / 'output'
        self.min_buffer_seconds = min_buffer_seconds
        self.waypoint_dt = waypoint_dt
        
        self.trajectory_file = self.output_dir / 'trajectory.xml'
        self.status_file = self.output_dir / 'status.json'
        
        # Current trajectory state
        self._waypoints: List[Waypoint] = []
        self._current_index: int = 0
        self._trajectory_id: Optional[str] = None
        self._last_load_time: float = 0
        self._lock = threading.Lock()
    
    def check_governor_status(self) -> Dict:
        """Check the Governor's status."""
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        
        return {
            'governor_status': 'unknown',
            'trajectory_valid': False
        }
    
    def is_governor_ready(self) -> bool:
        """Check if Governor is ready."""
        status = self.check_governor_status()
        return status.get('governor_status') == 'ready'
    
    def wait_for_governor(self, timeout: float = 60.0) -> bool:
        """
        Wait for Governor to be ready.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if Governor is ready, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_governor_ready():
                return True
            time.sleep(0.5)
        
        return False
    
    def load_trajectory(self) -> bool:
        """
        Load trajectory from file.
        
        Returns:
            True if new trajectory loaded, False otherwise
        """
        if not self.trajectory_file.exists():
            return False
        
        # Check if file has been modified
        mtime = self.trajectory_file.stat().st_mtime
        if mtime <= self._last_load_time:
            return False  # No new trajectory
        
        try:
            tree = ET.parse(self.trajectory_file)
            root = tree.getroot()
            
            waypoints = []
            for wp_elem in root.findall('waypoint'):
                wp = Waypoint(
                    x=float(wp_elem.get('x', 0)),
                    y=float(wp_elem.get('y', 0)),
                    z=float(wp_elem.get('z', 0)),
                    pitch=float(wp_elem.get('pitch', 0)),
                    yaw=float(wp_elem.get('yaw', 0)),
                    roll=float(wp_elem.get('roll', 0))
                )
                waypoints.append(wp)
            
            if not waypoints:
                return False
            
            with self._lock:
                self._waypoints = waypoints
                self._current_index = 0
                self._trajectory_id = root.get('id', str(time.time()))
                self._last_load_time = mtime
            
            print(f"[TrajectoryManager] Loaded trajectory with {len(waypoints)} waypoints")
            return True
            
        except Exception as e:
            print(f"[TrajectoryManager] Error loading trajectory: {e}")
            return False
    
    def get_remaining_waypoints(self) -> List[Waypoint]:
        """Get remaining waypoints in current trajectory."""
        with self._lock:
            return self._waypoints[self._current_index:]
    
    def get_remaining_seconds(self) -> float:
        """Get remaining trajectory duration in seconds."""
        with self._lock:
            remaining = len(self._waypoints) - self._current_index
            return remaining * self.waypoint_dt
    
    def needs_replan(self) -> bool:
        """Check if we need to request a new trajectory."""
        return self.get_remaining_seconds() < self.min_buffer_seconds
    
    def consume_waypoint(self) -> Optional[Waypoint]:
        """
        Consume and return the next waypoint.
        
        Returns:
            Next waypoint or None if exhausted
        """
        with self._lock:
            if self._current_index < len(self._waypoints):
                wp = self._waypoints[self._current_index]
                self._current_index += 1
                return wp
        return None
    
    def peek_waypoints(self, n: int = 10) -> List[Waypoint]:
        """Peek at upcoming waypoints without consuming them."""
        with self._lock:
            end_idx = min(self._current_index + n, len(self._waypoints))
            return self._waypoints[self._current_index:end_idx]
    
    def get_waypoint_at_distance(self, distance: float) -> Optional[Waypoint]:
        """
        Get waypoint at approximate distance ahead.
        
        Args:
            distance: Distance in meters
            
        Returns:
            Waypoint at that distance or None
        """
        # Estimate waypoint index based on average speed assumption (10 m/s)
        avg_speed = 10.0  # m/s
        time_ahead = distance / avg_speed
        waypoint_ahead = int(time_ahead / self.waypoint_dt)
        
        with self._lock:
            idx = self._current_index + waypoint_ahead
            if idx < len(self._waypoints):
                return self._waypoints[idx]
        return None
    
    def get_trajectory_for_carl(self) -> List[Tuple]:
        """
        Get remaining trajectory in format suitable for CaRL.
        
        Returns:
            List of (transform_dict, road_option) tuples
        """
        waypoints = self.get_remaining_waypoints()
        
        # Convert to CaRL format
        trajectory = []
        for wp in waypoints:
            transform = {
                'location': {'x': wp.x, 'y': wp.y, 'z': wp.z},
                'rotation': {'pitch': wp.pitch, 'yaw': wp.yaw, 'roll': wp.roll}
            }
            # Default to LANEFOLLOW (value 4)
            road_option = 4
            trajectory.append((transform, road_option))
        
        return trajectory
    
    def reset(self):
        """Reset trajectory state."""
        with self._lock:
            self._waypoints.clear()
            self._current_index = 0
            self._trajectory_id = None

