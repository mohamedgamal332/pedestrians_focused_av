"""Egomotion history buffer for Alpamayo input."""

import numpy as np
from collections import deque
from typing import Dict, List, Optional
from dataclasses import dataclass
import threading
import time


@dataclass
class EgomotionSample:
    """Single egomotion sample."""
    timestamp: float
    x: float
    y: float
    z: float
    pitch: float
    yaw: float
    roll: float
    vx: float
    vy: float
    vz: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            'timestamp': self.timestamp,
            'position': {'x': self.x, 'y': self.y, 'z': self.z},
            'rotation': {'pitch': self.pitch, 'yaw': self.yaw, 'roll': self.roll},
            'velocity': {'x': self.vx, 'y': self.vy, 'z': self.vz},
            'rotation_matrix': self._get_rotation_matrix().tolist()
        }
    
    def _get_rotation_matrix(self) -> np.ndarray:
        """Compute rotation matrix from Euler angles."""
        pitch = np.radians(self.pitch)
        yaw = np.radians(self.yaw)
        roll = np.radians(self.roll)
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        return Rz @ Ry @ Rx


class EgomotionBuffer:
    """
    Buffer for egomotion history.
    
    Maintains a rolling buffer of ego vehicle poses sampled at 10Hz
    for Alpamayo model input (16 waypoints = 1.6 seconds of history).
    """
    
    def __init__(
        self,
        history_size: int = 16,
        target_frequency: float = 10.0,  # Hz
        carla_tick_rate: float = 15.0    # Hz
    ):
        self.history_size = history_size
        self.target_frequency = target_frequency
        self.target_dt = 1.0 / target_frequency
        self.carla_tick_rate = carla_tick_rate
        
        # Buffer with extra capacity for interpolation
        self._buffer = deque(maxlen=history_size * 3)
        self._lock = threading.Lock()
        
        # Timing
        self._last_sample_time = None
        self._tick_count = 0
        
        # Sample every N ticks to approximate 10Hz from 15Hz
        # 15Hz / 10Hz = 1.5, so sample every 1-2 ticks alternating
        self._sample_pattern = [1, 2, 1]  # Approximates 10Hz from 15Hz
        self._pattern_index = 0
    
    def update(self, vehicle, timestamp: float):
        """
        Update buffer with current vehicle state.
        
        Args:
            vehicle: CARLA vehicle actor
            timestamp: Current simulation timestamp
        """
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        
        sample = EgomotionSample(
            timestamp=timestamp,
            x=transform.location.x,
            y=transform.location.y,
            z=transform.location.z,
            pitch=transform.rotation.pitch,
            yaw=transform.rotation.yaw,
            roll=transform.rotation.roll,
            vx=velocity.x,
            vy=velocity.y,
            vz=velocity.z
        )
        
        self._tick_count += 1
        
        # Check if we should sample this tick (approximate 10Hz from 15Hz)
        should_sample = False
        
        if self._last_sample_time is None:
            should_sample = True
        else:
            # Sample based on pattern to approximate 10Hz
            ticks_since_sample = self._tick_count % sum(self._sample_pattern[:self._pattern_index + 1])
            if ticks_since_sample == 0:
                should_sample = True
                self._pattern_index = (self._pattern_index + 1) % len(self._sample_pattern)
        
        if should_sample:
            with self._lock:
                self._buffer.append(sample)
                self._last_sample_time = timestamp
    
    def get_history(self, num_samples: int = 16) -> List[Dict]:
        """
        Get egomotion history for model input.
        
        Args:
            num_samples: Number of samples to return
            
        Returns:
            List of egomotion dicts, oldest to newest
        """
        with self._lock:
            samples = list(self._buffer)[-num_samples:]
        
        # Pad with first sample if not enough history
        if samples and len(samples) < num_samples:
            first_sample = samples[0]
            padding = [first_sample] * (num_samples - len(samples))
            samples = padding + samples
        elif not samples:
            # No samples yet, return zeros
            return [self._zero_sample().to_dict() for _ in range(num_samples)]
        
        return [s.to_dict() for s in samples]
    
    def get_current_state(self) -> Optional[Dict]:
        """Get the most recent egomotion state."""
        with self._lock:
            if self._buffer:
                return self._buffer[-1].to_dict()
        return None
    
    def _zero_sample(self) -> EgomotionSample:
        """Create a zero-initialized sample."""
        return EgomotionSample(
            timestamp=0.0,
            x=0.0, y=0.0, z=0.0,
            pitch=0.0, yaw=0.0, roll=0.0,
            vx=0.0, vy=0.0, vz=0.0
        )
    
    def get_current_speed(self) -> float:
        """Get current speed in m/s."""
        with self._lock:
            if self._buffer:
                s = self._buffer[-1]
                return np.sqrt(s.vx**2 + s.vy**2 + s.vz**2)
        return 0.0
    
    def clear(self):
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
            self._last_sample_time = None
            self._tick_count = 0
