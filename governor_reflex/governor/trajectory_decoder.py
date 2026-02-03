"""Decode Alpamayo trajectory outputs."""

import numpy as np
from typing import List, Dict, Tuple, Optional
import re


class TrajectoryDecoder:
    """Decode trajectory from Alpamayo model output."""
    
    def __init__(
        self,
        n_waypoints: int = 64,
        dt: float = 0.1,
    ):
        self.n_waypoints = n_waypoints
        self.dt = dt
    
    def decode_from_model_output(
        self,
        model_output: Dict,
        initial_speed: float = 0.0
    ) -> Tuple[List[Dict], str]:
        """
        Decode trajectory from Alpamayo model output.
        
        Args:
            model_output: Dictionary from AlpamayoWrapper.generate()
            initial_speed: Current vehicle speed in m/s
            
        Returns:
            Tuple of (waypoints list, causation text)
        """
        waypoints = []
        causation = ""
        
        try:
            trajectory = model_output.get('trajectory', {})
            
            # Check for direct waypoints (new format from Alpamayo)
            if 'waypoints' in trajectory:
                waypoints = trajectory['waypoints']
            elif 'xyz' in trajectory:
                # Convert xyz array to waypoints
                xyz = trajectory['xyz']
                rot = trajectory.get('rot')
                
                for i in range(len(xyz)):
                    x, y, z = xyz[i]
                    
                    if rot is not None:
                        # Extract yaw from rotation matrix
                        yaw = np.degrees(np.arctan2(rot[i][1, 0], rot[i][0, 0]))
                    else:
                        # Estimate yaw from trajectory direction
                        if i < len(xyz) - 1:
                            dx = xyz[i+1][0] - x
                            dy = xyz[i+1][1] - y
                            yaw = np.degrees(np.arctan2(dy, dx))
                        elif i > 0:
                            dx = x - xyz[i-1][0]
                            dy = y - xyz[i-1][1]
                            yaw = np.degrees(np.arctan2(dy, dx))
                        else:
                            yaw = 0.0
                    
                    waypoints.append({
                        'x': float(x),
                        'y': float(y),
                        'z': float(z),
                        'yaw': float(yaw)
                    })
            
            # Legacy format: acceleration/curvature
            elif 'accelerations' in trajectory:
                accelerations = np.array(trajectory.get('accelerations', [0.0] * self.n_waypoints))
                curvatures = np.array(trajectory.get('curvatures', [0.0] * self.n_waypoints))
                waypoints = self.unicycle_to_waypoints(accelerations, curvatures, initial_speed)
            
            # Fallback: generate straight trajectory
            else:
                waypoints = self._generate_straight_trajectory(initial_speed)
            
            # Get causation text
            causation = model_output.get('reasoning', '')
            if not causation:
                causation = "Trajectory generated without explicit reasoning."
            
        except Exception as e:
            print(f"Error decoding trajectory: {e}")
            waypoints = self._generate_straight_trajectory(initial_speed)
            causation = f"Fallback trajectory due to decoding error: {e}"
        
        return waypoints, causation
    
    def unicycle_to_waypoints(
        self,
        accelerations: np.ndarray,
        curvatures: np.ndarray,
        initial_speed: float
    ) -> List[Dict]:
        """
        Convert unicycle model actions to waypoints.
        
        Args:
            accelerations: Array of accelerations (m/s^2)
            curvatures: Array of curvatures (1/m)
            initial_speed: Initial speed (m/s)
            
        Returns:
            List of waypoints: [{'x': float, 'y': float, 'yaw': float}, ...]
        """
        # Initialize state at ego origin
        x, y, theta = 0.0, 0.0, 0.0
        v = initial_speed
        
        waypoints = [{'x': x, 'y': y, 'z': 0.0, 'yaw': 0.0, 'speed': v}]
        
        for i in range(len(accelerations)):
            # Update velocity
            v = max(0.0, v + accelerations[i] * self.dt)
            
            # Update heading
            theta = theta + v * curvatures[i] * self.dt
            
            # Update position
            x = x + v * np.cos(theta) * self.dt
            y = y + v * np.sin(theta) * self.dt
            
            waypoints.append({
                'x': x,
                'y': y,
                'z': 0.0,
                'yaw': np.degrees(theta),
                'speed': v
            })
        
        return waypoints
    
    def _generate_straight_trajectory(self, initial_speed: float) -> List[Dict]:
        """Generate a straight trajectory as fallback."""
        waypoints = []
        x = 0.0
        v = initial_speed
        
        for i in range(self.n_waypoints):
            waypoints.append({
                'x': x,
                'y': 0.0,
                'z': 0.0,
                'yaw': 0.0
            })
            x += v * self.dt
        
        return waypoints
    
    def extract_causation_from_text(self, text: str) -> str:
        """
        Extract Chain-of-Causation reasoning from model output text.
        
        Args:
            text: Raw model output text
            
        Returns:
            Cleaned causation text
        """
        if not text:
            return ""
        
        # Clean up the text
        clean = text.strip()
        
        # Remove excessive whitespace
        clean = ' '.join(clean.split())
        
        # Truncate if too long
        if len(clean) > 1000:
            clean = clean[:1000] + "..."
        
        return clean
