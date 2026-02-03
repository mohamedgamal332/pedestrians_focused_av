"""Coordinate transformation utilities."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Pose:
    """3D pose with position and rotation."""
    x: float
    y: float
    z: float
    pitch: float  # degrees
    yaw: float    # degrees
    roll: float   # degrees
    
    def to_dict(self) -> Dict:
        return {
            'x': self.x, 'y': self.y, 'z': self.z,
            'pitch': self.pitch, 'yaw': self.yaw, 'roll': self.roll
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Pose':
        return cls(
            x=d['x'], y=d['y'], z=d.get('z', 0.0),
            pitch=d.get('pitch', 0.0), yaw=d['yaw'], roll=d.get('roll', 0.0)
        )
    
    def to_rotation_matrix(self) -> np.ndarray:
        """Convert Euler angles to 3x3 rotation matrix."""
        pitch = np.radians(self.pitch)
        yaw = np.radians(self.yaw)
        roll = np.radians(self.roll)
        
        # Rotation matrices
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


class CoordinateTransformer:
    """Transform between ego-centric and world coordinate systems."""
    
    def __init__(self):
        pass
    
    def ego_to_world(self, ego_waypoints: List[Dict], ego_pose: Pose) -> List[Pose]:
        """
        Transform waypoints from ego-centric to world coordinates.
        
        Args:
            ego_waypoints: List of {'x': float, 'y': float, 'yaw': float} in ego frame
            ego_pose: Current pose of ego vehicle in world frame
            
        Returns:
            List of Pose objects in world frame
        """
        world_poses = []
        
        # Get ego transformation
        ego_yaw_rad = np.radians(ego_pose.yaw)
        cos_yaw = np.cos(ego_yaw_rad)
        sin_yaw = np.sin(ego_yaw_rad)
        
        for wp in ego_waypoints:
            # Rotate from ego to world
            ego_x = wp['x']
            ego_y = wp['y']
            
            world_x = ego_pose.x + ego_x * cos_yaw - ego_y * sin_yaw
            world_y = ego_pose.y + ego_x * sin_yaw + ego_y * cos_yaw
            world_yaw = ego_pose.yaw + wp.get('yaw', 0.0)
            
            # Normalize yaw to [-180, 180]
            while world_yaw > 180:
                world_yaw -= 360
            while world_yaw < -180:
                world_yaw += 360
            
            world_poses.append(Pose(
                x=world_x,
                y=world_y,
                z=ego_pose.z,
                pitch=0.0,
                yaw=world_yaw,
                roll=0.0
            ))
        
        return world_poses
    
    def world_to_ego(self, world_pose: Pose, ego_pose: Pose) -> Dict:
        """
        Transform a world pose to ego-centric coordinates.
        
        Args:
            world_pose: Pose in world frame
            ego_pose: Current ego vehicle pose in world frame
            
        Returns:
            Dict with x, y, yaw in ego frame
        """
        # Translate to ego origin
        dx = world_pose.x - ego_pose.x
        dy = world_pose.y - ego_pose.y
        
        # Rotate to ego frame
        ego_yaw_rad = np.radians(ego_pose.yaw)
        cos_yaw = np.cos(-ego_yaw_rad)
        sin_yaw = np.sin(-ego_yaw_rad)
        
        ego_x = dx * cos_yaw - dy * sin_yaw
        ego_y = dx * sin_yaw + dy * cos_yaw
        ego_yaw = world_pose.yaw - ego_pose.yaw
        
        return {'x': ego_x, 'y': ego_y, 'yaw': ego_yaw}
    
    def decode_unicycle_trajectory(
        self,
        accelerations: np.ndarray,
        curvatures: np.ndarray,
        initial_speed: float,
        dt: float = 0.1
    ) -> List[Dict]:
        """
        Decode unicycle model actions to ego-centric trajectory.
        
        Args:
            accelerations: Array of accelerations (m/s^2), shape (N,)
            curvatures: Array of curvatures (1/m), shape (N,)
            initial_speed: Initial speed (m/s)
            dt: Time step (s)
            
        Returns:
            List of waypoints in ego frame: [{'x': float, 'y': float, 'yaw': float}, ...]
        """
        n_waypoints = len(accelerations)
        
        # Initialize state
        x, y, theta = 0.0, 0.0, 0.0  # Start at ego origin, facing forward
        v = initial_speed
        
        waypoints = [{'x': x, 'y': y, 'yaw': np.degrees(theta)}]
        
        for i in range(n_waypoints):
            # Update velocity
            v = max(0.0, v + accelerations[i] * dt)  # Clamp to non-negative
            
            # Update heading
            theta = theta + v * curvatures[i] * dt
            
            # Update position
            x = x + v * np.cos(theta) * dt
            y = y + v * np.sin(theta) * dt
            
            waypoints.append({
                'x': x,
                'y': y,
                'yaw': np.degrees(theta)
            })
        
        return waypoints
