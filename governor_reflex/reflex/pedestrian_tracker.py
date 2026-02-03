"""Pedestrian tracking from CARLA ground truth."""

import carla
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time


@dataclass
class PedestrianState:
    """State of a tracked pedestrian."""
    id: int
    timestamp: float
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    
    @property
    def speed(self) -> float:
        """Speed in m/s."""
        return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
    
    @property
    def heading(self) -> float:
        """Heading in degrees."""
        return np.degrees(np.arctan2(self.vy, self.vx))
    
    def get_behavior(
        self,
        standing_threshold: float = 0.5,
        walking_threshold: float = 2.0
    ) -> str:
        """
        Classify behavior based on speed.
        
        Args:
            standing_threshold: Max speed for standing (m/s)
            walking_threshold: Max speed for walking (m/s)
            
        Returns:
            'standing', 'walking', or 'running'
        """
        if self.speed < standing_threshold:
            return 'standing'
        elif self.speed < walking_threshold:
            return 'walking'
        else:
            return 'running'


class PedestrianTracker:
    """
    Track pedestrians using CARLA ground truth.
    
    Extracts pedestrian positions and velocities, classifies behavior,
    and formats data for Alpamayo model input.
    """
    
    def __init__(
        self,
        world: carla.World,
        detection_radius: float = 50.0,
        standing_threshold: float = 0.5,
        walking_threshold: float = 2.0,
        history_size: int = 10
    ):
        self.world = world
        self.detection_radius = detection_radius
        self.standing_threshold = standing_threshold
        self.walking_threshold = walking_threshold
        self.history_size = history_size
        
        # Track pedestrian history for velocity estimation
        self._history: Dict[int, List[PedestrianState]] = defaultdict(list)
        self._last_update_time = None
    
    def update(self, ego_vehicle: carla.Vehicle, timestamp: float) -> List[Dict]:
        """
        Update pedestrian tracking and get current pedestrian states.
        
        Args:
            ego_vehicle: Ego vehicle actor
            timestamp: Current simulation timestamp
            
        Returns:
            List of pedestrian info dictionaries
        """
        ego_location = ego_vehicle.get_transform().location
        
        # Get all pedestrians (walkers) in the world
        actors = self.world.get_actors()
        walkers = actors.filter('walker.pedestrian.*')
        
        pedestrians = []
        current_ids = set()
        
        for walker in walkers:
            walker_id = walker.id
            current_ids.add(walker_id)
            
            # Get position
            transform = walker.get_transform()
            location = transform.location
            
            # Check if within detection radius
            distance = location.distance(ego_location)
            if distance > self.detection_radius:
                continue
            
            # Get velocity
            velocity = walker.get_velocity()
            
            # Create state
            state = PedestrianState(
                id=walker_id,
                timestamp=timestamp,
                x=location.x,
                y=location.y,
                z=location.z,
                vx=velocity.x,
                vy=velocity.y,
                vz=velocity.z
            )
            
            # Update history
            self._history[walker_id].append(state)
            if len(self._history[walker_id]) > self.history_size:
                self._history[walker_id].pop(0)
            
            # Format for output
            ped_info = self._format_pedestrian(state, ego_location, distance)
            pedestrians.append(ped_info)
        
        # Clean up old pedestrians no longer in scene
        old_ids = set(self._history.keys()) - current_ids
        for old_id in old_ids:
            del self._history[old_id]
        
        self._last_update_time = timestamp
        
        # Sort by distance
        pedestrians.sort(key=lambda p: p['distance_to_ego'])
        
        return pedestrians
    
    def _format_pedestrian(
        self,
        state: PedestrianState,
        ego_location: carla.Location,
        distance: float
    ) -> Dict:
        """Format pedestrian state for model input."""
        # Compute relative position to ego
        rel_x = state.x - ego_location.x
        rel_y = state.y - ego_location.y
        
        # Estimate trajectory (simple linear prediction)
        trajectory = self._predict_trajectory(state, horizon=3.0, dt=0.5)
        
        return {
            'id': state.id,
            'position': {
                'x': state.x,
                'y': state.y,
                'z': state.z
            },
            'relative_position': {
                'x': rel_x,
                'y': rel_y
            },
            'velocity': {
                'x': state.vx,
                'y': state.vy,
                'z': state.vz
            },
            'speed': state.speed,
            'heading': state.heading,
            'behavior': state.get_behavior(self.standing_threshold, self.walking_threshold),
            'distance_to_ego': distance,
            'predicted_trajectory': trajectory
        }
    
    def _predict_trajectory(
        self,
        state: PedestrianState,
        horizon: float = 3.0,
        dt: float = 0.5
    ) -> List[Dict]:
        """
        Predict future trajectory using constant velocity model.
        
        Args:
            state: Current pedestrian state
            horizon: Prediction horizon in seconds
            dt: Time step for predictions
            
        Returns:
            List of predicted positions
        """
        trajectory = []
        x, y, z = state.x, state.y, state.z
        
        num_steps = int(horizon / dt)
        for i in range(1, num_steps + 1):
            t = i * dt
            pred_x = x + state.vx * t
            pred_y = y + state.vy * t
            pred_z = z  # Assume constant height
            
            trajectory.append({
                'time': t,
                'position': {'x': pred_x, 'y': pred_y, 'z': pred_z}
            })
        
        return trajectory
    
    def get_pedestrian_by_id(self, ped_id: int) -> Optional[Dict]:
        """Get information about a specific pedestrian."""
        if ped_id in self._history and self._history[ped_id]:
            state = self._history[ped_id][-1]
            return {
                'id': state.id,
                'position': {'x': state.x, 'y': state.y, 'z': state.z},
                'velocity': {'x': state.vx, 'y': state.vy, 'z': state.vz},
                'speed': state.speed,
                'behavior': state.get_behavior(self.standing_threshold, self.walking_threshold)
            }
        return None
    
    def get_pedestrians_in_path(
        self,
        ego_vehicle: carla.Vehicle,
        path_width: float = 4.0,
        path_length: float = 30.0
    ) -> List[Dict]:
        """
        Get pedestrians that are in the vehicle's path.
        
        Args:
            ego_vehicle: Ego vehicle
            path_width: Width of the path corridor (meters)
            path_length: Length of the path to check (meters)
            
        Returns:
            List of pedestrians in path
        """
        transform = ego_vehicle.get_transform()
        ego_location = transform.location
        ego_yaw = np.radians(transform.rotation.yaw)
        
        # Forward and lateral unit vectors
        forward = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])
        lateral = np.array([-np.sin(ego_yaw), np.cos(ego_yaw)])
        
        in_path = []
        
        for ped_id, history in self._history.items():
            if not history:
                continue
            
            state = history[-1]
            
            # Vector from ego to pedestrian
            to_ped = np.array([state.x - ego_location.x, state.y - ego_location.y])
            
            # Project onto forward and lateral axes
            forward_dist = np.dot(to_ped, forward)
            lateral_dist = abs(np.dot(to_ped, lateral))
            
            # Check if in path corridor
            if 0 < forward_dist < path_length and lateral_dist < path_width / 2:
                ped_info = self._format_pedestrian(
                    state, ego_location, np.linalg.norm(to_ped)
                )
                ped_info['forward_distance'] = forward_dist
                ped_info['lateral_distance'] = lateral_dist
                in_path.append(ped_info)
        
        # Sort by forward distance
        in_path.sort(key=lambda p: p['forward_distance'])
        
        return in_path
    
    def clear(self):
        """Clear tracking history."""
        self._history.clear()
        self._last_update_time = None
