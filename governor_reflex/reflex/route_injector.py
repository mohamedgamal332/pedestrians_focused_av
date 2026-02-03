"""Inject Alpamayo trajectories into CaRL/PCLA."""

import carla
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum


class RoadOption(Enum):
    """Road options for route planning."""
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class RouteInjector:
    """
    Inject trajectories from Governor into CaRL agent.
    
    This class bridges the Alpamayo trajectory output with
    the CaRL agent's route following system.
    """
    
    def __init__(self, world: carla.World, pcla_instance=None):
        self.world = world
        self.carla_map = world.get_map()
        self.pcla = pcla_instance
        self._last_injected_id = None
    
    def set_pcla_instance(self, pcla_instance):
        """Set the PCLA instance for route injection."""
        self.pcla = pcla_instance
    
    def waypoints_to_carla_route(
        self,
        waypoints: List[Dict],
        default_road_option: RoadOption = RoadOption.LANEFOLLOW
    ) -> Tuple[List, List]:
        """
        Convert waypoints to CARLA route format.
        
        Args:
            waypoints: List of waypoint dicts with position and rotation
            default_road_option: Default road option for waypoints
            
        Returns:
            Tuple of (gps_route, world_coord_route)
        """
        gps_route = []
        world_coord_route = []
        
        for wp_dict in waypoints:
            # Handle different input formats
            if isinstance(wp_dict, dict):
                if 'location' in wp_dict:
                    loc = wp_dict['location']
                    rot = wp_dict.get('rotation', {})
                else:
                    loc = wp_dict
                    rot = wp_dict
                
                x = loc.get('x', 0)
                y = loc.get('y', 0)
                z = loc.get('z', 0)
                pitch = rot.get('pitch', 0)
                yaw = rot.get('yaw', 0)
                roll = rot.get('roll', 0)
            else:
                # Assume it's a Waypoint object
                x, y, z = wp_dict.x, wp_dict.y, wp_dict.z
                pitch, yaw, roll = wp_dict.pitch, wp_dict.yaw, wp_dict.roll
            
            # Create CARLA transform
            location = carla.Location(x=x, y=y, z=z)
            rotation = carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
            transform = carla.Transform(location, rotation)
            
            # Get nearest CARLA waypoint for GPS data
            carla_wp = self.carla_map.get_waypoint(
                location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            
            # Create GPS entry (simplified - using world coords)
            gps_entry = {
                'lat': y,  # Simplified mapping
                'lon': x,
                'z': z
            }
            
            # Determine road option based on consecutive waypoints
            road_option = default_road_option
            
            gps_route.append((gps_entry, road_option.value))
            world_coord_route.append((transform, road_option.value))
        
        return gps_route, world_coord_route
    
    def inject_route(
        self,
        waypoints: List[Dict],
        route_id: str = None
    ) -> bool:
        """
        Inject a new route into the PCLA/CaRL agent.
        
        Args:
            waypoints: List of waypoints to inject
            route_id: Optional route identifier
            
        Returns:
            True if injection successful
        """
        if self.pcla is None:
            print("[RouteInjector] ERROR: No PCLA instance set")
            return False
        
        if not waypoints:
            print("[RouteInjector] WARNING: Empty waypoints list")
            return False
        
        try:
            # Convert to CARLA route format
            gps_route, world_coord_route = self.waypoints_to_carla_route(waypoints)
            
            # Get the agent instance from PCLA
            agent = self.pcla.agent_instance
            
            if agent is None:
                print("[RouteInjector] ERROR: No agent instance")
                return False
            
            # Inject the route
            agent.set_global_plan(gps_route, world_coord_route)
            
            self._last_injected_id = route_id
            print(f"[RouteInjector] Injected route with {len(waypoints)} waypoints")
            
            return True
            
        except Exception as e:
            print(f"[RouteInjector] ERROR injecting route: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def inject_from_trajectory_manager(self, trajectory_manager) -> bool:
        """
        Inject route from TrajectoryManager.
        
        Args:
            trajectory_manager: TrajectoryManager instance
            
        Returns:
            True if injection successful
        """
        carl_trajectory = trajectory_manager.get_trajectory_for_carl()
        
        if not carl_trajectory:
            return False
        
        # Convert to waypoints format
        waypoints = []
        for transform_dict, road_option in carl_trajectory:
            waypoints.append(transform_dict)
        
        return self.inject_route(
            waypoints,
            route_id=trajectory_manager._trajectory_id
        )
    
    def get_remaining_route_length(self) -> float:
        """
        Get the remaining length of the current route.
        
        Returns:
            Remaining distance in meters
        """
        if self.pcla is None or self.pcla.agent_instance is None:
            return 0.0
        
        try:
            agent = self.pcla.agent_instance
            
            if hasattr(agent, 'dense_global_plan_world_coord'):
                route = agent.dense_global_plan_world_coord
                if route:
                    # Calculate total remaining distance
                    total_dist = 0.0
                    for i in range(len(route) - 1):
                        t1 = route[i][0]
                        t2 = route[i + 1][0]
                        dist = t1.location.distance(t2.location)
                        total_dist += dist
                    return total_dist
        except Exception:
            pass
        
        return 0.0
