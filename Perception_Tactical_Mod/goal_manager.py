import carla
import math
from agents.navigation.global_route_planner import GlobalRoutePlanner

class GoalManager:
    def __init__(self, world, sampling_resolution=2.0):
        self.world = world
        self.map = world.get_map()
        
        # Initialize the Global Route Planner
        self._planner = GlobalRoutePlanner(self.map, sampling_resolution)
        self.global_route = []
        self.horizon_distance = 25.0  # The "Street-Wise" lookahead distance (meters)

    def set_destination(self, start_location, end_location):
        """Generates the full path from A to B."""
        self.global_route = self._planner.trace_route(start_location, end_location)
        print(f"Global Route Generated: {len(self.global_route)} waypoints.")

    def get_horizon_target(self, vehicle):
        """
        Finds the point on the route that is exactly 'horizon_distance' 
        ahead of the vehicle's current position.
        """
        if not self.global_route:
            return None

        ego_loc = vehicle.get_location()
        
        # 1. Find the index of the waypoint closest to the vehicle
        closest_idx = 0
        min_dist = float('inf')
        
        # Optimize: Only search the first 50 points of the current route buffer
        search_range = min(len(self.global_route), 50)
        for i in range(search_range):
            dist = ego_loc.distance(self.global_route[i][0].transform.location)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # 2. Prune the route (remove waypoints we already passed)
        self.global_route = self.global_route[closest_idx:]

        # 3. Walk down the route until we reach the 25m horizon
        accumulated_dist = 0.0
        target_waypoint = self.global_route[0][0] # Default to current if route is short

        for i in range(1, len(self.global_route)):
            prev_loc = self.global_route[i-1][0].transform.location
            curr_loc = self.global_route[i][0].transform.location
            accumulated_dist += prev_loc.distance(curr_loc)
            
            if accumulated_dist >= self.horizon_distance:
                target_waypoint = self.global_route[i][0]
                break
        
        # Return as [x, y, z] for the Governor's API
        loc = target_waypoint.transform.location
        return [loc.x, loc.y, loc.z]

    def is_goal_reached(self, vehicle, threshold=3.0):
        if not self.global_route: return True
        return vehicle.get_location().distance(self.global_route[-1][0].transform.location) < threshold