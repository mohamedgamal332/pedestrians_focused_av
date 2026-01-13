import carla
from agents.navigation.controller import VehiclePIDController

class PIDController:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        # Standard tuned gains for CARLA 0.9.16
        self._controller = VehiclePIDController(
            vehicle,
            args_lateral={'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.01},
            args_longitudinal={'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': 0.01}
        )

    def apply(self, target_v, target_theta):
        """
        target_v: Speed in m/s
        target_theta: Yaw/Heading in radians
        """
        # 1. Convert Target Theta to a Waypoint transform for the controller
        v_transform = self.vehicle.get_transform()
        
        # Calculate a point 5m ahead in the target direction
        target_loc = v_transform.location
        target_loc.x += 5.0 * np.cos(target_theta)
        target_loc.y += 5.0 * np.sin(target_theta)
        
        target_waypoint = carla.Transform(target_loc)

        # 2. Get PID Control (Speed converted to km/h for CARLA controller)
        control = self._controller.run_step(target_v * 3.6, target_waypoint)
        
        # 3. Apply to vehicle
        self.vehicle.apply_control(control)