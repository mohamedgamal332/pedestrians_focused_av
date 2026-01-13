import carla
import requests
import time
from goal_manager import GoalManager
from reflex_agent import ReflexAgent
from safety_switch import EvasiveArbiter
from actuator_pid import PIDController

class AutonomousBridge:
    def __init__(self, vehicle, world, govenor_url):
        self.vehicle = vehicle
        self.world = world
        self.gov_url = govenor_url # DigitalOcean API
        
        # Initialize Sub-modules
        self.goal_mgr = GoalManager(world)
        self.reflex = ReflexAgent(weights="active_reflex.pth")
        self.safety = EvasiveArbiter(critical_dist=3.0)
        self.controller = PIDController(vehicle)
        
        self.current_trajectory = None

    def run_step(self):
        # 1. Perception & Local Goal (Local HPC)
        bev_data = get_bev_tensor()        # Fixed BEV input
        ped_data = get_mmpose_data()       # From teammate
        local_goal = self.goal_mgr.get_horizon_target(self.vehicle)
        
        # 2. Async Call to Governor (DigitalOcean)
        # We only call the Governor at 10Hz to save bandwidth/latency
        if self.world.get_snapshot().frame % 10 == 0:
            self.update_governor_trajectory(bev_data, ped_data, local_goal)

        # 3. Reflex Execution (100Hz Local)
        # Reflex calculates the "Ideal Action" based on Governor waypoints
        reflex_v, reflex_theta = self.reflex.get_action(
            self.current_trajectory, 
            self.vehicle.get_velocity()
        )

        # 4. Safety Switch (Evasive Residual Logic)
        # This overrides the 'stickiness' to the trajectory if an obstacle is near
        d_min = get_custom_depth_min() 
        final_v, final_theta, mode = self.safety.mediate(
            reflex_v, reflex_theta, d_min
        )

        # 5. Physical Actuation
        self.controller.apply(final_v, final_theta)

        # 6. Log to Memory (For Llama 3.3 Auditor)
        self.log_experience(mode, d_min)

    def update_governor_trajectory(self, bev, peds, goal):
        try:
            response = requests.post(self.gov_url, json={
                "bev": bev.tolist(), "peds": peds, "goal": goal
            }, timeout=0.2)
            data = response.json()
            self.current_trajectory = data['waypoints']
            # Store reasoning trace for the Auditor
            save_trace_to_buffer(data['reasoning'])
        except Exception as e:
            print(f"Governor Latency Spike: {e}")

# --- Execution Loop ---
# Initializing CARLA and running the bridge