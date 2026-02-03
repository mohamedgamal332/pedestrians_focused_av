#!/usr/bin/env python3
"""
Reflex Process - CaRL control with Governor integration.

Run with: conda activate PCLA && python main_reflex.py
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import os
import sys
import json
import time
import yaml
import signal
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import threading

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/jemmi/PCLA')

import carla

from reflex.camera_manager import CameraManager
from reflex.egomotion_buffer import EgomotionBuffer
from reflex.pedestrian_tracker import PedestrianTracker
from reflex.trajectory_manager import TrajectoryManager
from reflex.route_injector import RouteInjector
from utils.file_lock import FileLock

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [REFLEX] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class ReflexProcess:
    """Main Reflex process that runs CaRL with Governor integration."""
    
    def __init__(self, config_path: str):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Paths
        self.runtime_dir = Path(self.config['paths']['runtime_dir'])
        self.input_dir = self.runtime_dir / 'input'
        self.output_dir = self.runtime_dir / 'output'
        self.logs_dir = self.runtime_dir / 'logs'
        self.cameras_dir = self.runtime_dir / 'cameras'
        
        # Ensure directories
        for d in [self.input_dir, self.output_dir, self.logs_dir, self.cameras_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.request_file = self.input_dir / 'request.json'
        self.status_file = self.output_dir / 'status.json'
        
        # Locks
        self.input_lock = FileLock(str(self.input_dir / '.input.lock'))
        
        # CARLA
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.vehicle: Optional[carla.Actor] = None
        self.traffic_manager = None
        
        # Components
        self.camera_manager: Optional[CameraManager] = None
        self.egomotion_buffer: Optional[EgomotionBuffer] = None
        self.pedestrian_tracker: Optional[PedestrianTracker] = None
        self.trajectory_manager: Optional[TrajectoryManager] = None
        self.route_injector: Optional[RouteInjector] = None
        self.pcla = None
        
        # State
        self.running = False
        self.request_count = 0
        self.last_replan_time = 0
        self.simulation_time = 0
        
        # Experiment setting
        self.include_pedestrian_info = self.config.get('experiment', {}).get('include_pedestrian_info', True)
        
        # Timing
        self.replan_interval = self.config.get('timing', {}).get('replan_interval_seconds', 3.0)
        self.min_buffer = self.config.get('timing', {}).get('min_trajectory_buffer_seconds', 2.0)
        
        # Metrics
        self.metrics = {
            'start_time': None,
            'total_steps': 0,
            'total_distance': 0.0,
            'replans': 0,
            'speeds': [],
            'infractions': []
        }
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        self.running = False
    
    def connect_carla(self):
        """Connect to CARLA simulator."""
        host = self.config.get('carla', {}).get('host', 'localhost')
        port = self.config.get('carla', {}).get('port', 2000)
        timeout = self.config.get('carla', {}).get('timeout', 30.0)
        
        logger.info(f"Connecting to CARLA at {host}:{port}...")
        
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        
        # Load map
        map_name = self.config.get('carla', {}).get('map', 'Town10HD_Opt')
        logger.info(f"Loading map: {map_name}")
        self.client.load_world(map_name)
        time.sleep(3.0)
        
        self.world = self.client.get_world()
        
        # Setup synchronous mode
        settings = self.world.get_settings()
        tick_rate = self.config.get('carla', {}).get('tick_rate', 15)
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / tick_rate
        self.world.apply_settings(settings)
        
        # Traffic manager
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)
        
        logger.info("Connected to CARLA")
    
    def spawn_ego_vehicle(self):
        """Spawn the ego vehicle."""
        bp_library = self.world.get_blueprint_library()
        vehicle_bp = bp_library.filter('model3')[0]
        
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = spawn_points[31] if len(spawn_points) > 31 else spawn_points[0]
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        logger.info(f"Spawned ego vehicle at {spawn_point.location}")
        
        self.world.tick()
    
    def setup_traffic(self):
        """Spawn traffic vehicles and pedestrians."""
        num_vehicles = self.config.get('traffic', {}).get('num_vehicles', 25)
        num_pedestrians = self.config.get('traffic', {}).get('num_pedestrians', 300)
        
        logger.info(f"Spawning {num_vehicles} vehicles and {num_pedestrians} pedestrians...")
        
        # Spawn vehicles
        vehicle_bps = [bp for bp in self.world.get_blueprint_library().filter('vehicle.*')
                       if int(bp.get_attribute('number_of_wheels')) == 4]
        spawn_points = self.world.get_map().get_spawn_points()
        
        vehicles_spawned = 0
        for i, sp in enumerate(spawn_points[:num_vehicles]):
            bp = vehicle_bps[i % len(vehicle_bps)]
            bp.set_attribute('role_name', 'autopilot')
            try:
                v = self.world.spawn_actor(bp, sp)
                v.set_autopilot(True, self.traffic_manager.get_port())
                vehicles_spawned += 1
            except:
                pass
        
        # Spawn pedestrians
        walker_bps = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        
        pedestrians_spawned = 0
        for _ in range(num_pedestrians):
            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                continue
            
            bp = walker_bps[pedestrians_spawned % len(walker_bps)]
            try:
                walker = self.world.spawn_actor(bp, carla.Transform(loc))
                controller = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
                
                self.world.tick()
                
                controller.start()
                controller.go_to_location(self.world.get_random_location_from_navigation())
                controller.set_max_speed(1.4)
                
                pedestrians_spawned += 1
            except:
                pass
        
        logger.info(f"Spawned {vehicles_spawned} vehicles and {pedestrians_spawned} pedestrians")
    
    def setup_components(self):
        """Setup all components."""
        # Camera manager
        self.camera_manager = CameraManager(
            world=self.world,
            image_width=self.config.get('cameras', {}).get('image_width', 1920),
            image_height=self.config.get('cameras', {}).get('image_height', 1080),
            frames_history=self.config.get('cameras', {}).get('frames_history', 4),
            save_dir=str(self.cameras_dir)
        )
        self.camera_manager.setup_on_vehicle(self.vehicle)
        
        # Egomotion buffer
        tick_rate = self.config.get('carla', {}).get('tick_rate', 15)
        self.egomotion_buffer = EgomotionBuffer(
            history_size=self.config.get('timing', {}).get('egomotion_history_size', 16),
            target_frequency=10.0,
            carla_tick_rate=tick_rate
        )
        
        # Pedestrian tracker
        self.pedestrian_tracker = PedestrianTracker(
            world=self.world,
            detection_radius=self.config.get('pedestrian', {}).get('detection_radius', 50.0),
            standing_threshold=self.config.get('pedestrian', {}).get('standing_threshold', 0.5),
            walking_threshold=self.config.get('pedestrian', {}).get('walking_threshold', 2.0)
        )
        
        # Trajectory manager already created in initialize()
        # Just update if needed
        if self.trajectory_manager is None:
            self.trajectory_manager = TrajectoryManager(
                runtime_dir=str(self.runtime_dir),
                min_buffer_seconds=self.min_buffer
            )
        
        # Route injector
        self.route_injector = RouteInjector(world=self.world)
        
        logger.info("All components initialized")
    
    def setup_pcla(self, initial_route: str):
        """Setup PCLA/CaRL agent."""
        from PCLA import PCLA
        
        logger.info("Initializing PCLA/CaRL agent...")
        
        self.pcla = PCLA(
            agent='carl_carlv11',
            vehicle=self.vehicle,
            route=initial_route,
            client=self.client
        )
        
        self.route_injector.set_pcla_instance(self.pcla)
        
        logger.info("PCLA agent initialized")
    
    def create_initial_route(self) -> str:
        """Create an initial route for PCLA initialization."""
        # Create a simple initial route
        route_file = self.runtime_dir / 'initial_route.xml'
        
        # Get vehicle position
        transform = self.vehicle.get_transform()
        loc = transform.location
        
        # Create straight line route
        route_xml = f'''<?xml version='1.0' encoding='UTF-8'?>
        <route id="initial" town="{self.config.get('carla', {}).get('map', 'Town10HD_Opt')}">
        <waypoint x="{loc.x}" y="{loc.y}" z="{loc.z}" pitch="0" yaw="{transform.rotation.yaw}" roll="0"/>
        <waypoint x="{loc.x + 10}" y="{loc.y}" z="{loc.z}" pitch="0" yaw="{transform.rotation.yaw}" roll="0"/>
        <waypoint x="{loc.x + 20}" y="{loc.y}" z="{loc.z}" pitch="0" yaw="{transform.rotation.yaw}" roll="0"/>
        </route>'''
        
        with open(route_file, 'w') as f:
            f.write(route_xml)
        
        return str(route_file)
    
    def send_planning_request(self):
        """Send a planning request to the Governor."""
        self.request_count += 1
        request_id = f"req_{self.request_count:06d}"
        
        # Get current state
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed_ms = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        
        # Get pedestrians
        pedestrians = self.pedestrian_tracker.update(self.vehicle, self.simulation_time)
        
        # Save camera frames
        camera_paths = self.camera_manager.save_frames(request_id)
        
        # Build request
        request = {
            'request_id': request_id,
            'timestamp': self.simulation_time,
            'ego_state': {
                'position': {
                    'x': transform.location.x,
                    'y': transform.location.y,
                    'z': transform.location.z
                },
                'rotation': {
                    'pitch': transform.rotation.pitch,
                    'yaw': transform.rotation.yaw,
                    'roll': transform.rotation.roll
                },
                'velocity': {
                    'x': velocity.x,
                    'y': velocity.y,
                    'z': velocity.z
                },
                'speed_kmh': speed_ms * 3.6
            },
            'egomotion_history': self.egomotion_buffer.get_history(),
            'camera_paths': camera_paths,
            'pedestrians': pedestrians if self.include_pedestrian_info else [],
            'route_context': 'Continue following the road',
            'speed_limit': self.vehicle.get_speed_limit() or 50.0,
            'include_pedestrian_info': self.include_pedestrian_info
        }
        
        # Write request
        with self.input_lock.acquire():
            with open(self.request_file, 'w') as f:
                json.dump(request, f, indent=2)
        
        self.last_replan_time = self.simulation_time
        logger.info(f"Sent planning request {request_id}")
    
    def check_and_update_trajectory(self):
        """Check for new trajectory and update route."""
        if self.trajectory_manager.load_trajectory():
            # New trajectory available
            success = self.route_injector.inject_from_trajectory_manager(
                self.trajectory_manager
            )
            if success:
                self.metrics['replans'] += 1
                logger.info("Injected new trajectory from Governor")
    
    def should_request_replan(self) -> bool:
        """Check if we should request a new plan from Governor."""
        # Check trajectory buffer
        if self.trajectory_manager.needs_replan():
            return True
        
        # Check time since last replan
        time_since_replan = self.simulation_time - self.last_replan_time
        if time_since_replan >= self.replan_interval:
            return True
        
        return False
    
    def initialize(self):
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("REFLEX PROCESS INITIALIZING")
        logger.info("=" * 60)
        
        # Create trajectory manager FIRST (needed for governor check)
        self.trajectory_manager = TrajectoryManager(
            runtime_dir=str(self.runtime_dir),
            min_buffer_seconds=self.min_buffer
        )
        
        # Wait for Governor
        logger.info("Waiting for Governor to be ready...")
        if not self.trajectory_manager.wait_for_governor(timeout=120.0):
            logger.error("Governor not ready within timeout!")
            raise RuntimeError("Governor not ready")
        logger.info("Governor is ready")
        
        # Connect to CARLA
        self.connect_carla()
        
        # Spawn vehicle
        self.spawn_ego_vehicle()
        
        # Setup traffic
        self.setup_traffic()
        
        # Wait for world to settle
        for _ in range(10):
            self.world.tick()
        
        # Setup remaining components (cameras, egomotion, pedestrian tracker)
        self.setup_components()
        
        # Create initial route and setup PCLA
        initial_route = self.create_initial_route()
        self.setup_pcla(initial_route)
        
        # Initial planning request
        self.send_planning_request()
        
        # Wait for first trajectory
        logger.info("Waiting for initial trajectory from Governor...")
        timeout = 30.0
        start = time.time()
        while time.time() - start < timeout:
            if self.trajectory_manager.load_trajectory():
                self.route_injector.inject_from_trajectory_manager(self.trajectory_manager)
                break
            time.sleep(0.5)
        
        logger.info("Reflex initialized and ready")
    
    def run(self):
        """Main control loop."""
        self.running = True
        self.metrics['start_time'] = datetime.now().isoformat()
        
        logger.info("Starting main control loop...")
        
        spectator = self.world.get_spectator()
        last_location = self.vehicle.get_location()
        
        try:
            while self.running:
                # World tick
                self.world.tick()
                self.simulation_time += 1.0 / self.config.get('carla', {}).get('tick_rate', 15)
                
                # Update buffers
                self.egomotion_buffer.update(self.vehicle, self.simulation_time)
                self.camera_manager.tick()
                
                # Get and apply control from CaRL
                try:
                    action = self.pcla.get_action()
                    if action is None:
                        logger.info("Route completed")
                        break
                    self.vehicle.apply_control(action)
                except Exception as e:
                    logger.warning(f"Control error: {e}")
                
                # Update spectator
                transform = self.vehicle.get_transform()
                spectator.set_transform(carla.Transform(
                    transform.location + carla.Location(z=50),
                    carla.Rotation(pitch=-90)
                ))
                
                # Track metrics
                current_location = self.vehicle.get_location()
                self.metrics['total_distance'] += current_location.distance(last_location)
                last_location = current_location
                
                velocity = self.vehicle.get_velocity()
                speed_kmh = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5 * 3.6
                self.metrics['speeds'].append(speed_kmh)
                self.metrics['total_steps'] += 1
                
                # Check for new trajectory
                self.check_and_update_trajectory()
                
                # Request replan if needed
                if self.should_request_replan():
                    self.send_planning_request()
                
                # Log progress
                if self.metrics['total_steps'] % 100 == 0:
                    logger.info(
                        f"Step {self.metrics['total_steps']} | "
                        f"Speed {speed_kmh:.1f} km/h | "
                        f"Distance {self.metrics['total_distance']:.1f}m | "
                        f"Replans {self.metrics['replans']}"
                    )
                    
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        
        self.save_metrics()
    
    def save_metrics(self):
        """Save run metrics."""
        import numpy as np
        
        self.metrics['end_time'] = datetime.now().isoformat()
        self.metrics['avg_speed'] = float(np.mean(self.metrics['speeds'])) if self.metrics['speeds'] else 0
        self.metrics['max_speed'] = float(np.max(self.metrics['speeds'])) if self.metrics['speeds'] else 0
        self.metrics['include_pedestrian_info'] = self.include_pedestrian_info
        
        # Remove large arrays for JSON
        metrics_to_save = {k: v for k, v in self.metrics.items() if k != 'speeds'}
        
        metrics_file = self.logs_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        logger.info(f"Metrics saved to {metrics_file}")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("RUN SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Steps:     {self.metrics['total_steps']}")
        logger.info(f"Total Distance:  {self.metrics['total_distance']:.2f} m")
        logger.info(f"Avg Speed:       {self.metrics['avg_speed']:.2f} km/h")
        logger.info(f"Max Speed:       {self.metrics['max_speed']:.2f} km/h")
        logger.info(f"Replans:         {self.metrics['replans']}")
        logger.info(f"Pedestrian Info: {'Enabled' if self.include_pedestrian_info else 'Disabled'}")
        logger.info("=" * 60)
    
    def cleanup(self):
        """Cleanup all resources."""
        logger.info("Cleaning up...")
        
        self.running = False
        
        # Cleanup camera manager
        if self.camera_manager:
            self.camera_manager.destroy()
        
        # Cleanup PCLA
        if self.pcla:
            try:
                self.pcla.cleanup()
            except:
                pass
        
        # Destroy vehicle
        if self.vehicle and self.vehicle.is_alive:
            self.vehicle.destroy()
        
        # Reset CARLA settings
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        
        logger.info("Cleanup complete")


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Reflex Process - CaRL with Governor integration')
    parser.add_argument(
        '--config',
        type=str,
        default=str(Path(__file__).parent.parent / 'config.yaml'),
        help='Path to config file'
    )
    parser.add_argument(
        '--with-pedestrians',
        action='store_true',
        default=None,
        help='Enable pedestrian info (overrides config)'
    )
    parser.add_argument(
        '--without-pedestrians',
        action='store_true',
        default=None,
        help='Disable pedestrian info (overrides config)'
    )
    args = parser.parse_args()
    
    reflex = ReflexProcess(args.config)
    
    # Override pedestrian setting from command line
    if args.with_pedestrians:
        reflex.include_pedestrian_info = True
        logger.info("Pedestrian info ENABLED via command line")
    elif args.without_pedestrians:
        reflex.include_pedestrian_info = False
        logger.info("Pedestrian info DISABLED via command line")
    
    try:
        reflex.initialize()
        reflex.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        reflex.cleanup()


if __name__ == '__main__':
    main()
