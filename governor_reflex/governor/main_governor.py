#!/usr/bin/env python3
"""
Governor Process - Alpamayo trajectory planning.

Run with: conda activate alpo && python main_governor.py
"""

import os
import sys
import json
import time
import yaml
import signal
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import threading
import torch



# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from governor.alpamayo_wrapper import AlpamayoWrapper
from governor.input_builder import InputBuilder
from governor.trajectory_decoder import TrajectoryDecoder
from governor.prompt_templates import PromptBuilder
from utils.coordinate_transform import CoordinateTransformer, Pose
from utils.xml_route_writer import XMLRouteWriter
from utils.causation_logger import CausationLogger
from utils.file_lock import FileLock

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [GOVERNOR] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class GovernorProcess:
    """Main Governor process that runs Alpamayo model."""
    
    def __init__(self, config_path: str):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup paths
        self.runtime_dir = Path(self.config['paths']['runtime_dir'])
        self.input_dir = self.runtime_dir / 'input'
        self.output_dir = self.runtime_dir / 'output'
        self.logs_dir = self.runtime_dir / 'logs'
        
        # Ensure directories exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.input_file = self.input_dir / 'request.json'
        self.status_file = self.output_dir / 'status.json'
        self.trajectory_file = self.output_dir / 'trajectory.xml'
        
        # Locks
        self.input_lock = FileLock(str(self.input_dir / '.input.lock'))
        self.output_lock = FileLock(str(self.output_dir / '.output.lock'))
        
        # Components
        self.model: Optional[AlpamayoWrapper] = None
        self.input_builder = InputBuilder()
        self.trajectory_decoder = TrajectoryDecoder()
        self.coord_transformer = CoordinateTransformer()
        self.route_writer = XMLRouteWriter(str(self.output_dir))
        self.causation_logger = CausationLogger(str(self.logs_dir))
        
        # Experiment settings
        include_ped = self.config.get('experiment', {}).get('include_pedestrian_info', True)
        self.prompt_builder = PromptBuilder(include_pedestrian_info=include_ped)
        self.mask_pedestrians = self.config.get('experiment', {}).get('mask_pedestrians_in_images', False)
        
        # State
        self.running = False
        self.last_request_id = None
        self.request_count = 0
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        self.running = False
    
    def initialize(self):
        """Initialize the model and components."""
        logger.info("=" * 60)
        logger.info("GOVERNOR PROCESS INITIALIZING")
        logger.info("=" * 60)
        
        # Update status
        self._update_status("starting", "Initializing model...")
        
        # Load model (with mock fallback)
        model_path = self.config['paths']['model_path']
        quantization = self.config.get('alpamayo', {}).get('quantization', 'int8')
        use_mock = self.config.get('alpamayo', {}).get('use_mock', False)
        
        # Try to load real model, fall back to mock
        try:
            from governor.mock_alpamayo import create_alpamayo_wrapper
            
            self.model = create_alpamayo_wrapper(
                model_path=model_path,
                quantization=quantization,
                device=self.config.get('alpamayo', {}).get('device', 'cuda'),
                max_new_tokens=self.config.get('alpamayo', {}).get('max_new_tokens', 512),
                use_mock=use_mock
            )
        except Exception as e:
            logger.error(f"Failed to create wrapper: {e}")
            # Last resort: use mock directly
            from governor.mock_alpamayo import MockAlpamayoWrapper
            self.model = MockAlpamayoWrapper(model_path, quantization)
            self.model.load()
        
        mem = self.model.get_memory_usage()
        if mem and mem.get('allocated_gb', 0) > 0:
            logger.info(f"GPU Memory: {mem['allocated_gb']:.2f} GB allocated")
        
        # Ready
        self._update_status("ready", "Model loaded, waiting for requests")
        logger.info("Governor ready and waiting for requests")
    
    def _update_status(self, status: str, message: str = None, trajectory_id: str = None):
        """Update the status file."""
        status_data = {
            'governor_status': status,
            'last_update_time': datetime.now().timestamp(),
            'last_trajectory_id': trajectory_id or self.last_request_id,
            'trajectory_valid': status == 'ready' and self.last_request_id is not None,
            'error_message': message if status == 'error' else None,
            'message': message,
            'request_count': self.request_count
        }
        
        with self.output_lock.acquire():
            with open(self.status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
    
    def _check_for_request(self) -> Optional[dict]:
        """Check if there's a new input request."""
        if not self.input_file.exists():
            return None
        
        try:
            with self.input_lock.acquire():
                with open(self.input_file, 'r') as f:
                    request = json.load(f)
                
                # Check if it's a new request
                request_id = request.get('request_id')
                if request_id == self.last_request_id:
                    return None  # Already processed
                
                return request
        except Exception as e:
            logger.warning(f"Error reading request: {e}")
            return None
    
    def process_request(self, request: dict):
        """Process a trajectory planning request."""
        request_id = request.get('request_id', f'req_{self.request_count}')
        logger.info(f"Processing request: {request_id}")
        
        start_time = time.time()
        self._update_status("processing", f"Processing {request_id}")
        
        try:
            # Extract ego state
            ego_state = request.get('ego_state', {})
            position = ego_state.get('position', {'x': 0, 'y': 0, 'z': 0})
            rotation = ego_state.get('rotation', {'pitch': 0, 'yaw': 0, 'roll': 0})
            velocity = ego_state.get('velocity', {'x': 0, 'y': 0, 'z': 0})
            
            current_speed_ms = (velocity['x']**2 + velocity['y']**2 + velocity['z']**2) ** 0.5
            current_speed_kmh = current_speed_ms * 3.6
            
            ego_pose = Pose(
                x=position['x'], y=position['y'], z=position['z'],
                pitch=rotation['pitch'], yaw=rotation['yaw'], roll=rotation['roll']
            )
            
            # Build prompt
            speed_limit = request.get('speed_limit', 50.0)
            route_context = request.get('route_context', 'Continue following the road')
            pedestrians = request.get('pedestrians', []) if request.get('include_pedestrian_info', True) else []
            
            prompt = self.prompt_builder.build_prompt(
                speed_kmh=current_speed_kmh,
                speed_limit=speed_limit,
                route_context=route_context,
                pedestrians=pedestrians
            )
            
            # Build model inputs
            model_inputs = self.input_builder.build_model_inputs(
                request=request,
                prompt=prompt,
            )
            
            # Generate trajectory
            logger.info("Running Alpamayo inference...")
            inference_start = time.time()
            
            # Prepare egomotion in combined format for wrapper
            ego_history_xyz = model_inputs['ego_history_xyz']
            ego_history_rot = model_inputs['ego_history_rot']
            
            # Combine xyz and rot into single tensor for generate()
            # ego_history_rot is (16, 3, 3), flatten to (16, 9)
            ego_rot_flat = ego_history_rot.reshape(-1, 9)
            egomotion_combined = torch.cat([ego_history_xyz, ego_rot_flat], dim=-1)  # (16, 12)
            
            output = self.model.generate(
                images=model_inputs['images'],
                egomotion_history=egomotion_combined,
                prompt=model_inputs['prompt'],
                current_speed=model_inputs['current_speed']
            )
            
            inference_time = time.time() - inference_start
            logger.info(f"Inference completed in {inference_time:.2f}s")
            
            # Decode trajectory
            ego_waypoints, causation = self.trajectory_decoder.decode_from_model_output(
                output,
                initial_speed=current_speed_ms
            )
            
            # Transform to world coordinates
            world_waypoints = self.coord_transformer.ego_to_world(ego_waypoints, ego_pose)
            
            # Write trajectory file
            with self.output_lock.acquire():
                self.route_writer.write_route(
                    waypoints=world_waypoints,
                    route_id=request_id,
                    town=self.config.get('carla', {}).get('map', 'Town10HD_Opt'),
                    filename='trajectory.xml'
                )
            
            # Log causation
            self.causation_logger.log(
                request_id=request_id,
                causation_text=causation,
                timestamp=request.get('timestamp')
            )
            
            # Update state
            self.last_request_id = request_id
            self.request_count += 1
            
            total_time = time.time() - start_time
            
            logger.info(f"Request {request_id} completed in {total_time:.2f}s")
            logger.info(f"  - Inference: {inference_time:.2f}s")
            logger.info(f"  - Waypoints: {len(world_waypoints)}")
            
            self._update_status("ready", f"Completed {request_id}", trajectory_id=request_id)
            
        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            import traceback
            traceback.print_exc()
            self._update_status("error", str(e))
    
    def run(self):
        """Main loop."""
        self.running = True
        poll_interval = 0.1  # Check for new requests every 100ms
        
        logger.info("Starting main loop...")
        
        while self.running:
            try:
                # Check for new request
                request = self._check_for_request()
                
                if request is not None:
                    self.process_request(request)
                else:
                    time.sleep(poll_interval)
                    
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1.0)
        
        logger.info("Governor shutting down...")
        self._update_status("stopped", "Governor process stopped")
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up...")
        self.running = False


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Governor Process - Alpamayo trajectory planning')
    parser.add_argument(
        '--config', 
        type=str, 
        default=str(Path(__file__).parent.parent / 'config.yaml'),
        help='Path to config file'
    )
    args = parser.parse_args()
    
    governor = GovernorProcess(args.config)
    
    try:
        governor.initialize()
        governor.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        governor.cleanup()


if __name__ == '__main__':
    main()
