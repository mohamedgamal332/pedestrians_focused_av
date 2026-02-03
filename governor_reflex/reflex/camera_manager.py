"""Camera manager for 4-camera Alpamayo input."""

import carla
import numpy as np
from queue import Queue, Empty
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque
from PIL import Image
import threading
import time


class CameraManager:
    """Manage 4 cameras for Alpamayo input."""
    
    CAMERA_CONFIGS = {
        'front_wide': {
            'location': (2.2, 0.0, 1.5),
            'rotation': (0.0, 0.0, 0.0),
            'fov': 120
        },
        'front_tele': {
            'location': (2.2, 0.0, 1.5),
            'rotation': (0.0, 0.0, 0.0),
            'fov': 50
        },
        'cross_left': {
            'location': (1.5, -0.8, 1.5),
            'rotation': (0.0, -60.0, 0.0),
            'fov': 90
        },
        'cross_right': {
            'location': (1.5, 0.8, 1.5),
            'rotation': (0.0, 60.0, 0.0),
            'fov': 90
        }
    }
    
    def __init__(
        self,
        world: carla.World,
        image_width: int = 1920,
        image_height: int = 1080,
        frames_history: int = 4,
        save_dir: Optional[str] = None
    ):
        self.world = world
        self.blueprint_library = world.get_blueprint_library()
        self.image_width = image_width
        self.image_height = image_height
        self.frames_history = frames_history
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage
        self.sensors: Dict[str, carla.Sensor] = {}
        self.queues: Dict[str, Queue] = {}
        self.frame_buffers: Dict[str, deque] = {}
        self._frame_counter = 0
        self._lock = threading.Lock()
        
        # Initialize frame buffers
        for cam_name in self.CAMERA_CONFIGS.keys():
            self.frame_buffers[cam_name] = deque(maxlen=frames_history)
    
    def setup_on_vehicle(self, vehicle: carla.Vehicle, custom_configs: Optional[Dict] = None):
        """
        Setup all 4 cameras on the vehicle.
        
        Args:
            vehicle: CARLA vehicle actor
            custom_configs: Optional custom camera configurations
        """
        configs = custom_configs or self.CAMERA_CONFIGS
        
        for cam_name, cam_config in configs.items():
            self._spawn_camera(vehicle, cam_name, cam_config)
        
        print(f"[CameraManager] Spawned {len(self.sensors)} cameras")
    
    def _spawn_camera(self, vehicle: carla.Vehicle, name: str, config: Dict):
        """Spawn a single camera."""
        # Get blueprint
        bp = self.blueprint_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.image_width))
        bp.set_attribute('image_size_y', str(self.image_height))
        bp.set_attribute('fov', str(config.get('fov', 90)))
        bp.set_attribute('motion_blur_intensity', '0.0')
        
        # Create transform
        loc = config.get('location', (0, 0, 2))
        rot = config.get('rotation', (0, 0, 0))
        transform = carla.Transform(
            carla.Location(x=loc[0], y=loc[1], z=loc[2]),
            carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2])
        )
        
        # Spawn sensor
        camera = self.world.spawn_actor(bp, transform, attach_to=vehicle)
        
        # Setup queue and listener
        self.queues[name] = Queue()
        camera.listen(lambda image, n=name: self._on_image(n, image))
        
        self.sensors[name] = camera
    
    def _on_image(self, camera_name: str, image: carla.Image):
        """Callback when image is received."""
        # Convert to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # BGRA
        array = array[:, :, :3]  # Remove alpha -> BGR
        array = array[:, :, ::-1]  # BGR to RGB
        
        with self._lock:
            self.frame_buffers[camera_name].append({
                'frame': self._frame_counter,
                'timestamp': image.timestamp,
                'data': array.copy()
            })
        
        self.queues[camera_name].put(image)
    
    def tick(self):
        """Call after each world tick to update frame counter."""
        with self._lock:
            self._frame_counter += 1
    
    def get_current_frames(self) -> Dict[str, np.ndarray]:
        """Get the most recent frame from each camera."""
        frames = {}
        
        with self._lock:
            for cam_name, buffer in self.frame_buffers.items():
                if buffer:
                    frames[cam_name] = buffer[-1]['data']
                else:
                    frames[cam_name] = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        
        return frames
    
    def get_frame_history(self, num_frames: int = 4) -> Dict[str, List[np.ndarray]]:
        """
        Get historical frames from each camera.
        
        Args:
            num_frames: Number of historical frames to return
            
        Returns:
            Dict mapping camera name to list of frames (oldest to newest)
        """
        history = {}
        
        with self._lock:
            for cam_name, buffer in self.frame_buffers.items():
                frames = list(buffer)[-num_frames:]
                
                # Pad with black frames if not enough history
                while len(frames) < num_frames:
                    frames.insert(0, {
                        'frame': -1,
                        'timestamp': 0,
                        'data': np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
                    })
                
                history[cam_name] = [f['data'] for f in frames]
        
        return history
    
    def save_frames(self, request_id: str) -> Dict[str, List[str]]:
        """
        Save current frame history to disk.
        
        Args:
            request_id: Request identifier for filename
            
        Returns:
            Dict mapping camera name to list of saved file paths
        """
        if self.save_dir is None:
            return {}
        
        saved_paths = {}
        history = self.get_frame_history()
        
        for cam_name, frames in history.items():
            cam_dir = self.save_dir / cam_name
            cam_dir.mkdir(exist_ok=True)
            
            paths = []
            for i, frame in enumerate(frames):
                filename = f"{request_id}_{cam_name}_{i}.jpg"
                filepath = cam_dir / filename
                
                img = Image.fromarray(frame)
                img.save(filepath, quality=90)
                paths.append(str(filepath))
            
            saved_paths[cam_name] = paths
        
        return saved_paths
    
    def destroy(self):
        """Destroy all camera sensors."""
        for name, sensor in self.sensors.items():
            if sensor is not None and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        
        self.sensors.clear()
        self.queues.clear()
        self.frame_buffers.clear()
        print("[CameraManager] All cameras destroyed")
