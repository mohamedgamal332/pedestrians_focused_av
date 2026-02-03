"""Build inputs for Alpamayo model from JSON request."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import torch


class InputBuilder:
    """Build Alpamayo model inputs from runtime data."""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (1920, 1080),  # Original size, processor will resize
        num_cameras: int = 4,
        frames_history: int = 4,
        egomotion_history_size: int = 16
    ):
        self.image_size = image_size
        self.num_cameras = num_cameras
        self.frames_history = frames_history
        self.egomotion_history_size = egomotion_history_size
        
        self.camera_names = ['front_wide', 'front_tele', 'cross_left', 'cross_right']
    
    def load_input_request(self, input_path: str) -> Dict:
        """Load input request from JSON file."""
        with open(input_path, 'r') as f:
            return json.load(f)
    
    def prepare_images(
        self,
        camera_paths: Dict[str, List[str]],
    ) -> torch.Tensor:
        """
        Load and prepare camera images.
        
        Args:
            camera_paths: Dict mapping camera name to list of frame paths
            
        Returns:
            Tensor of shape (num_cameras * frames, 3, H, W)
        """
        all_images = []
        
        for cam_name in self.camera_names:
            paths = camera_paths.get(cam_name, [])
            
            for i in range(self.frames_history):
                if i < len(paths) and Path(paths[i]).exists():
                    img = Image.open(paths[i]).convert('RGB')
                    img_array = np.array(img, dtype=np.float32) / 255.0
                else:
                    # Create black image if path doesn't exist
                    img_array = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.float32)
                
                # Convert to tensor (C, H, W)
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                all_images.append(img_tensor)
        
        # Stack all images: (num_cameras * frames, 3, H, W)
        images = torch.stack(all_images, dim=0)
        
        return images
    
    def prepare_egomotion(self, egomotion_history: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare egomotion history for Alpamayo.
        
        Args:
            egomotion_history: List of egomotion dicts with position and rotation
            
        Returns:
            Tuple of (ego_history_xyz, ego_history_rot) tensors
            - ego_history_xyz: (history_size, 3)
            - ego_history_rot: (history_size, 3, 3)
        """
        xyz_data = []
        rot_data = []
        
        for i in range(self.egomotion_history_size):
            if i < len(egomotion_history):
                ego = egomotion_history[i]
                pos = ego.get('position', {'x': 0, 'y': 0, 'z': 0})
                
                # Position
                xyz_data.append([pos['x'], pos['y'], pos['z']])
                
                # Get rotation matrix or compute from euler angles
                if 'rotation_matrix' in ego:
                    rot = np.array(ego['rotation_matrix']).reshape(3, 3)
                else:
                    rot_dict = ego.get('rotation', {'pitch': 0, 'yaw': 0, 'roll': 0})
                    rot = self._euler_to_rotation_matrix(
                        rot_dict.get('pitch', 0),
                        rot_dict.get('yaw', 0),
                        rot_dict.get('roll', 0)
                    )
                rot_data.append(rot)
            else:
                # Pad with identity/zeros
                xyz_data.append([0.0, 0.0, 0.0])
                rot_data.append(np.eye(3))
        
        ego_history_xyz = torch.tensor(xyz_data, dtype=torch.float32)
        ego_history_rot = torch.tensor(np.array(rot_data), dtype=torch.float32)
        
        return ego_history_xyz, ego_history_rot
    
    def _euler_to_rotation_matrix(
        self,
        pitch: float,
        yaw: float,
        roll: float
    ) -> np.ndarray:
        """Convert Euler angles (degrees) to rotation matrix."""
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
        roll = np.radians(roll)
        
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
    
    def build_model_inputs(
        self,
        request: Dict,
        prompt: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Build complete model inputs from request.
        
        Args:
            request: Input request dictionary
            prompt: Text prompt for the model
            
        Returns:
            Dictionary of model inputs
        """
        # Prepare images
        images = self.prepare_images(
            request.get('camera_paths', {}),
        )
        
        # Prepare egomotion
        ego_history_xyz, ego_history_rot = self.prepare_egomotion(
            request.get('egomotion_history', [])
        )
        
        # Get current speed
        ego_state = request.get('ego_state', {})
        velocity = ego_state.get('velocity', {'x': 0, 'y': 0, 'z': 0})
        current_speed = np.sqrt(
            velocity['x']**2 + velocity['y']**2 + velocity['z']**2
        )
        
        return {
            'images': images,
            'ego_history_xyz': ego_history_xyz,
            'ego_history_rot': ego_history_rot,
            'prompt': prompt,
            'current_speed': current_speed
        }
