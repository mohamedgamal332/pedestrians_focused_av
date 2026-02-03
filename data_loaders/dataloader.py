#!/usr/bin/env python3
"""
CARLA Stereo Pedestrian Dataset Dataloader

A PyTorch-compatible dataloader for the CARLA stereo pedestrian dataset
with COCO-17 keypoint annotations.

Author: Assistant
License: MIT
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings

# Optional PyTorch support
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # Fallback for type hints


# =============================================================================
# Data Structures
# =============================================================================

class VisibilityState(Enum):
    """Keypoint visibility states."""
    VISIBLE = "visible"
    OCCLUDED = "occluded"
    OUT_OF_FRAME = "out_of_frame"
    BEHIND_CAMERA = "behind_camera"


class AnimationState(Enum):
    """Pedestrian animation states based on movement speed."""
    STANDING = "standing"   # speed < 0.1 m/s
    WALKING = "walking"     # 0.1 <= speed < 2.0 m/s
    RUNNING = "running"     # speed >= 2.0 m/s


class BehaviorState(Enum):
    """Pedestrian behavior/intent states (risk-relevant for analysis)."""
    WALKING = "walking"
    RUNNING = "running"
    CROSSING = "crossing"
    WAITING_TO_CROSS = "waiting_to_cross"
    IDLE = "idle"
    # Risk-relevant: distraction / communication
    WAVING = "waving"
    WAVING_WALKING = "waving_walking"  # waving while walking
    TEXTING = "texting"
    CALLING = "calling"
    TALKING = "talking"


# COCO-17 Keypoint definitions
COCO_KEYPOINTS = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle',    # 16
]

COCO_SKELETON = [
    [0, 1], [0, 2], [1, 3], [2, 4],           # Face
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # Arms
    [5, 11], [6, 12], [11, 12],               # Torso
    [11, 13], [13, 15], [12, 14], [14, 16],   # Legs
]


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float           # Focal length x
    fy: float           # Focal length y
    cx: float           # Principal point x
    cy: float           # Principal point y
    width: int          # Image width
    height: int         # Image height
    fov: float          # Field of view in degrees
    
    @property
    def K(self) -> np.ndarray:
        """Get 3x3 intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CameraIntrinsics':
        K = np.array(data['K'])
        return cls(
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            width=data['width'],
            height=data['height'],
            fov=data['fov']
        )


@dataclass
class StereoIntrinsics:
    """Stereo camera intrinsic parameters."""
    left: CameraIntrinsics
    right: CameraIntrinsics
    baseline: float  # Distance between cameras in meters
    
    @classmethod
    def from_dict(cls, data: dict) -> 'StereoIntrinsics':
        return cls(
            left=CameraIntrinsics.from_dict(data['left']),
            right=CameraIntrinsics.from_dict(data['right']),
            baseline=data['baseline']
        )


@dataclass
class Transform:
    """3D transform with location and rotation."""
    x: float
    y: float
    z: float
    pitch: float
    yaw: float
    roll: float
    
    @property
    def location(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)
    
    @property
    def rotation(self) -> np.ndarray:
        return np.array([self.pitch, self.yaw, self.roll], dtype=np.float32)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Transform':
        loc = data['location']
        rot = data['rotation']
        return cls(
            x=loc['x'], y=loc['y'], z=loc['z'],
            pitch=rot['pitch'], yaw=rot['yaw'], roll=rot['roll']
        )


@dataclass
class Keypoint:
    """Single keypoint with multi-camera projections."""
    name: str
    coco_index: int
    world_position: np.ndarray  # [x, y, z] in world coordinates
    
    # Per-camera data: dict with 'left' and 'right' keys
    pixel_left: Optional[np.ndarray] = None      # [x, y] pixel coordinates
    pixel_right: Optional[np.ndarray] = None
    depth_left: Optional[float] = None           # Depth in meters
    depth_right: Optional[float] = None
    camera_relative_left: Optional[np.ndarray] = None   # [x, y, z] relative to camera
    camera_relative_right: Optional[np.ndarray] = None
    visibility_left: VisibilityState = VisibilityState.OUT_OF_FRAME
    visibility_right: VisibilityState = VisibilityState.OUT_OF_FRAME
    
    @property
    def is_visible_left(self) -> bool:
        return self.visibility_left == VisibilityState.VISIBLE
    
    @property
    def is_visible_right(self) -> bool:
        return self.visibility_right == VisibilityState.VISIBLE
    
    @classmethod
    def from_dict(cls, name: str, data: dict) -> 'Keypoint':
        world_pos = data.get('world_position', {})
        cameras = data.get('cameras', {})
        
        left_cam = cameras.get('left', {})
        right_cam = cameras.get('right', {})
        
        def get_array(d, key):
            v = d.get(key)
            if v is None:
                return None
            if isinstance(v, dict):
                return np.array([v['x'], v['y'], v['z']], dtype=np.float32)
            return np.array(v, dtype=np.float32)
        
        def get_visibility(d):
            state = d.get('visibility_state', 'out_of_frame')
            try:
                return VisibilityState(state)
            except ValueError:
                return VisibilityState.OUT_OF_FRAME
        
        return cls(
            name=name,
            coco_index=data.get('coco_index', COCO_KEYPOINTS.index(name) if name in COCO_KEYPOINTS else -1),
            world_position=np.array([world_pos.get('x', 0), world_pos.get('y', 0), world_pos.get('z', 0)], dtype=np.float32),
            pixel_left=get_array(left_cam, 'pixel'),
            pixel_right=get_array(right_cam, 'pixel'),
            depth_left=left_cam.get('depth'),
            depth_right=right_cam.get('depth'),
            camera_relative_left=get_array(left_cam, 'camera_relative'),
            camera_relative_right=get_array(right_cam, 'camera_relative'),
            visibility_left=get_visibility(left_cam),
            visibility_right=get_visibility(right_cam),
        )


@dataclass
class Pedestrian:
    """Single pedestrian with all annotations."""
    id: int
    transform: Transform
    speed: float                    # Actual speed in m/s (from position tracking)
    distance_to_ego: Optional[float]  # Distance to ego vehicle in meters
    animation_state: AnimationState   # Physical animation: standing/walking/running
    behavior: BehaviorState           # Semantic state: crossing, waiting, etc.
    assigned_behavior: BehaviorState  # Originally assigned behavior
    is_runner: bool                   # Whether assigned as runner
    is_visible: bool                  # Visible in at least one camera
    
    # Visibility per camera
    visible_keypoints_left: int
    visible_keypoints_right: int
    total_keypoints: int
    fully_visible_left: bool
    fully_visible_right: bool
    
    # Jaywalking info
    is_jaywalker: bool
    jaywalking_state: str
    blocking_vehicles: List[int]
    
    # Keypoints
    keypoints: Dict[str, Keypoint]
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Pedestrian':
        # Parse animation state
        try:
            anim_state = AnimationState(data.get('animation_state', 'standing'))
        except ValueError:
            anim_state = AnimationState.STANDING
        
        # Parse behavior states
        try:
            behavior = BehaviorState(data.get('behavior', 'walking'))
        except ValueError:
            behavior = BehaviorState.WALKING
        
        try:
            assigned = BehaviorState(data.get('assigned_behavior', 'walking'))
        except ValueError:
            assigned = BehaviorState.WALKING
        
        # Parse visibility
        visibility = data.get('visibility', {})
        left_vis = visibility.get('left', {})
        right_vis = visibility.get('right', {})
        
        # Parse keypoints
        keypoints = {}
        for kp_name, kp_data in data.get('keypoints', {}).items():
            keypoints[kp_name] = Keypoint.from_dict(kp_name, kp_data)
        
        # Parse jaywalking info
        jw = data.get('jaywalking', {})
        
        return cls(
            id=data['id'],
            transform=Transform.from_dict(data['world_transform']),
            speed=data.get('speed', 0.0),
            distance_to_ego=data.get('distance_to_ego'),
            animation_state=anim_state,
            behavior=behavior,
            assigned_behavior=assigned,
            is_runner=data.get('is_runner', False),
            is_visible=data.get('visible_in_frame', False),
            visible_keypoints_left=left_vis.get('visible_keypoints', 0),
            visible_keypoints_right=right_vis.get('visible_keypoints', 0),
            total_keypoints=left_vis.get('total_keypoints', 17),
            fully_visible_left=left_vis.get('fully_visible', False),
            fully_visible_right=right_vis.get('fully_visible', False),
            is_jaywalker=jw.get('is_jaywalker', False),
            jaywalking_state=jw.get('jaywalking_state', 'normal'),
            blocking_vehicles=jw.get('blocking_vehicles', []),
            keypoints=keypoints,
        )
    
    def get_keypoints_array(self, camera: str = 'left', include_visibility: bool = True) -> np.ndarray:
        """Get keypoints as numpy array in COCO format.
        
        Args:
            camera: 'left' or 'right'
            include_visibility: If True, returns [N, 3] with (x, y, v), else [N, 2]
        
        Returns:
            Array of shape [17, 3] or [17, 2] with keypoint coordinates
            Visibility: 0=not labeled, 1=labeled but not visible, 2=labeled and visible
        """
        n_keypoints = 17
        if include_visibility:
            result = np.zeros((n_keypoints, 3), dtype=np.float32)
        else:
            result = np.zeros((n_keypoints, 2), dtype=np.float32)
        
        for kp_name, kp in self.keypoints.items():
            idx = kp.coco_index
            if idx < 0 or idx >= n_keypoints:
                continue
            
            if camera == 'left':
                pixel = kp.pixel_left
                visible = kp.is_visible_left
            else:
                pixel = kp.pixel_right
                visible = kp.is_visible_right
            
            if pixel is not None:
                result[idx, 0] = pixel[0]
                result[idx, 1] = pixel[1]
                if include_visibility:
                    result[idx, 2] = 2 if visible else 1
        
        return result
    
    def get_camera_relative_positions(self, camera: str = 'left') -> np.ndarray:
        """Get 3D positions relative to camera.
        
        Returns:
            Array of shape [17, 3] with (x_right, y_down, z_forward) coordinates
        """
        result = np.zeros((17, 3), dtype=np.float32)
        
        for kp_name, kp in self.keypoints.items():
            idx = kp.coco_index
            if idx < 0 or idx >= 17:
                continue
            
            if camera == 'left':
                pos = kp.camera_relative_left
            else:
                pos = kp.camera_relative_right
            
            if pos is not None:
                result[idx] = pos
        
        return result
    
    def get_depths(self, camera: str = 'left') -> np.ndarray:
        """Get depth values for each keypoint.
        
        Returns:
            Array of shape [17] with depth values in meters (0 if not available)
        """
        result = np.zeros(17, dtype=np.float32)
        
        for kp_name, kp in self.keypoints.items():
            idx = kp.coco_index
            if idx < 0 or idx >= 17:
                continue
            
            if camera == 'left':
                depth = kp.depth_left
            else:
                depth = kp.depth_right
            
            if depth is not None:
                result[idx] = depth
        
        return result
    
    def get_bounding_box(self, camera: str = 'left', padding: float = 0.1) -> Optional[np.ndarray]:
        """Calculate bounding box from visible keypoints.
        
        Args:
            camera: 'left' or 'right'
            padding: Padding as fraction of box size
        
        Returns:
            Array [x1, y1, x2, y2] or None if no visible keypoints
        """
        keypoints = self.get_keypoints_array(camera, include_visibility=True)
        visible_mask = keypoints[:, 2] > 0
        
        if not visible_mask.any():
            return None
        
        visible_kps = keypoints[visible_mask, :2]
        x1, y1 = visible_kps.min(axis=0)
        x2, y2 = visible_kps.max(axis=0)
        
        # Add padding
        w, h = x2 - x1, y2 - y1
        x1 -= w * padding
        y1 -= h * padding
        x2 += w * padding
        y2 += h * padding
        
        return np.array([x1, y1, x2, y2], dtype=np.float32)


@dataclass
class Vehicle:
    """Vehicle annotation."""
    id: int
    type: str
    transform: Transform
    velocity: np.ndarray  # [vx, vy, vz]
    is_ego: bool = False
    
    @classmethod
    def from_dict(cls, data: dict, is_ego: bool = False) -> 'Vehicle':
        vel = data.get('velocity', {})
        return cls(
            id=data['id'],
            type=data.get('type', 'unknown'),
            transform=Transform.from_dict(data['transform']),
            velocity=np.array([vel.get('x', 0), vel.get('y', 0), vel.get('z', 0)], dtype=np.float32),
            is_ego=is_ego,
        )


@dataclass
class Weather:
    """Weather conditions."""
    preset: str
    cloudiness: float
    precipitation: float
    sun_altitude_angle: float
    sun_azimuth_angle: float
    fog_density: float
    wetness: float
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Weather':
        return cls(
            preset=data.get('preset', 'unknown'),
            cloudiness=data.get('cloudiness', 0),
            precipitation=data.get('precipitation', 0),
            sun_altitude_angle=data.get('sun_altitude_angle', 70),
            sun_azimuth_angle=data.get('sun_azimuth_angle', 0),
            fog_density=data.get('fog_density', 0),
            wetness=data.get('wetness', 0),
        )


@dataclass
class FrameAnnotation:
    """Complete annotation for a single frame."""
    frame_id: int
    timestamp: float
    weather: Weather
    ego_vehicle: Optional[Vehicle]
    vehicles: List[Vehicle]
    pedestrians: List[Pedestrian]
    camera_transforms: Dict[str, Transform]
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FrameAnnotation':
        # Parse ego vehicle
        ego_data = data.get('ego_vehicle')
        ego = Vehicle.from_dict(ego_data, is_ego=True) if ego_data else None
        
        # Parse other vehicles
        vehicles = [Vehicle.from_dict(v) for v in data.get('vehicles', [])]
        
        # Parse pedestrians
        pedestrians = [Pedestrian.from_dict(p) for p in data.get('pedestrians', [])]
        
        # Parse camera transforms
        cam_transforms = {}
        for side, t_data in data.get('camera_transforms', {}).items():
            if t_data:
                cam_transforms[side] = Transform.from_dict(t_data)
        
        return cls(
            frame_id=data['frame_id'],
            timestamp=data['timestamp'],
            weather=Weather.from_dict(data.get('weather', {})),
            ego_vehicle=ego,
            vehicles=vehicles,
            pedestrians=pedestrians,
            camera_transforms=cam_transforms,
        )
    
    def get_visible_pedestrians(self, camera: str = 'left', 
                                 min_keypoints: int = 1,
                                 max_distance: Optional[float] = None,
                                 fully_visible_only: bool = False) -> List[Pedestrian]:
        """Filter pedestrians by visibility criteria.
        
        Args:
            camera: 'left' or 'right'
            min_keypoints: Minimum number of visible keypoints
            max_distance: Maximum distance to ego vehicle (meters)
            fully_visible_only: Only return fully visible pedestrians
        
        Returns:
            List of pedestrians matching criteria
        """
        result = []
        for ped in self.pedestrians:
            # Check keypoint count
            if camera == 'left':
                vis_count = ped.visible_keypoints_left
                is_full = ped.fully_visible_left
            else:
                vis_count = ped.visible_keypoints_right
                is_full = ped.fully_visible_right
            
            if vis_count < min_keypoints:
                continue
            
            if fully_visible_only and not is_full:
                continue
            
            # Check distance
            if max_distance is not None:
                if ped.distance_to_ego is None or ped.distance_to_ego > max_distance:
                    continue
            
            result.append(ped)
        
        return result


@dataclass 
class FrameData:
    """Complete data for a single frame including images."""
    annotation: FrameAnnotation
    rgb_left: Optional[np.ndarray] = None
    rgb_right: Optional[np.ndarray] = None
    depth_left: Optional[np.ndarray] = None
    depth_right: Optional[np.ndarray] = None


# =============================================================================
# Dataset Class
# =============================================================================

class CARLAStereoPedestrianDataset(Dataset):
    """
    PyTorch-compatible dataset for CARLA stereo pedestrian data.
    
    Dataset Structure:
        session_YYYYMMDD_HHMMSS/
            rgb_left/           - Left camera RGB images (PNG)
            rgb_right/          - Right camera RGB images (PNG)
            depth_left/         - Left depth maps (16-bit PNG, millimeters)
            depth_right/        - Right depth maps (16-bit PNG, millimeters)
            annotations/        - Per-frame JSON annotations
            showcase/           - Annotated visualization frames
            camera_intrinsics.json
            keypoint_config.json
            road_layout.json
            session_metadata.json
    
    Annotation Fields per Pedestrian:
        - id: Unique pedestrian ID
        - animation_state: Physical state (standing/walking/running)
        - behavior: Semantic state (walking/running/crossing/waiting_to_cross/idle)
        - speed: Actual speed in m/s (calculated from position tracking)
        - distance_to_ego: Distance to ego vehicle in meters
        - is_runner: Whether assigned as runner
        - is_jaywalker: Whether assigned as jaywalker
        - keypoints: COCO-17 keypoints with:
            - world_position: [x, y, z] in world coordinates
            - cameras.left/right:
                - pixel: [x, y] image coordinates
                - depth: Distance to camera in meters
                - camera_relative: [x, y, z] position relative to camera
                - visibility_state: visible/occluded/out_of_frame/behind_camera
    
    Usage:
        dataset = CARLAStereoPedestrianDataset('/path/to/session')
        
        # Get single frame
        frame = dataset[0]
        
        # Access data
        rgb_left = frame.rgb_left                    # [H, W, 3] uint8
        depth_left = frame.depth_left                # [H, W] float32 (meters)
        pedestrians = frame.annotation.pedestrians   # List[Pedestrian]
        
        # Get keypoints for first pedestrian
        ped = pedestrians[0]
        keypoints = ped.get_keypoints_array('left')  # [17, 3] with (x, y, visibility)
        depths = ped.get_depths('left')              # [17] depth values
        cam_pos = ped.get_camera_relative_positions('left')  # [17, 3]
        
        # Filter visible pedestrians
        visible = frame.annotation.get_visible_pedestrians(
            camera='left', min_keypoints=10, max_distance=15.0
        )
    """
    
    def __init__(
        self,
        session_path: Union[str, Path],
        load_images: bool = True,
        load_depth: bool = True,
        cameras: List[str] = ['left', 'right'],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        max_frames: Optional[int] = None,
        filter_min_pedestrians: int = 0,
        filter_min_visible_keypoints: int = 0,
        filter_max_distance: Optional[float] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            session_path: Path to session directory
            load_images: Whether to load RGB images
            load_depth: Whether to load depth maps
            cameras: Which cameras to load ('left', 'right', or both)
            transform: Transform to apply to images
            target_transform: Transform to apply to annotations
            max_frames: Maximum number of frames to load (None for all)
            filter_min_pedestrians: Only include frames with at least this many pedestrians
            filter_min_visible_keypoints: Minimum visible keypoints per pedestrian
            filter_max_distance: Maximum distance for pedestrians (meters)
        """
        self.session_path = Path(session_path)
        self.load_images = load_images
        self.load_depth = load_depth
        self.cameras = cameras
        self.transform = transform
        self.target_transform = target_transform
        self.filter_min_pedestrians = filter_min_pedestrians
        self.filter_min_visible_keypoints = filter_min_visible_keypoints
        self.filter_max_distance = filter_max_distance
        
        # Validate session path
        if not self.session_path.exists():
            raise FileNotFoundError(f"Session path not found: {self.session_path}")
        
        # Load metadata
        self.metadata = self._load_json('session_metadata.json')
        self.intrinsics = self._load_intrinsics()
        self.keypoint_config = self._load_json('keypoint_config.json', required=False)
        
        # Get frame list
        self.frame_ids = self._get_frame_ids(max_frames)
        
        # Apply filtering if needed
        if filter_min_pedestrians > 0 or filter_min_visible_keypoints > 0 or filter_max_distance is not None:
            self.frame_ids = self._filter_frames()
        
        print(f"Loaded dataset with {len(self.frame_ids)} frames from {self.session_path.name}")
    
    def _load_json(self, filename: str, required: bool = True) -> Optional[dict]:
        """Load a JSON file from the session directory."""
        path = self.session_path / filename
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Required file not found: {path}")
            return None
        with open(path, 'r') as f:
            return json.load(f)
    
    def _load_intrinsics(self) -> Optional[StereoIntrinsics]:
        """Load camera intrinsics."""
        data = self._load_json('camera_intrinsics.json', required=False)
        if data is None:
            return None
        return StereoIntrinsics.from_dict(data)
    
    def _get_frame_ids(self, max_frames: Optional[int]) -> List[int]:
        """Get list of available frame IDs."""
        annotation_dir = self.session_path / 'annotations'
        if not annotation_dir.exists():
            raise FileNotFoundError(f"Annotations directory not found: {annotation_dir}")
        
        frame_ids = []
        for f in sorted(annotation_dir.glob('frame_*.json')):
            try:
                frame_id = int(f.stem.split('_')[1])
                frame_ids.append(frame_id)
            except (ValueError, IndexError):
                continue
        
        if max_frames is not None:
            frame_ids = frame_ids[:max_frames]
        
        return frame_ids
    
    def _filter_frames(self) -> List[int]:
        """Filter frames based on criteria."""
        filtered = []
        for frame_id in self.frame_ids:
            ann = self._load_annotation(frame_id)
            
            # Count qualifying pedestrians
            qualifying = 0
            for ped in ann.pedestrians:
                # Check visibility
                vis_count = max(ped.visible_keypoints_left, ped.visible_keypoints_right)
                if vis_count < self.filter_min_visible_keypoints:
                    continue
                
                # Check distance
                if self.filter_max_distance is not None:
                    if ped.distance_to_ego is None or ped.distance_to_ego > self.filter_max_distance:
                        continue
                
                qualifying += 1
            
            if qualifying >= self.filter_min_pedestrians:
                filtered.append(frame_id)
        
        print(f"Filtered {len(self.frame_ids)} -> {len(filtered)} frames")
        return filtered
    
    def _load_annotation(self, frame_id: int) -> FrameAnnotation:
        """Load annotation for a specific frame."""
        path = self.session_path / 'annotations' / f'frame_{frame_id:06d}.json'
        with open(path, 'r') as f:
            data = json.load(f)
        return FrameAnnotation.from_dict(data)
    
    def _load_image(self, subdir: str, frame_id: int) -> Optional[np.ndarray]:
        """Load an image from a subdirectory."""
        path = self.session_path / subdir / f'frame_{frame_id:06d}.png'
        if not path.exists():
            return None
        
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        
        # Convert BGR to RGB for color images
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    def _load_depth(self, subdir: str, frame_id: int) -> Optional[np.ndarray]:
        """Load depth map and convert to meters."""
        path = self.session_path / subdir / f'frame_{frame_id:06d}.png'
        if not path.exists():
            return None
        
        # Load as 16-bit (stored in millimeters)
        depth_mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            return None
        
        # Convert to meters
        depth_m = depth_mm.astype(np.float32) / 1000.0
        return depth_m
    
    def __len__(self) -> int:
        return len(self.frame_ids)
    
    def __getitem__(self, idx: int) -> FrameData:
        """Get a single frame with all data."""
        frame_id = self.frame_ids[idx]
        
        # Load annotation
        annotation = self._load_annotation(frame_id)
        
        # Load images
        rgb_left = None
        rgb_right = None
        depth_left = None
        depth_right = None
        
        if self.load_images:
            if 'left' in self.cameras:
                rgb_left = self._load_image('rgb_left', frame_id)
            if 'right' in self.cameras:
                rgb_right = self._load_image('rgb_right', frame_id)
        
        if self.load_depth:
            if 'left' in self.cameras:
                depth_left = self._load_depth('depth_left', frame_id)
            if 'right' in self.cameras:
                depth_right = self._load_depth('depth_right', frame_id)
        
        # Apply transforms
        if self.transform is not None:
            if rgb_left is not None:
                rgb_left = self.transform(rgb_left)
            if rgb_right is not None:
                rgb_right = self.transform(rgb_right)
        
        if self.target_transform is not None:
            annotation = self.target_transform(annotation)
        
        return FrameData(
            annotation=annotation,
            rgb_left=rgb_left,
            rgb_right=rgb_right,
            depth_left=depth_left,
            depth_right=depth_right,
        )
    
    def get_frame_by_id(self, frame_id: int) -> FrameData:
        """Get frame by its actual frame ID (not index)."""
        if frame_id not in self.frame_ids:
            raise ValueError(f"Frame {frame_id} not in dataset")
        idx = self.frame_ids.index(frame_id)
        return self[idx]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate dataset statistics."""
        stats = {
            'total_frames': len(self.frame_ids),
            'total_pedestrians': 0,
            'visible_pedestrians': 0,
            'fully_visible_pedestrians': 0,
            'animation_states': {s.value: 0 for s in AnimationState},
            'behaviors': {b.value: 0 for b in BehaviorState},
            'jaywalkers': 0,
            'runners': 0,
            'avg_speed': [],
            'avg_distance': [],
            'avg_visible_keypoints': [],
        }
        
        for frame_id in self.frame_ids:
            ann = self._load_annotation(frame_id)
            
            for ped in ann.pedestrians:
                stats['total_pedestrians'] += 1
                
                if ped.is_visible:
                    stats['visible_pedestrians'] += 1
                
                if ped.fully_visible_left or ped.fully_visible_right:
                    stats['fully_visible_pedestrians'] += 1
                
                stats['animation_states'][ped.animation_state.value] += 1
                stats['behaviors'][ped.behavior.value] += 1
                
                if ped.is_jaywalker:
                    stats['jaywalkers'] += 1
                if ped.is_runner:
                    stats['runners'] += 1
                
                stats['avg_speed'].append(ped.speed)
                if ped.distance_to_ego is not None:
                    stats['avg_distance'].append(ped.distance_to_ego)
                stats['avg_visible_keypoints'].append(
                    max(ped.visible_keypoints_left, ped.visible_keypoints_right)
                )
        
        # Calculate averages
        if stats['avg_speed']:
            stats['avg_speed'] = np.mean(stats['avg_speed'])
        else:
            stats['avg_speed'] = 0
        
        if stats['avg_distance']:
            stats['avg_distance'] = np.mean(stats['avg_distance'])
        else:
            stats['avg_distance'] = 0
        
        if stats['avg_visible_keypoints']:
            stats['avg_visible_keypoints'] = np.mean(stats['avg_visible_keypoints'])
        else:
            stats['avg_visible_keypoints'] = 0
        
        return stats
    
    def create_dataloader(self, batch_size: int = 1, shuffle: bool = True, 
                          num_workers: int = 0, **kwargs) -> 'DataLoader':
        """Create a PyTorch DataLoader for this dataset."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DataLoader")
        
        return DataLoader(
            self, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            **kwargs
        )
    
    @staticmethod
    def collate_fn(batch: List[FrameData]) -> Dict[str, Any]:
        """Custom collate function for batching FrameData objects."""
        result = {
            'annotations': [item.annotation for item in batch],
            'rgb_left': None,
            'rgb_right': None,
            'depth_left': None,
            'depth_right': None,
        }
        
        # Stack images if available and all have same shape
        for key in ['rgb_left', 'rgb_right', 'depth_left', 'depth_right']:
            images = [getattr(item, key) for item in batch]
            if all(img is not None for img in images):
                try:
                    if TORCH_AVAILABLE:
                        # Convert to tensor and stack
                        tensors = [torch.from_numpy(img) for img in images]
                        result[key] = torch.stack(tensors)
                    else:
                        result[key] = np.stack(images)
                except ValueError:
                    # Images have different shapes, keep as list
                    result[key] = images
        
        return result


# =============================================================================
# Utility Functions
# =============================================================================

def visualize_frame(frame: FrameData, camera: str = 'left', 
                    show_keypoints: bool = True,
                    show_skeleton: bool = True,
                    show_bboxes: bool = True,
                    show_labels: bool = True) -> np.ndarray:
    """
    Visualize a frame with annotations.
    
    Args:
        frame: FrameData object
        camera: 'left' or 'right'
        show_keypoints: Draw keypoint circles
        show_skeleton: Draw skeleton connections
        show_bboxes: Draw bounding boxes
        show_labels: Draw text labels
    
    Returns:
        Annotated image as numpy array
    """
    # Get image
    if camera == 'left':
        img = frame.rgb_left
    else:
        img = frame.rgb_right
    
    if img is None:
        raise ValueError(f"No {camera} image available")
    
    # Convert to BGR for OpenCV
    img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    
    # Colors for different states
    animation_colors = {
        AnimationState.STANDING: (128, 128, 128),
        AnimationState.WALKING: (0, 255, 0),
        AnimationState.RUNNING: (0, 165, 255),
    }
    
    behavior_colors = {
        BehaviorState.WALKING: (0, 255, 0),
        BehaviorState.RUNNING: (0, 165, 255),
        BehaviorState.CROSSING: (0, 255, 255),
        BehaviorState.WAITING_TO_CROSS: (0, 0, 255),
        BehaviorState.IDLE: (128, 128, 128),
    }
    
    for ped in frame.annotation.pedestrians:
        if not ped.is_visible:
            continue
        
        keypoints = ped.get_keypoints_array(camera, include_visibility=True)
        color = behavior_colors.get(ped.behavior, (255, 255, 255))
        
        # Draw skeleton
        if show_skeleton:
            for i, j in COCO_SKELETON:
                if keypoints[i, 2] > 0 and keypoints[j, 2] > 0:
                    pt1 = tuple(keypoints[i, :2].astype(int))
                    pt2 = tuple(keypoints[j, :2].astype(int))
                    cv2.line(img, pt1, pt2, color, 2)
        
        # Draw keypoints
        if show_keypoints:
            for i, (x, y, v) in enumerate(keypoints):
                if v > 0:
                    pt = (int(x), int(y))
                    kp_color = (0, 255, 0) if v == 2 else (0, 0, 255)
                    cv2.circle(img, pt, 4, kp_color, -1)
                    cv2.circle(img, pt, 4, (255, 255, 255), 1)
        
        # Draw bounding box
        if show_bboxes:
            bbox = ped.get_bounding_box(camera)
            if bbox is not None:
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        if show_labels:
            bbox = ped.get_bounding_box(camera)
            if bbox is not None:
                x1, y1 = bbox[:2].astype(int)
                
                label_lines = [
                    f"ID:{ped.id}",
                    f"Anim:{ped.animation_state.value}",
                    f"Behav:{ped.behavior.value}",
                    f"Spd:{ped.speed:.2f}m/s",
                ]
                if ped.distance_to_ego:
                    label_lines.append(f"Dist:{ped.distance_to_ego:.1f}m")
                
                for i, line in enumerate(label_lines):
                    y = y1 - 10 - (len(label_lines) - i - 1) * 15
                    cv2.putText(img, line, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.4, (0, 0, 0), 2)
                    cv2.putText(img, line, (x1, y), cv2.FONT_HERSHEY_SIMPLEX,
                               0.4, (255, 255, 255), 1)
    
    # Add frame info
    info_text = f"Frame: {frame.annotation.frame_id} | Pedestrians: {len(frame.annotation.pedestrians)}"
    cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def visualize_depth(depth: np.ndarray, max_depth: float = 100.0) -> np.ndarray:
    """Convert depth map to colorized visualization."""
    depth_clipped = np.clip(depth, 0, max_depth)
    depth_normalized = (depth_clipped / max_depth * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='CARLA Dataset Dataloader Demo')
    parser.add_argument('session_path', type=str, help='Path to session directory')
    parser.add_argument('--show', action='store_true', help='Display sample frames')
    parser.add_argument('--stats', action='store_true', help='Print dataset statistics')
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from: {args.session_path}")
    dataset = CARLAStereoPedestrianDataset(
        args.session_path,
        load_images=True,
        load_depth=True,
    )
    
    print(f"Dataset size: {len(dataset)} frames")
    print(f"Stereo baseline: {dataset.intrinsics.baseline}m" if dataset.intrinsics else "No intrinsics")
    
    if args.stats:
        print("\nCalculating statistics...")
        stats = dataset.get_statistics()
        print(f"\nDataset Statistics:")
        print(f"  Total frames: {stats['total_frames']}")
        print(f"  Total pedestrians: {stats['total_pedestrians']}")
        print(f"  Visible pedestrians: {stats['visible_pedestrians']}")
        print(f"  Fully visible: {stats['fully_visible_pedestrians']}")
        print(f"  Jaywalkers: {stats['jaywalkers']}")
        print(f"  Runners: {stats['runners']}")
        print(f"  Avg speed: {stats['avg_speed']:.2f} m/s")
        print(f"  Avg distance: {stats['avg_distance']:.2f} m")
        print(f"  Avg visible keypoints: {stats['avg_visible_keypoints']:.1f}")
        print(f"\nAnimation states:")
        for state, count in stats['animation_states'].items():
            print(f"    {state}: {count}")
        print(f"\nBehaviors:")
        for behavior, count in stats['behaviors'].items():
            print(f"    {behavior}: {count}")
    
    # Load and display sample frame
    print("\nLoading sample frame...")
    frame = dataset[0]
    
    print(f"\nFrame {frame.annotation.frame_id}:")
    print(f"  Timestamp: {frame.annotation.timestamp:.3f}")
    print(f"  Weather: {frame.annotation.weather.preset}")
    print(f"  Pedestrians: {len(frame.annotation.pedestrians)}")
    print(f"  Vehicles: {len(frame.annotation.vehicles)}")
    
    if frame.rgb_left is not None:
        print(f"  RGB Left shape: {frame.rgb_left.shape}")
    if frame.depth_left is not None:
        print(f"  Depth Left shape: {frame.depth_left.shape}")
        print(f"  Depth range: {frame.depth_left.min():.2f} - {frame.depth_left.max():.2f} m")
    
    # Show pedestrian details
    visible_peds = frame.annotation.get_visible_pedestrians('left', min_keypoints=5)
    print(f"\n  Visible pedestrians (>=5 keypoints): {len(visible_peds)}")
    
    for ped in visible_peds[:3]:  # Show first 3
        print(f"\n  Pedestrian {ped.id}:")
        print(f"    Animation: {ped.animation_state.value}")
        print(f"    Behavior: {ped.behavior.value}")
        print(f"    Speed: {ped.speed:.2f} m/s")
        print(f"    Distance: {ped.distance_to_ego:.2f} m" if ped.distance_to_ego else "    Distance: N/A")
        print(f"    Visible keypoints: L={ped.visible_keypoints_left}, R={ped.visible_keypoints_right}")
        print(f"    Fully visible: L={ped.fully_visible_left}, R={ped.fully_visible_right}")
        print(f"    Is jaywalker: {ped.is_jaywalker}")
        print(f"    Is runner: {ped.is_runner}")
        
        # Get keypoint data
        kps = ped.get_keypoints_array('left')
        depths = ped.get_depths('left')
        cam_pos = ped.get_camera_relative_positions('left')
        
        print(f"    Keypoints shape: {kps.shape}")
        print(f"    Depths shape: {depths.shape}")
        print(f"    Camera-relative positions shape: {cam_pos.shape}")
    
    if args.show:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # RGB Left with annotations
            if frame.rgb_left is not None:
                vis_img = visualize_frame(frame, 'left')
                axes[0, 0].imshow(vis_img)
                axes[0, 0].set_title('Left Camera - Annotated')
                axes[0, 0].axis('off')
            
            # RGB Right with annotations
            if frame.rgb_right is not None:
                vis_img = visualize_frame(frame, 'right')
                axes[0, 1].imshow(vis_img)
                axes[0, 1].set_title('Right Camera - Annotated')
                axes[0, 1].axis('off')
            
            # Depth Left
            if frame.depth_left is not None:
                depth_vis = visualize_depth(frame.depth_left)
                axes[1, 0].imshow(depth_vis)
                axes[1, 0].set_title('Left Depth')
                axes[1, 0].axis('off')
            
            # Depth Right
            if frame.depth_right is not None:
                depth_vis = visualize_depth(frame.depth_right)
                axes[1, 1].imshow(depth_vis)
                axes[1, 1].set_title('Right Depth')
                axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig('sample_frame.png', dpi=150)
            print("\nSaved visualization to sample_frame.png")
            plt.show()
            
        except ImportError:
            print("\nMatplotlib not available for visualization")