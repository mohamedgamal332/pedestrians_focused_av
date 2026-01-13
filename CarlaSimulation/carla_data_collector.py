#!/usr/bin/env python3
"""
CARLA Stereo Pedestrian Data Collection System (FIXED VERSION)
Features:
- Robust connection handling with retry logic
- Jaywalking behavior with traffic safety checking
- Pedestrians wait for safe gap before crossing
- Behavior states: walking, running, waiting_to_cross, crossing, idle
- Occlusion-aware visibility detection
- Skeleton-to-frame synchronization
"""

import carla
import numpy as np
import cv2
import json
import yaml
import random
import argparse
import time
import sys
import socket
from pathlib import Path
from datetime import datetime
from queue import Queue, Empty
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from enum import Enum


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class PedestrianBehavior(Enum):
    WALKING = "walking"
    RUNNING = "running"
    CROSSING = "crossing"
    WAITING_TO_CROSS = "waiting_to_cross"
    IDLE = "idle"


class JaywalkingState(Enum):
    NORMAL = "normal"
    SEARCHING = "searching"
    APPROACHING = "approaching"
    WAITING = "waiting"
    CROSSING = "crossing"
    COMPLETED = "completed"


class VisibilityState(Enum):
    VISIBLE = "visible"
    OCCLUDED = "occluded"
    OUT_OF_FRAME = "out_of_frame"
    BEHIND_CAMERA = "behind_camera"


DEFAULT_SKELETON_LINKS = [
    ["crl_root", "crl_hips__C"],
    ["crl_hips__C", "crl_spine__C"],
    ["crl_spine__C", "crl_spine01__C"],
    ["crl_spine01__C", "crl_shoulder__L"],
    ["crl_shoulder__L", "crl_arm__L"],
    ["crl_arm__L", "crl_foreArm__L"],
    ["crl_foreArm__L", "crl_hand__L"],
    ["crl_hand__L", "crl_handThumb__L"],
    ["crl_hand__L", "crl_handIndex__L"],
    ["crl_hand__L", "crl_handMiddle__L"],
    ["crl_hand__L", "crl_handRing__L"],
    ["crl_hand__L", "crl_handPinky__L"],
    ["crl_spine01__C", "crl_shoulder__R"],
    ["crl_shoulder__R", "crl_arm__R"],
    ["crl_arm__R", "crl_foreArm__R"],
    ["crl_foreArm__R", "crl_hand__R"],
    ["crl_hand__R", "crl_handThumb__R"],
    ["crl_hand__R", "crl_handIndex__R"],
    ["crl_hand__R", "crl_handMiddle__R"],
    ["crl_hand__R", "crl_handRing__R"],
    ["crl_hand__R", "crl_handPinky__R"],
    ["crl_spine01__C", "crl_neck__C"],
    ["crl_neck__C", "crl_Head__C"],
    ["crl_hips__C", "crl_thigh__L"],
    ["crl_thigh__L", "crl_leg__L"],
    ["crl_leg__L", "crl_foot__L"],
    ["crl_foot__L", "crl_toe__L"],
    ["crl_hips__C", "crl_thigh__R"],
    ["crl_thigh__R", "crl_leg__R"],
    ["crl_leg__R", "crl_foot__R"],
    ["crl_foot__R", "crl_toe__R"],
]

SKELETON_COLORS = {
    'spine': (0, 255, 255),
    'left_arm': (255, 0, 0),
    'right_arm': (0, 0, 255),
    'left_leg': (255, 255, 0),
    'right_leg': (0, 255, 0),
    'head': (255, 0, 255),
    'default': (255, 255, 255),
}

BEHAVIOR_COLORS = {
    PedestrianBehavior.WALKING.value: (0, 255, 0),
    PedestrianBehavior.RUNNING.value: (0, 165, 255),
    PedestrianBehavior.CROSSING.value: (255, 255, 0),
    PedestrianBehavior.WAITING_TO_CROSS.value: (0, 0, 255),
    PedestrianBehavior.IDLE.value: (128, 128, 128),
}


def get_bone_color(bone1: str, bone2: str) -> Tuple[int, int, int]:
    """Get color for a skeleton bone link."""
    link_str = f"{bone1}_{bone2}".lower()
    if '__l' in link_str:
        if 'arm' in link_str or 'hand' in link_str or 'shoulder' in link_str:
            return SKELETON_COLORS['left_arm']
        elif 'thigh' in link_str or 'leg' in link_str or 'foot' in link_str or 'toe' in link_str:
            return SKELETON_COLORS['left_leg']
    elif '__r' in link_str:
        if 'arm' in link_str or 'hand' in link_str or 'shoulder' in link_str:
            return SKELETON_COLORS['right_arm']
        elif 'thigh' in link_str or 'leg' in link_str or 'foot' in link_str or 'toe' in link_str:
            return SKELETON_COLORS['right_leg']
    elif 'spine' in link_str or 'hips' in link_str or 'root' in link_str:
        return SKELETON_COLORS['spine']
    elif 'head' in link_str or 'neck' in link_str:
        return SKELETON_COLORS['head']
    return SKELETON_COLORS['default']


def load_skeleton_links(filepath: str) -> List[List[str]]:
    """Load skeleton links from file."""
    links = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and ',' in line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 2:
                        links.append(parts[:2])
    except FileNotFoundError:
        print(f"Skeleton file not found: {filepath}, using defaults")
        return DEFAULT_SKELETON_LINKS
    return links if links else DEFAULT_SKELETON_LINKS


def check_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a port is open and accepting connections."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except socket.error:
        return False


@dataclass
class SensorData:
    """Container for synchronized sensor data."""
    frame: int
    timestamp: float
    rgb_left: Optional[np.ndarray] = None
    rgb_right: Optional[np.ndarray] = None
    depth_left: Optional[np.ndarray] = None
    depth_right: Optional[np.ndarray] = None


@dataclass
class BoneVisibilityResult:
    """Result of bone visibility check."""
    pixel: Optional[Tuple[int, int]]
    bone_depth: float
    rendered_depth: Optional[float]
    visibility_state: VisibilityState
    occlusion_distance: Optional[float]


@dataclass
class JaywalkingInfo:
    """Jaywalking state information for a pedestrian."""
    is_jaywalker: bool = False
    state: JaywalkingState = JaywalkingState.NORMAL
    crossing_point: Optional[carla.Location] = None
    destination_after: Optional[carla.Location] = None
    approach_point: Optional[carla.Location] = None
    road_waypoint: Optional[carla.Waypoint] = None
    wait_start_time: float = 0.0
    last_safety_check: float = 0.0
    last_crossing_attempt: float = 0.0
    crossing_count: int = 0
    original_speed: float = 1.4
    is_safe_to_cross: bool = False
    blocking_vehicles: List[int] = field(default_factory=list)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_session_folder(base_path: str) -> Path:
    """Create session folder structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_path = Path(base_path) / f"session_{timestamp}"
    session_path.mkdir(parents=True, exist_ok=True)
    (session_path / "rgb_left").mkdir(exist_ok=True)
    (session_path / "rgb_right").mkdir(exist_ok=True)
    (session_path / "depth_left").mkdir(exist_ok=True)
    (session_path / "depth_right").mkdir(exist_ok=True)
    (session_path / "annotations").mkdir(exist_ok=True)
    (session_path / "demo").mkdir(exist_ok=True)
    return session_path


def build_projection_matrix(w: int, h: int, fov: float) -> np.ndarray:
    """Build camera projection matrix."""
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc_x: float, loc_y: float, loc_z: float, 
                    world_2_camera: np.ndarray, K: np.ndarray) -> Tuple[Optional[Tuple[int, int]], float]:
    """Project 3D point to image coordinates."""
    bone = np.array([loc_x, loc_y, loc_z, 1])
    point_camera = np.dot(world_2_camera, bone)
    point_camera_std = np.array([point_camera[1], -point_camera[2], point_camera[0]])
    depth = point_camera_std[2]
    if depth <= 0:
        return None, depth
    point_img = np.dot(K, point_camera_std)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return (int(point_img[0]), int(point_img[1])), depth


def check_bone_visibility_with_depth(
    pixel: Optional[Tuple[int, int]],
    bone_depth: float,
    depth_image: np.ndarray,
    width: int,
    height: int,
    depth_tolerance_factor: float = 0.05,
    sample_radius: int = 2
) -> BoneVisibilityResult:
    """Check bone visibility using depth buffer."""
    if bone_depth <= 0:
        return BoneVisibilityResult(
            pixel=None, bone_depth=bone_depth, rendered_depth=None,
            visibility_state=VisibilityState.BEHIND_CAMERA, occlusion_distance=None
        )
    if pixel is None:
        return BoneVisibilityResult(
            pixel=None, bone_depth=bone_depth, rendered_depth=None,
            visibility_state=VisibilityState.BEHIND_CAMERA, occlusion_distance=None
        )
    px, py = pixel
    if not (0 <= px < width and 0 <= py < height):
        return BoneVisibilityResult(
            pixel=pixel, bone_depth=bone_depth, rendered_depth=None,
            visibility_state=VisibilityState.OUT_OF_FRAME, occlusion_distance=None
        )
    x_min = max(0, px - sample_radius)
    x_max = min(width, px + sample_radius + 1)
    y_min = max(0, py - sample_radius)
    y_max = min(height, py + sample_radius + 1)
    depth_patch = depth_image[y_min:y_max, x_min:x_max]
    valid_depths = depth_patch[depth_patch > 0]
    if len(valid_depths) == 0:
        rendered_depth = depth_image[py, px]
    else:
        rendered_depth = np.min(valid_depths)
    if rendered_depth <= 0 or rendered_depth > 999:
        return BoneVisibilityResult(
            pixel=pixel, bone_depth=bone_depth, rendered_depth=float(rendered_depth),
            visibility_state=VisibilityState.VISIBLE, occlusion_distance=None
        )
    depth_tolerance = max(bone_depth * depth_tolerance_factor, 0.3)
    depth_difference = bone_depth - rendered_depth
    if depth_difference <= depth_tolerance:
        return BoneVisibilityResult(
            pixel=pixel, bone_depth=bone_depth, rendered_depth=float(rendered_depth),
            visibility_state=VisibilityState.VISIBLE, occlusion_distance=None
        )
    else:
        return BoneVisibilityResult(
            pixel=pixel, bone_depth=bone_depth, rendered_depth=float(rendered_depth),
            visibility_state=VisibilityState.OCCLUDED, occlusion_distance=float(depth_difference)
        )


def check_pedestrian_visibility_detailed(bones_data: dict, camera_side: str, 
                                          min_visible_bones: int = 3) -> dict:
    """Check overall pedestrian visibility from bone data."""
    total_bones = 0
    visible_bones = 0
    occluded_bones = 0
    out_of_frame_bones = 0
    behind_camera_bones = 0
    for bone_name, bone_data in bones_data.items():
        cam_data = bone_data.get('cameras', {}).get(camera_side, {})
        if not cam_data:
            continue
        total_bones += 1
        state = cam_data.get('visibility_state', 'unknown')
        if state == VisibilityState.VISIBLE.value:
            visible_bones += 1
        elif state == VisibilityState.OCCLUDED.value:
            occluded_bones += 1
        elif state == VisibilityState.OUT_OF_FRAME.value:
            out_of_frame_bones += 1
        elif state == VisibilityState.BEHIND_CAMERA.value:
            behind_camera_bones += 1
    is_visible = visible_bones >= min_visible_bones
    visibility_ratio = visible_bones / total_bones if total_bones > 0 else 0.0
    occlusion_ratio = occluded_bones / total_bones if total_bones > 0 else 0.0
    return {
        'visible': is_visible,
        'total_bones': total_bones,
        'visible_bones': visible_bones,
        'occluded_bones': occluded_bones,
        'out_of_frame_bones': out_of_frame_bones,
        'behind_camera_bones': behind_camera_bones,
        'visibility_ratio': visibility_ratio,
        'occlusion_ratio': occlusion_ratio
    }


def transform_to_dict(transform: carla.Transform) -> Optional[dict]:
    """Convert CARLA transform to dictionary."""
    if transform is None:
        return None
    return {
        'location': {
            'x': float(transform.location.x),
            'y': float(transform.location.y),
            'z': float(transform.location.z)
        },
        'rotation': {
            'pitch': float(transform.rotation.pitch),
            'yaw': float(transform.rotation.yaw),
            'roll': float(transform.rotation.roll)
        }
    }


def location_to_dict(location: carla.Location) -> dict:
    """Convert CARLA location to dictionary."""
    return {
        'x': float(location.x),
        'y': float(location.y),
        'z': float(location.z)
    }


def snapshot_bone_data(walker: carla.Walker) -> Dict[str, Dict]:
    """Capture bone transforms from walker."""
    bones_snapshot = {}
    try:
        walker_bones = walker.get_bones()
        if walker_bones and hasattr(walker_bones, 'bone_transforms'):
            for bone in walker_bones.bone_transforms:
                bone_name = str(bone.name)
                bones_snapshot[bone_name] = {
                    'world': {
                        'location': {
                            'x': float(bone.world.location.x),
                            'y': float(bone.world.location.y),
                            'z': float(bone.world.location.z)
                        },
                        'rotation': {
                            'pitch': float(bone.world.rotation.pitch),
                            'yaw': float(bone.world.rotation.yaw),
                            'roll': float(bone.world.rotation.roll)
                        }
                    },
                    'relative': {
                        'location': {
                            'x': float(bone.relative.location.x),
                            'y': float(bone.relative.location.y),
                            'z': float(bone.relative.location.z)
                        },
                        'rotation': {
                            'pitch': float(bone.relative.rotation.pitch),
                            'yaw': float(bone.relative.rotation.yaw),
                            'roll': float(bone.relative.rotation.roll)
                        }
                    }
                }
    except RuntimeError:
        pass
    return bones_snapshot


class JaywalkingManager:
    """Manages jaywalking behavior for pedestrians."""
    
    def __init__(self, world: carla.World, config: dict):
        self.world = world
        self.map = world.get_map()
        self.config = config.get('jaywalking', {})
        self.enabled = self.config.get('enabled', True)
        self.safety_time_threshold = self.config.get('safety_time_threshold', 4.0)
        self.safety_distance_threshold = self.config.get('safety_distance_threshold', 15.0)
        self.max_wait_time = self.config.get('max_wait_time', 10.0)
        self.check_interval = self.config.get('check_interval', 0.5)
        self.crossing_speed_multiplier = self.config.get('crossing_speed_multiplier', 1.3)
        self.approach_distance = self.config.get('approach_distance', 3.0)
        self.min_crossing_interval = self.config.get('min_crossing_interval', 30.0)
        self.jaywalking_info: Dict[int, JaywalkingInfo] = {}
    
    def register_pedestrian(self, walker_id: int, is_jaywalker: bool, original_speed: float):
        """Register a pedestrian with jaywalking info."""
        self.jaywalking_info[walker_id] = JaywalkingInfo(
            is_jaywalker=is_jaywalker,
            original_speed=original_speed
        )
    
    def unregister_pedestrian(self, walker_id: int):
        """Remove pedestrian from jaywalking tracking."""
        self.jaywalking_info.pop(walker_id, None)
    
    def find_crossing_point(self, walker_location: carla.Location) -> Optional[Tuple[carla.Waypoint, carla.Location, carla.Location]]:
        """Find a road to cross near the pedestrian."""
        waypoint = self.map.get_waypoint(
            walker_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if waypoint is None:
            return None
        
        distance_to_road = walker_location.distance(waypoint.transform.location)
        if distance_to_road > 20.0 or distance_to_road < 2.0:
            return None
        
        road_right = waypoint.transform.get_right_vector()
        lane_width = waypoint.lane_width
        road_center = waypoint.transform.location
        
        ped_to_road = carla.Location(
            x=road_center.x - walker_location.x,
            y=road_center.y - walker_location.y,
            z=0
        )
        
        cross_product = ped_to_road.x * road_right.y - ped_to_road.y * road_right.x
        
        crossing_start = carla.Location(
            x=road_center.x - road_right.x * (lane_width + 1),
            y=road_center.y - road_right.y * (lane_width + 1),
            z=walker_location.z
        )
        crossing_end = carla.Location(
            x=road_center.x + road_right.x * (lane_width + 1),
            y=road_center.y + road_right.y * (lane_width + 1),
            z=walker_location.z
        )
        
        if walker_location.distance(crossing_start) > walker_location.distance(crossing_end):
            crossing_start, crossing_end = crossing_end, crossing_start
        
        return waypoint, crossing_start, crossing_end
    
    def check_traffic_safety(self, crossing_point: carla.Location, 
                             road_waypoint: carla.Waypoint) -> Tuple[bool, List[int]]:
        """Check if it's safe to cross by analyzing approaching vehicles."""
        vehicles = self.world.get_actors().filter('vehicle.*')
        blocking_vehicles = []
        
        for vehicle in vehicles:
            try:
                veh_location = vehicle.get_location()
                veh_velocity = vehicle.get_velocity()
                
                distance = veh_location.distance(crossing_point)
                if distance > self.safety_distance_threshold * 2:
                    continue
                
                speed = np.sqrt(veh_velocity.x**2 + veh_velocity.y**2 + veh_velocity.z**2)
                if speed < 0.5:
                    continue
                
                to_crossing = carla.Location(
                    x=crossing_point.x - veh_location.x,
                    y=crossing_point.y - veh_location.y,
                    z=0
                )
                to_crossing_dist = np.sqrt(to_crossing.x**2 + to_crossing.y**2)
                if to_crossing_dist < 0.1:
                    continue
                
                to_crossing_norm = carla.Location(
                    x=to_crossing.x / to_crossing_dist,
                    y=to_crossing.y / to_crossing_dist,
                    z=0
                )
                
                vel_norm = carla.Location(
                    x=veh_velocity.x / speed,
                    y=veh_velocity.y / speed,
                    z=0
                )
                
                dot_product = to_crossing_norm.x * vel_norm.x + to_crossing_norm.y * vel_norm.y
                
                if dot_product > 0.3:
                    time_to_arrival = distance / speed
                    
                    if time_to_arrival < self.safety_time_threshold:
                        blocking_vehicles.append(vehicle.id)
                    elif distance < self.safety_distance_threshold:
                        blocking_vehicles.append(vehicle.id)
            except RuntimeError:
                continue
        
        is_safe = len(blocking_vehicles) == 0
        return is_safe, blocking_vehicles
    
    def update_pedestrian(self, walker: carla.Walker, controller: carla.WalkerAIController, 
                          current_time: float) -> JaywalkingState:
        """Update jaywalking state for a pedestrian."""
        walker_id = walker.id
        
        if walker_id not in self.jaywalking_info:
            return JaywalkingState.NORMAL
        
        info = self.jaywalking_info[walker_id]
        
        if not info.is_jaywalker or not self.enabled:
            return JaywalkingState.NORMAL
        
        walker_location = walker.get_location()
        
        if info.state == JaywalkingState.NORMAL:
            if current_time - info.last_crossing_attempt < self.min_crossing_interval:
                return info.state
            
            if random.random() < 0.02:
                crossing_data = self.find_crossing_point(walker_location)
                if crossing_data:
                    waypoint, approach_point, destination = crossing_data
                    info.road_waypoint = waypoint
                    info.approach_point = approach_point
                    info.crossing_point = waypoint.transform.location
                    info.destination_after = destination
                    info.state = JaywalkingState.APPROACHING
                    info.last_crossing_attempt = current_time
                    
                    try:
                        controller.go_to_location(approach_point)
                        controller.set_max_speed(info.original_speed)
                    except RuntimeError:
                        pass
        
        elif info.state == JaywalkingState.APPROACHING:
            if info.approach_point:
                distance_to_approach = walker_location.distance(info.approach_point)
                
                if distance_to_approach < self.approach_distance:
                    info.state = JaywalkingState.WAITING
                    info.wait_start_time = current_time
                    info.last_safety_check = current_time
                    
                    try:
                        controller.set_max_speed(0)
                    except RuntimeError:
                        pass
        
        elif info.state == JaywalkingState.WAITING:
            if current_time - info.last_safety_check >= self.check_interval:
                info.last_safety_check = current_time
                
                if info.crossing_point:
                    is_safe, blocking = self.check_traffic_safety(
                        info.crossing_point, 
                        info.road_waypoint
                    )
                    info.is_safe_to_cross = is_safe
                    info.blocking_vehicles = blocking
                    
                    if is_safe:
                        info.state = JaywalkingState.CROSSING
                        try:
                            controller.go_to_location(info.destination_after)
                            crossing_speed = info.original_speed * self.crossing_speed_multiplier
                            controller.set_max_speed(crossing_speed)
                        except RuntimeError:
                            pass
            
            wait_duration = current_time - info.wait_start_time
            if wait_duration > self.max_wait_time:
                info.state = JaywalkingState.CROSSING
                try:
                    controller.go_to_location(info.destination_after)
                    crossing_speed = info.original_speed * self.crossing_speed_multiplier
                    controller.set_max_speed(crossing_speed)
                except RuntimeError:
                    pass
        
        elif info.state == JaywalkingState.CROSSING:
            if info.destination_after:
                distance_to_dest = walker_location.distance(info.destination_after)
                
                road_waypoint = self.map.get_waypoint(
                    walker_location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )
                on_road = False
                if road_waypoint:
                    on_road = walker_location.distance(road_waypoint.transform.location) < 2.0
                
                if distance_to_dest < 2.0 or (not on_road and distance_to_dest < 5.0):
                    info.state = JaywalkingState.COMPLETED
                    info.crossing_count += 1
        
        elif info.state == JaywalkingState.COMPLETED:
            info.state = JaywalkingState.NORMAL
            info.crossing_point = None
            info.destination_after = None
            info.approach_point = None
            info.road_waypoint = None
            info.is_safe_to_cross = False
            info.blocking_vehicles = []
            
            try:
                destination = self.world.get_random_location_from_navigation()
                if destination:
                    controller.go_to_location(destination)
                controller.set_max_speed(info.original_speed)
            except RuntimeError:
                pass
        
        return info.state
    
    def get_pedestrian_jaywalking_state(self, walker_id: int) -> dict:
        """Get jaywalking state info for a pedestrian."""
        if walker_id not in self.jaywalking_info:
            return {
                'is_jaywalker': False,
                'jaywalking_state': JaywalkingState.NORMAL.value,
                'is_safe_to_cross': True,
                'blocking_vehicles': [],
                'crossing_count': 0
            }
        
        info = self.jaywalking_info[walker_id]
        result = {
            'is_jaywalker': info.is_jaywalker,
            'jaywalking_state': info.state.value,
            'is_safe_to_cross': info.is_safe_to_cross,
            'blocking_vehicles': info.blocking_vehicles.copy(),
            'crossing_count': info.crossing_count,
            'wait_start_time': info.wait_start_time if info.state == JaywalkingState.WAITING else None
        }
        
        if info.crossing_point:
            result['crossing_point'] = location_to_dict(info.crossing_point)
        if info.destination_after:
            result['destination_after'] = location_to_dict(info.destination_after)
        
        return result


class StereoSensorManager:
    """Manages stereo camera and depth sensor setup."""
    
    def __init__(self, world: carla.World, config: dict):
        self.world = world
        self.config = config
        self.blueprint_library = world.get_blueprint_library()
        self.sensors = {}
        self.queues = {}
        self.intrinsics = {}
        self.transforms = {}
        self._sensor_config = config['sensors']

    def setup_on_vehicle(self, vehicle: carla.Vehicle) -> dict:
        """Setup stereo cameras on the vehicle."""
        baseline = self._sensor_config['stereo_baseline']
        height = self._sensor_config['camera_height']
        forward = self._sensor_config['camera_forward']
        left_transform = carla.Transform(
            carla.Location(x=forward, y=-baseline/2, z=height),
            carla.Rotation(pitch=0, yaw=0, roll=0)
        )
        right_transform = carla.Transform(
            carla.Location(x=forward, y=baseline/2, z=height),
            carla.Rotation(pitch=0, yaw=0, roll=0)
        )
        self.sensors['rgb_left'] = self._spawn_rgb_camera(vehicle, left_transform, 'rgb_left')
        self.sensors['rgb_right'] = self._spawn_rgb_camera(vehicle, right_transform, 'rgb_right')
        self.sensors['depth_left'] = self._spawn_depth_camera(vehicle, left_transform, 'depth_left')
        self.sensors['depth_right'] = self._spawn_depth_camera(vehicle, right_transform, 'depth_right')
        self.transforms['left'] = left_transform
        self.transforms['right'] = right_transform
        fov = self._sensor_config['fov']
        width = self._sensor_config['image_width']
        height_img = self._sensor_config['image_height']
        K = build_projection_matrix(width, height_img, fov)
        self.intrinsics = {
            'left': {'K': K, 'width': width, 'height': height_img, 'fov': fov},
            'right': {'K': K, 'width': width, 'height': height_img, 'fov': fov},
            'baseline': baseline
        }
        return self.intrinsics

    def _spawn_rgb_camera(self, vehicle: carla.Vehicle, transform: carla.Transform, 
                          name: str) -> carla.Sensor:
        """Spawn RGB camera sensor."""
        bp = self.blueprint_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self._sensor_config['image_width']))
        bp.set_attribute('image_size_y', str(self._sensor_config['image_height']))
        bp.set_attribute('fov', str(self._sensor_config['fov']))
        bp.set_attribute('motion_blur_intensity', str(self._sensor_config.get('motion_blur_intensity', 0.0)))
        bp.set_attribute('motion_blur_max_distortion', str(self._sensor_config.get('motion_blur_max_distortion', 0.0)))
        bp.set_attribute('motion_blur_min_object_screen_size', str(self._sensor_config.get('motion_blur_min_object_screen_size', 0.0)))
        camera = self.world.spawn_actor(bp, transform, attach_to=vehicle)
        self.queues[name] = Queue()
        camera.listen(self.queues[name].put)
        return camera

    def _spawn_depth_camera(self, vehicle: carla.Vehicle, transform: carla.Transform, 
                            name: str) -> carla.Sensor:
        """Spawn depth camera sensor."""
        bp = self.blueprint_library.find('sensor.camera.depth')
        bp.set_attribute('image_size_x', str(self._sensor_config['image_width']))
        bp.set_attribute('image_size_y', str(self._sensor_config['image_height']))
        bp.set_attribute('fov', str(self._sensor_config['fov']))
        camera = self.world.spawn_actor(bp, transform, attach_to=vehicle)
        self.queues[name] = Queue()
        camera.listen(self.queues[name].put)
        return camera

    def clear_queues(self):
        """Clear any stale data from sensor queues."""
        for name, queue in self.queues.items():
            dropped = 0
            while not queue.empty():
                try:
                    queue.get_nowait()
                    dropped += 1
                except Empty:
                    break
            if dropped > 0:
                print(f"  Cleared {dropped} stale frames from {name}")

    def get_synchronized_data(self, timeout: float = 2.0) -> Optional[SensorData]:
        """Get synchronized data from all sensors."""
        sensor_names = list(self.queues.keys())
        sensor_data_raw = {}
        target_frame = None
        try:
            for name in sensor_names:
                data = self.queues[name].get(timeout=timeout)
                sensor_data_raw[name] = data
                if target_frame is None:
                    target_frame = data.frame
                else:
                    target_frame = max(target_frame, data.frame)
            all_synced = all(sensor_data_raw[name].frame == target_frame for name in sensor_names)
            if not all_synced:
                for name in sensor_names:
                    while sensor_data_raw[name].frame < target_frame:
                        try:
                            sensor_data_raw[name] = self.queues[name].get(timeout=0.1)
                        except Empty:
                            break
            actual_frame = min(sensor_data_raw[name].frame for name in sensor_names)
            data = SensorData(frame=actual_frame, timestamp=sensor_data_raw[sensor_names[0]].timestamp)
            for name, raw_data in sensor_data_raw.items():
                if 'rgb' in name:
                    array = np.frombuffer(raw_data.raw_data, dtype=np.uint8)
                    array = array.reshape((raw_data.height, raw_data.width, 4))
                    array = array[:, :, :3].copy()
                    setattr(data, name, array)
                elif 'depth' in name:
                    array = np.frombuffer(raw_data.raw_data, dtype=np.uint8)
                    array = array.reshape((raw_data.height, raw_data.width, 4))
                    depth = (array[:, :, 2].astype(np.float32) +
                            array[:, :, 1].astype(np.float32) * 256 +
                            array[:, :, 0].astype(np.float32) * 256 * 256)
                    depth = depth / (256 * 256 * 256 - 1) * 1000
                    setattr(data, name, depth)
            return data
        except Empty:
            print(f"Timeout waiting for sensor data")
            return None

    def snapshot_camera_state(self) -> Tuple[dict, dict]:
        """Capture current camera transforms and matrices."""
        transforms = {}
        matrices = {}
        for side in ['left', 'right']:
            sensor_name = f'rgb_{side}'
            if sensor_name in self.sensors:
                transform = self.sensors[sensor_name].get_transform()
                transforms[side] = transform_to_dict(transform)
                matrices[side] = np.array(transform.get_inverse_matrix())
        return transforms, matrices

    def destroy(self):
        """Destroy all sensors."""
        for sensor in self.sensors.values():
            if sensor is not None and sensor.is_alive:
                try:
                    sensor.stop()
                    sensor.destroy()
                except:
                    pass
        self.sensors.clear()
        self.queues.clear()


class TrafficManager:
    """Manages vehicles and pedestrians in the simulation."""
    
    def __init__(self, client: carla.Client, world: carla.World, config: dict):
        self.client = client
        self.world = world
        self.config = config
        self.blueprint_library = world.get_blueprint_library()
        self.vehicles: List[carla.Vehicle] = []
        self.pedestrians: List[Tuple[carla.Walker, carla.WalkerAIController]] = []
        self.pedestrian_assigned_behaviors: Dict[int, PedestrianBehavior] = {}
        self.pedestrian_speeds: Dict[int, float] = {}
        self.pedestrian_is_runner: Dict[int, bool] = {}
        self._traffic_config = config['traffic']
        self._visibility_config = config.get('visibility', {})
        self._visible_count = 0
        self._total_count = 0
        self.tm = self.client.get_trafficmanager()
        self.tm.set_global_distance_to_leading_vehicle(2.5)
        self.tm.set_synchronous_mode(True)
        self.jaywalking_manager = JaywalkingManager(world, config)
        self.simulation_time = 0.0

    def spawn_ego_vehicle(self) -> carla.Vehicle:
        """Spawn the ego vehicle."""
        bp = self.blueprint_library.find('vehicle.tesla.model3')
        bp.set_attribute('role_name', 'ego')
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        ego_vehicle = self.world.spawn_actor(bp, spawn_point)
        ego_vehicle.set_autopilot(True, self.tm.get_port())
        self.tm.auto_lane_change(ego_vehicle, True)
        self.tm.distance_to_leading_vehicle(ego_vehicle, 5.0)
        self.tm.vehicle_percentage_speed_difference(ego_vehicle, -20)
        self.vehicles.append(ego_vehicle)
        return ego_vehicle

    def spawn_traffic_vehicles(self, num_vehicles: int = None):
        """Spawn traffic vehicles."""
        if num_vehicles is None:
            num_vehicles = self._traffic_config['num_vehicles']
        vehicle_bps = self.blueprint_library.filter('vehicle.*')
        vehicle_bps = [bp for bp in vehicle_bps if int(bp.get_attribute('number_of_wheels')) == 4]
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        batch = []
        for i, spawn_point in enumerate(spawn_points[:num_vehicles]):
            bp = random.choice(vehicle_bps)
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)
            batch.append(carla.command.SpawnActor(bp, spawn_point)
                        .then(carla.command.SetAutopilot(carla.command.FutureActor, True, self.tm.get_port())))
        results = self.client.apply_batch_sync(batch, True)
        for result in results:
            if not result.error:
                vehicle = self.world.get_actor(result.actor_id)
                if vehicle:
                    self.vehicles.append(vehicle)
        print(f"Spawned {len(self.vehicles) - 1} traffic vehicles")

    def spawn_pedestrians(self, num_pedestrians: int = None):
        """Spawn pedestrians with jaywalking behavior."""
        if num_pedestrians is None:
            num_pedestrians = self._traffic_config['num_pedestrians']
        
        runner_ratio = self._traffic_config.get('runner_ratio', 0.25)
        jaywalker_ratio = self.config.get('jaywalking', {}).get('jaywalker_ratio', 0.3)
        walk_speed_min = self._traffic_config.get('walk_speed_min', 0.8)
        walk_speed_max = self._traffic_config.get('walk_speed_max', 1.8)
        run_speed_min = self._traffic_config.get('run_speed_min', 2.5)
        run_speed_max = self._traffic_config.get('run_speed_max', 4.5)
        
        walker_bps = self.blueprint_library.filter('walker.pedestrian.*')
        walker_controller_bp = self.blueprint_library.find('controller.ai.walker')
        spawn_points = []
        for _ in range(num_pedestrians * 2):
            loc = self.world.get_random_location_from_navigation()
            if loc:
                spawn_points.append(carla.Transform(loc))
        
        batch = []
        walker_configs = []
        
        for spawn_point in spawn_points[:num_pedestrians]:
            bp = random.choice(walker_bps)
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')
            
            is_runner = random.random() < runner_ratio
            is_jaywalker = random.random() < jaywalker_ratio
            
            if is_runner:
                speed = random.uniform(run_speed_min, run_speed_max)
                behavior = PedestrianBehavior.RUNNING
            else:
                speed = random.uniform(walk_speed_min, walk_speed_max)
                behavior = PedestrianBehavior.WALKING
            
            walker_configs.append({
                'speed': speed,
                'is_runner': is_runner,
                'is_jaywalker': is_jaywalker,
                'behavior': behavior
            })
            batch.append(carla.command.SpawnActor(bp, spawn_point))
        
        walker_results = self.client.apply_batch_sync(batch, True)
        walkers = []
        for i, result in enumerate(walker_results):
            if not result.error:
                walkers.append({
                    'id': result.actor_id,
                    'config': walker_configs[i] if i < len(walker_configs) else {
                        'speed': 1.4, 'is_runner': False, 'is_jaywalker': False,
                        'behavior': PedestrianBehavior.WALKING
                    }
                })
        
        batch = []
        for walker in walkers:
            batch.append(carla.command.SpawnActor(
                walker_controller_bp, carla.Transform(), walker['id']
            ))
        
        controller_results = self.client.apply_batch_sync(batch, True)
        self.world.tick()
        
        cross_factor = self._traffic_config.get('pedestrian_cross_factor', 0.3)
        num_runners = 0
        num_walkers = 0
        num_jaywalkers = 0
        
        for i, result in enumerate(controller_results):
            if not result.error:
                walker = self.world.get_actor(walkers[i]['id'])
                controller = self.world.get_actor(result.actor_id)
                if walker and controller:
                    controller.start()
                    destination = self.world.get_random_location_from_navigation()
                    if destination:
                        controller.go_to_location(destination)
                    
                    config = walkers[i]['config']
                    speed = config['speed']
                    is_runner = config['is_runner']
                    is_jaywalker = config['is_jaywalker']
                    behavior = config['behavior']
                    
                    controller.set_max_speed(speed)
                    
                    self.pedestrian_speeds[walker.id] = speed
                    self.pedestrian_is_runner[walker.id] = is_runner
                    self.pedestrian_assigned_behaviors[walker.id] = behavior
                    
                    self.jaywalking_manager.register_pedestrian(
                        walker.id, is_jaywalker, speed
                    )
                    
                    if is_runner:
                        num_runners += 1
                    else:
                        num_walkers += 1
                    if is_jaywalker:
                        num_jaywalkers += 1
                    
                    if random.random() < cross_factor:
                        controller.set_max_speed(speed * 1.1)
                    
                    self.pedestrians.append((walker, controller))
        
        print(f"Spawned {len(self.pedestrians)} pedestrians ({num_walkers} walkers, {num_runners} runners, {num_jaywalkers} jaywalkers)")

    def update_jaywalking(self, delta_time: float):
        """Update jaywalking behavior for all pedestrians."""
        self.simulation_time += delta_time
        
        for walker, controller in self.pedestrians:
            try:
                if walker is None or not walker.is_alive:
                    continue
                self.jaywalking_manager.update_pedestrian(
                    walker, controller, self.simulation_time
                )
            except RuntimeError:
                continue

    def get_current_behavior(self, walker_id: int, velocity: dict) -> Tuple[str, dict]:
        """Determine current behavior with jaywalking state."""
        speed = np.sqrt(velocity['x']**2 + velocity['y']**2 + velocity['z']**2)
        
        jaywalking_state = self.jaywalking_manager.get_pedestrian_jaywalking_state(walker_id)
        jw_state = jaywalking_state['jaywalking_state']
        
        if jw_state == JaywalkingState.WAITING.value:
            return PedestrianBehavior.WAITING_TO_CROSS.value, jaywalking_state
        elif jw_state == JaywalkingState.CROSSING.value:
            return PedestrianBehavior.CROSSING.value, jaywalking_state
        elif jw_state == JaywalkingState.APPROACHING.value:
            if self.pedestrian_is_runner.get(walker_id, False):
                return PedestrianBehavior.RUNNING.value, jaywalking_state
            return PedestrianBehavior.WALKING.value, jaywalking_state
        
        if speed < 0.1:
            return PedestrianBehavior.IDLE.value, jaywalking_state
        
        try:
            walker = self.world.get_actor(walker_id)
            if walker:
                walker_location = walker.get_location()
                waypoint = self.world.get_map().get_waypoint(
                    walker_location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )
                if waypoint:
                    distance_to_road = walker_location.distance(waypoint.transform.location)
                    if distance_to_road < 2.0:
                        return PedestrianBehavior.CROSSING.value, jaywalking_state
        except RuntimeError:
            pass
        
        assigned = self.pedestrian_assigned_behaviors.get(walker_id, PedestrianBehavior.WALKING)
        return assigned.value, jaywalking_state

    def snapshot_pedestrian_state(self) -> Dict[int, Dict[str, Any]]:
        """Capture current pedestrian states."""
        snapshot = {}
        walkers_to_remove = []
        for i, (walker, controller) in enumerate(self.pedestrians):
            try:
                if walker is None or not walker.is_alive:
                    walkers_to_remove.append(i)
                    continue
                transform = walker.get_transform()
                velocity = walker.get_velocity()
                bones = snapshot_bone_data(walker)
                
                velocity_dict = {
                    'x': float(velocity.x),
                    'y': float(velocity.y),
                    'z': float(velocity.z)
                }
                
                current_behavior, jaywalking_info = self.get_current_behavior(walker.id, velocity_dict)
                
                snapshot[walker.id] = {
                    'id': walker.id,
                    'transform': transform_to_dict(transform),
                    'velocity': velocity_dict,
                    'speed': float(np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)),
                    'behavior': current_behavior,
                    'assigned_behavior': self.pedestrian_assigned_behaviors.get(walker.id, PedestrianBehavior.WALKING).value,
                    'is_runner': self.pedestrian_is_runner.get(walker.id, False),
                    'jaywalking': jaywalking_info,
                    'bones': bones
                }
            except RuntimeError:
                walkers_to_remove.append(i)
                continue
        for i in sorted(walkers_to_remove, reverse=True):
            walker, controller = self.pedestrians[i]
            if walker:
                self.pedestrian_assigned_behaviors.pop(walker.id, None)
                self.pedestrian_speeds.pop(walker.id, None)
                self.pedestrian_is_runner.pop(walker.id, None)
                self.jaywalking_manager.unregister_pedestrian(walker.id)
            del self.pedestrians[i]
        return snapshot

    def snapshot_vehicle_state(self) -> List[Dict[str, Any]]:
        """Capture current vehicle states."""
        snapshot = []
        for vehicle in self.vehicles:
            try:
                if not vehicle.is_alive:
                    continue
                snapshot.append({
                    'id': vehicle.id,
                    'type': vehicle.type_id,
                    'transform': transform_to_dict(vehicle.get_transform()),
                    'velocity': {
                        'x': float(vehicle.get_velocity().x),
                        'y': float(vehicle.get_velocity().y),
                        'z': float(vehicle.get_velocity().z)
                    },
                    'is_ego': vehicle.attributes.get('role_name') == 'ego'
                })
            except RuntimeError:
                continue
        return snapshot

    def process_pedestrian_visibility(
        self,
        pedestrian_snapshot: Dict[int, Dict[str, Any]],
        world_to_camera_matrices: dict,
        intrinsics: dict,
        depth_images: dict
    ) -> List[dict]:
        """Process visibility for all pedestrians."""
        pedestrian_data = []
        visible_count = 0
        depth_tolerance = self._visibility_config.get('depth_tolerance_factor', 0.05)
        min_visible_bones = self._visibility_config.get('min_visible_bones', 3)
        for ped_id, ped_snapshot in pedestrian_snapshot.items():
            visibility = {'visible': False, 'left': {}, 'right': {}}
            bones_data = {}
            bones = ped_snapshot.get('bones', {})
            if world_to_camera_matrices and intrinsics and depth_images and bones:
                for side in ['left', 'right']:
                    w2c = world_to_camera_matrices.get(side)
                    K = intrinsics.get(side, {}).get('K')
                    width = intrinsics.get(side, {}).get('width', 1920)
                    height = intrinsics.get(side, {}).get('height', 1080)
                    depth_img = depth_images.get(side)
                    if w2c is None or K is None or depth_img is None:
                        continue
                    for bone_name, bone_info in bones.items():
                        world_loc = bone_info['world']['location']
                        pixel, bone_depth = get_image_point(
                            world_loc['x'], world_loc['y'], world_loc['z'],
                            w2c, K
                        )
                        vis_result = check_bone_visibility_with_depth(
                            pixel=pixel,
                            bone_depth=bone_depth,
                            depth_image=depth_img,
                            width=width,
                            height=height,
                            depth_tolerance_factor=depth_tolerance
                        )
                        if bone_name not in bones_data:
                            bones_data[bone_name] = {
                                'world': bone_info['world'],
                                'relative': bone_info['relative'],
                                'cameras': {}
                            }
                        bones_data[bone_name]['cameras'][side] = {
                            'pixel': list(vis_result.pixel) if vis_result.pixel else None,
                            'bone_depth': float(vis_result.bone_depth) if vis_result.bone_depth else None,
                            'rendered_depth': float(vis_result.rendered_depth) if vis_result.rendered_depth else None,
                            'visibility_state': vis_result.visibility_state.value,
                            'visible': vis_result.visibility_state == VisibilityState.VISIBLE,
                            'occluded': vis_result.visibility_state == VisibilityState.OCCLUDED,
                            'occlusion_distance': float(vis_result.occlusion_distance) if vis_result.occlusion_distance else None
                        }
                for side in ['left', 'right']:
                    side_vis = check_pedestrian_visibility_detailed(
                        bones_data, side, min_visible_bones
                    )
                    visibility[side] = side_vis
                    if side_vis['visible']:
                        visibility['visible'] = True
            if visibility['visible']:
                visible_count += 1
            pedestrian_data.append({
                'id': ped_id,
                'transform': ped_snapshot['transform'],
                'velocity': ped_snapshot['velocity'],
                'speed': ped_snapshot['speed'],
                'behavior': ped_snapshot['behavior'],
                'assigned_behavior': ped_snapshot['assigned_behavior'],
                'is_runner': ped_snapshot['is_runner'],
                'jaywalking': ped_snapshot['jaywalking'],
                'visibility': visibility,
                'bones': bones_data if visibility['visible'] else {}
            })
        self._visible_count = visible_count
        self._total_count = len(pedestrian_data)
        return pedestrian_data

    def get_visibility_stats(self) -> Tuple[int, int]:
        """Get visibility statistics."""
        return self._visible_count, self._total_count

    def destroy(self):
        """Destroy all traffic actors."""
        for walker, controller in self.pedestrians:
            try:
                if controller and controller.is_alive:
                    controller.stop()
            except:
                pass
        all_actors = []
        for v in self.vehicles:
            if v and v.is_alive:
                all_actors.append(v)
        for w, c in self.pedestrians:
            if w and w.is_alive:
                all_actors.append(w)
            if c and c.is_alive:
                all_actors.append(c)
        if all_actors:
            try:
                batch = [carla.command.DestroyActor(actor) for actor in all_actors]
                self.client.apply_batch_sync(batch, False)
            except:
                pass
        self.vehicles.clear()
        self.pedestrians.clear()
        self.pedestrian_assigned_behaviors.clear()
        self.pedestrian_speeds.clear()
        self.pedestrian_is_runner.clear()
        print("Cleaned up all traffic actors")


class DemoVisualizer:
    """Visualizer for demo frame generation."""
    
    def __init__(self, skeleton_links: List[List[str]]):
        self.skeleton_links = skeleton_links

    def draw_skeleton_on_image(
        self,
        image: np.ndarray,
        pedestrian_data: List[dict],
        camera_side: str
    ) -> np.ndarray:
        """Draw skeleton overlays on image."""
        img = image.copy()
        height, width = img.shape[:2]
        for ped in pedestrian_data:
            if not ped.get('visibility', {}).get('visible', False):
                continue
            bones = ped.get('bones', {})
            if not bones:
                continue
            behavior = ped.get('behavior', 'unknown')
            ped_id = ped.get('id', 0)
            speed = ped.get('speed', 0)
            is_runner = ped.get('is_runner', False)
            jaywalking = ped.get('jaywalking', {})
            jw_state = jaywalking.get('jaywalking_state', 'normal')
            is_jaywalker = jaywalking.get('is_jaywalker', False)
            blocking_vehicles = jaywalking.get('blocking_vehicles', [])
            
            for link in self.skeleton_links:
                if len(link) < 2:
                    continue
                bone1_name, bone2_name = link[0], link[1]
                bone1_data = bones.get(bone1_name, {})
                bone2_data = bones.get(bone2_name, {})
                cam1 = bone1_data.get('cameras', {}).get(camera_side, {})
                cam2 = bone2_data.get('cameras', {}).get(camera_side, {})
                pixel1 = cam1.get('pixel')
                pixel2 = cam2.get('pixel')
                if pixel1 is None or pixel2 is None:
                    continue
                if not (0 <= pixel1[0] < width and 0 <= pixel1[1] < height):
                    continue
                if not (0 <= pixel2[0] < width and 0 <= pixel2[1] < height):
                    continue
                vis1 = cam1.get('visible', False)
                vis2 = cam2.get('visible', False)
                if vis1 and vis2:
                    color = get_bone_color(bone1_name, bone2_name)
                    thickness = 2
                elif vis1 or vis2:
                    color = (128, 128, 0)
                    thickness = 1
                else:
                    color = (0, 0, 128)
                    thickness = 1
                cv2.line(img, tuple(pixel1), tuple(pixel2), color, thickness)
            
            for bone_name, bone_data in bones.items():
                cam_data = bone_data.get('cameras', {}).get(camera_side, {})
                pixel = cam_data.get('pixel')
                if pixel is None:
                    continue
                if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
                    continue
                vis_state = cam_data.get('visibility_state', 'unknown')
                if vis_state == VisibilityState.VISIBLE.value:
                    color = (0, 255, 0)
                    radius = 4
                elif vis_state == VisibilityState.OCCLUDED.value:
                    color = (0, 0, 255)
                    radius = 3
                else:
                    color = (128, 128, 128)
                    radius = 2
                cv2.circle(img, tuple(pixel), radius, color, -1)
                cv2.circle(img, tuple(pixel), radius, (255, 255, 255), 1)
            
            head_bone = bones.get('crl_Head__C', {})
            head_cam = head_bone.get('cameras', {}).get(camera_side, {})
            if head_cam.get('pixel'):
                label_pos = head_cam['pixel']
                depth = head_cam.get('bone_depth', 0)
                vis_info = ped.get('visibility', {}).get(camera_side, {})
                visible_bones = vis_info.get('visible_bones', 0)
                total_bones = vis_info.get('total_bones', 0)
                occluded_bones = vis_info.get('occluded_bones', 0)
                
                behavior_color = BEHAVIOR_COLORS.get(behavior, (255, 255, 255))
                behavior_label = behavior.upper()
                
                tags = []
                if is_runner:
                    tags.append("R")
                if is_jaywalker:
                    tags.append("JW")
                if tags:
                    behavior_label += f" [{','.join(tags)}]"
                
                label = f"ID:{ped_id}"
                behavior_text = f"{behavior_label}"
                speed_label = f"Spd:{speed:.1f}m/s"
                depth_label = f"D:{depth:.1f}m"
                vis_label = f"V:{visible_bones}/{total_bones} O:{occluded_bones}"
                
                extra_labels = []
                if jw_state == JaywalkingState.WAITING.value:
                    extra_labels.append(f"Wait: {len(blocking_vehicles)} cars")
                elif jw_state == JaywalkingState.CROSSING.value:
                    extra_labels.append("CROSSING!")
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.45
                thickness_text = 1
                
                labels = [label, behavior_text, speed_label, depth_label, vis_label] + extra_labels
                max_w = 0
                total_h = 0
                heights = []
                for lbl in labels:
                    (w, h), _ = cv2.getTextSize(lbl, font, font_scale, thickness_text)
                    max_w = max(max_w, w)
                    heights.append(h)
                    total_h += h + 4
                
                label_x = max(0, min(label_pos[0] - 10, width - max_w - 5))
                label_y = max(total_h + 5, label_pos[1] - 60)
                
                cv2.rectangle(img,
                             (label_x - 2, label_y - total_h - 4),
                             (label_x + max_w + 2, label_y + 4),
                             (0, 0, 0), -1)
                
                y_offset = label_y - total_h + heights[0]
                cv2.putText(img, labels[0], (label_x, y_offset),
                           font, font_scale, (255, 255, 255), thickness_text)
                
                y_offset += heights[1] + 4
                cv2.putText(img, labels[1], (label_x, y_offset),
                           font, font_scale, behavior_color, thickness_text)
                
                y_offset += heights[2] + 4
                cv2.putText(img, labels[2], (label_x, y_offset),
                           font, font_scale, (255, 200, 100), thickness_text)
                
                y_offset += heights[3] + 4
                cv2.putText(img, labels[3], (label_x, y_offset),
                           font, font_scale, (0, 255, 0), thickness_text)
                
                y_offset += heights[4] + 4
                cv2.putText(img, labels[4], (label_x, y_offset),
                           font, font_scale, (255, 128, 0), thickness_text)
                
                for i, extra in enumerate(extra_labels):
                    y_offset += heights[5 + i] + 4
                    if "Wait" in extra:
                        color = (0, 0, 255)
                    elif "CROSSING" in extra:
                        color = (0, 255, 255)
                    else:
                        color = (255, 255, 255)
                    cv2.putText(img, extra, (label_x, y_offset),
                               font, font_scale, color, thickness_text)
        
        self._draw_legend(img)
        return img

    def _draw_legend(self, img: np.ndarray):
        """Draw legend on image."""
        legend_x = 10
        legend_y = img.shape[0] - 180
        cv2.rectangle(img, (legend_x, legend_y), (legend_x + 220, legend_y + 170), (0, 0, 0), -1)
        cv2.rectangle(img, (legend_x, legend_y), (legend_x + 220, legend_y + 170), (255, 255, 255), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        
        cv2.putText(img, "Visibility:", (legend_x + 5, legend_y + 15),
                   font, font_scale, (255, 255, 255), 1)
        cv2.circle(img, (legend_x + 15, legend_y + 30), 4, (0, 255, 0), -1)
        cv2.putText(img, "Visible", (legend_x + 25, legend_y + 34),
                   font, font_scale, (0, 255, 0), 1)
        cv2.circle(img, (legend_x + 100, legend_y + 30), 4, (0, 0, 255), -1)
        cv2.putText(img, "Occluded", (legend_x + 110, legend_y + 34),
                   font, font_scale, (0, 0, 255), 1)
        
        cv2.putText(img, "Behavior:", (legend_x + 5, legend_y + 52),
                   font, font_scale, (255, 255, 255), 1)
        cv2.putText(img, "WALKING", (legend_x + 15, legend_y + 67),
                   font, font_scale, BEHAVIOR_COLORS[PedestrianBehavior.WALKING.value], 1)
        cv2.putText(img, "RUNNING", (legend_x + 100, legend_y + 67),
                   font, font_scale, BEHAVIOR_COLORS[PedestrianBehavior.RUNNING.value], 1)
        cv2.putText(img, "CROSSING", (legend_x + 15, legend_y + 82),
                   font, font_scale, BEHAVIOR_COLORS[PedestrianBehavior.CROSSING.value], 1)
        cv2.putText(img, "WAITING", (legend_x + 100, legend_y + 82),
                   font, font_scale, BEHAVIOR_COLORS[PedestrianBehavior.WAITING_TO_CROSS.value], 1)
        cv2.putText(img, "IDLE", (legend_x + 15, legend_y + 97),
                   font, font_scale, BEHAVIOR_COLORS[PedestrianBehavior.IDLE.value], 1)
        
        cv2.putText(img, "Tags:", (legend_x + 5, legend_y + 115),
                   font, font_scale, (255, 255, 255), 1)
        cv2.putText(img, "[R] = Runner", (legend_x + 15, legend_y + 130),
                   font, font_scale, (255, 200, 100), 1)
        cv2.putText(img, "[JW] = Jaywalker", (legend_x + 15, legend_y + 145),
                   font, font_scale, (255, 200, 100), 1)
        cv2.putText(img, "Wait = Checking traffic", (legend_x + 15, legend_y + 160),
                   font, font_scale, (0, 0, 255), 1)

    def create_depth_visualization(self, depth: np.ndarray) -> np.ndarray:
        """Create colorized depth visualization."""
        depth_clipped = np.clip(depth, 0, 100)
        depth_normalized = (depth_clipped / 100.0 * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
        return depth_colored


class DataCollector:
    """Collects and saves simulation data."""
    
    def __init__(self, session_path: Path, config: dict, skeleton_links: List[List[str]]):
        self.session_path = session_path
        self.config = config
        self.frame_count = 0
        self.skeleton_links = skeleton_links
        self.visualizer = DemoVisualizer(skeleton_links)
        self.demo_interval = config['output'].get('demo_interval', 1000)
        self.metadata = {
            'session_start': datetime.now().isoformat(),
            'config': config,
            'frames': []
        }

    def save_camera_intrinsics(self, intrinsics: dict):
        """Save camera intrinsics to file."""
        intrinsics_path = self.session_path / "camera_intrinsics.json"
        intrinsics_save = {}
        for key, value in intrinsics.items():
            if isinstance(value, dict):
                intrinsics_save[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        intrinsics_save[key][k] = v.tolist()
                    else:
                        intrinsics_save[key][k] = v
            else:
                intrinsics_save[key] = value
        with open(intrinsics_path, 'w') as f:
            json.dump(intrinsics_save, f, indent=2, cls=NumpyEncoder)
        print(f"Saved camera intrinsics to {intrinsics_path}")

    def collect_frame(
        self,
        frame_id: int,
        sensor_data: SensorData,
        pedestrian_data: List[dict],
        vehicle_data: List[dict],
        weather_data: dict,
        camera_transforms: dict,
        world_to_camera_matrices: dict,
        intrinsics: dict
    ):
        """Collect and save a single frame of data."""
        if self.config['output']['save_images']:
            self._save_images(frame_id, sensor_data)
        processed_pedestrians = self._process_pedestrians(pedestrian_data)
        processed_vehicles = self._process_vehicles(vehicle_data)
        annotation = {
            'frame_id': int(frame_id),
            'timestamp': float(sensor_data.timestamp),
            'weather': weather_data,
            'ego_vehicle': self._get_ego_vehicle(vehicle_data),
            'vehicles': processed_vehicles,
            'pedestrians': processed_pedestrians,
            'camera_transforms': camera_transforms
        }
        annotation_path = self.session_path / "annotations" / f"frame_{frame_id:06d}.json"
        with open(annotation_path, 'w') as f:
            json.dump(annotation, f, indent=2, cls=NumpyEncoder)
        self.metadata['frames'].append({
            'frame_id': int(frame_id),
            'timestamp': float(sensor_data.timestamp)
        })
        if self.frame_count % self.demo_interval == 0:
            self._save_demo_frame(frame_id, sensor_data, pedestrian_data)
        self.frame_count += 1

    def _save_images(self, frame_id: int, sensor_data: SensorData):
        """Save image data to files."""
        if sensor_data.rgb_left is not None:
            cv2.imwrite(
                str(self.session_path / "rgb_left" / f"frame_{frame_id:06d}.png"),
                cv2.cvtColor(sensor_data.rgb_left, cv2.COLOR_RGB2BGR)
            )
        if sensor_data.rgb_right is not None:
            cv2.imwrite(
                str(self.session_path / "rgb_right" / f"frame_{frame_id:06d}.png"),
                cv2.cvtColor(sensor_data.rgb_right, cv2.COLOR_RGB2BGR)
            )
        if sensor_data.depth_left is not None:
            depth_mm = (sensor_data.depth_left * 1000).astype(np.uint16)
            cv2.imwrite(
                str(self.session_path / "depth_left" / f"frame_{frame_id:06d}.png"),
                depth_mm
            )
        if sensor_data.depth_right is not None:
            depth_mm = (sensor_data.depth_right * 1000).astype(np.uint16)
            cv2.imwrite(
                str(self.session_path / "depth_right" / f"frame_{frame_id:06d}.png"),
                depth_mm
            )

    def _save_demo_frame(self, frame_id: int, sensor_data: SensorData, pedestrian_data: List[dict]):
        """Save annotated demo frames."""
        demo_path = self.session_path / "demo"
        for side in ['left', 'right']:
            rgb = getattr(sensor_data, f'rgb_{side}')
            depth = getattr(sensor_data, f'depth_{side}')
            if rgb is None:
                continue
            img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            annotated = self.visualizer.draw_skeleton_on_image(img_bgr, pedestrian_data, side)
            
            cv2.putText(annotated, f"Frame: {frame_id}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(annotated, f"Camera: {side}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            visible_peds = [p for p in pedestrian_data if p.get('visibility', {}).get('visible', False)]
            walkers = sum(1 for p in visible_peds if p.get('behavior') == PedestrianBehavior.WALKING.value)
            runners = sum(1 for p in visible_peds if p.get('behavior') == PedestrianBehavior.RUNNING.value)
            crossers = sum(1 for p in visible_peds if p.get('behavior') == PedestrianBehavior.CROSSING.value)
            waiting = sum(1 for p in visible_peds if p.get('behavior') == PedestrianBehavior.WAITING_TO_CROSS.value)
            idle = sum(1 for p in visible_peds if p.get('behavior') == PedestrianBehavior.IDLE.value)
            
            cv2.putText(annotated, f"Visible: {len(visible_peds)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Walk:{walkers} Run:{runners} Cross:{crossers} Wait:{waiting} Idle:{idle}", 
                       (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 2)
            
            cv2.imwrite(str(demo_path / f"frame_{frame_id:06d}_{side}_skeleton.png"), annotated)
            if depth is not None:
                depth_vis = self.visualizer.create_depth_visualization(depth)
                cv2.putText(depth_vis, f"Depth - Frame: {frame_id}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imwrite(str(demo_path / f"frame_{frame_id:06d}_{side}_depth.png"), depth_vis)
        print(f"Saved demo frames for frame {frame_id}")

    def _process_pedestrians(self, pedestrian_data: List[dict]) -> List[dict]:
        """Process pedestrian data for saving."""
        processed = []
        for ped in pedestrian_data:
            visibility = ped.get('visibility', {'visible': False})
            ped_entry = {
                'id': int(ped['id']),
                'behavior': str(ped['behavior']),
                'assigned_behavior': str(ped.get('assigned_behavior', ped['behavior'])),
                'is_runner': bool(ped.get('is_runner', False)),
                'speed': float(ped.get('speed', 0)),
                'jaywalking': ped.get('jaywalking', {}),
                'world_transform': ped['transform'],
                'velocity': ped['velocity'],
                'visible_in_frame': bool(visibility.get('visible', False)),
                'visibility': {
                    'left': visibility.get('left', {}),
                    'right': visibility.get('right', {})
                },
                'skeleton': ped.get('bones', {})
            }
            processed.append(ped_entry)
        return processed

    def _process_vehicles(self, vehicle_data: List[dict]) -> List[dict]:
        """Process vehicle data for saving."""
        processed = []
        for veh in vehicle_data:
            if veh.get('is_ego', False):
                continue
            processed.append({
                'id': int(veh['id']),
                'type': str(veh['type']),
                'transform': veh['transform'],
                'velocity': veh['velocity']
            })
        return processed

    def _get_ego_vehicle(self, vehicle_data: List[dict]) -> Optional[dict]:
        """Get ego vehicle data."""
        for veh in vehicle_data:
            if veh.get('is_ego', False):
                return {
                    'id': int(veh['id']),
                    'transform': veh['transform'],
                    'velocity': veh['velocity']
                }
        return None

    def save_road_layout(self, world: carla.World):
        """Save road layout information."""
        carla_map = world.get_map()
        waypoints = carla_map.generate_waypoints(2.0)
        road_data = {
            'map_name': carla_map.name,
            'waypoints': []
        }
        for wp in waypoints[:1000]:
            road_data['waypoints'].append({
                'location': {
                    'x': float(wp.transform.location.x),
                    'y': float(wp.transform.location.y),
                    'z': float(wp.transform.location.z)
                },
                'road_id': int(wp.road_id),
                'lane_id': int(wp.lane_id),
                'lane_type': str(wp.lane_type),
                'lane_width': float(wp.lane_width)
            })
        road_path = self.session_path / "road_layout.json"
        with open(road_path, 'w') as f:
            json.dump(road_data, f, indent=2, cls=NumpyEncoder)
        print(f"Saved road layout to {road_path}")

    def finalize(self):
        """Finalize and save session metadata."""
        self.metadata['session_end'] = datetime.now().isoformat()
        self.metadata['total_frames'] = self.frame_count
        metadata_path = self.session_path / "session_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str, cls=NumpyEncoder)
        print(f"Session finalized. Total frames: {self.frame_count}")
        print(f"Data saved to: {self.session_path}")


class SimulationManager:
    """Main simulation manager with robust connection handling."""
    
    def __init__(self, config: dict, skeleton_links: List[List[str]]):
        self.config = config
        self.skeleton_links = skeleton_links
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.original_settings: Optional[carla.WorldSettings] = None
        self.sensor_manager: Optional[StereoSensorManager] = None
        self.traffic_manager: Optional[TrafficManager] = None
        self.data_collector: Optional[DataCollector] = None
        self.ego_vehicle: Optional[carla.Vehicle] = None
        self.current_weather_index = 0
        self.delta_time = 1.0 / config['simulation']['tick_rate']

    def connect(self, host: str = 'localhost', port: int = 2000):
        """
        Connect to CARLA server with robust retry logic.
        
        This method handles:
        - Initial connection retries while CARLA is starting
        - Map loading with verification
        - Proper timeout handling
        """
        conn_config = self.config.get('connection', {})
        timeout = conn_config.get('timeout', 30.0)
        max_retries = conn_config.get('max_retries', 12)
        retry_delay = conn_config.get('retry_delay', 5.0)
        post_map_load_wait = conn_config.get('post_map_load_wait', 5.0)
        
        print(f"Connecting to CARLA at {host}:{port}...")
        print(f"  Timeout: {timeout}s, Max retries: {max_retries}, Retry delay: {retry_delay}s")
        
        # Phase 1: Wait for port to be open
        print("\nPhase 1: Waiting for CARLA port to be available...")
        port_wait_start = time.time()
        port_max_wait = 60.0
        
        while time.time() - port_wait_start < port_max_wait:
            if check_port_open(host, port, timeout=2.0):
                print(f"  Port {port} is open!")
                break
            elapsed = time.time() - port_wait_start
            print(f"  Waiting for port {port}... ({elapsed:.0f}s / {port_max_wait:.0f}s)")
            time.sleep(3.0)
        else:
            raise RuntimeError(f"Port {port} did not become available within {port_max_wait}s. "
                             f"Make sure CARLA is running.")
        
        # Phase 2: Establish connection with retries
        print("\nPhase 2: Establishing CARLA connection...")
        last_error = None
        
        for attempt in range(1, max_retries + 1):
            try:
                print(f"  Attempt {attempt}/{max_retries}...")
                
                # Create client
                self.client = carla.Client(host, port)
                self.client.set_timeout(timeout)
                
                # Test connection by getting world
                print(f"    Testing connection...")
                current_world = self.client.get_world()
                current_map = current_world.get_map().name
                print(f"    Connected! Current map: {current_map}")
                
                # Phase 3: Load the desired map if needed
                map_name = self.config['simulation'].get('map')
                if map_name:
                    # Check if map change is needed
                    need_map_change = True
                    if map_name in current_map or current_map in map_name:
                        need_map_change = False
                    
                    # Also check short name variants
                    map_short = map_name.split('/')[-1] if '/' in map_name else map_name
                    current_short = current_map.split('/')[-1] if '/' in current_map else current_map
                    if map_short == current_short:
                        need_map_change = False
                    
                    if need_map_change:
                        print(f"\nPhase 3: Loading map '{map_name}'...")
                        print(f"    This may take a while for large maps...")
                        
                        # Increase timeout for map loading
                        self.client.set_timeout(120.0)
                        
                        try:
                            self.world = self.client.load_world(map_name)
                            print(f"    Map load command sent, waiting for world to be ready...")
                            
                            # Wait for the world to be fully loaded
                            time.sleep(post_map_load_wait)
                            
                            # Verify the world is ready
                            self.world = self.client.get_world()
                            loaded_map = self.world.get_map().name
                            print(f"    Map loaded: {loaded_map}")
                            
                        except RuntimeError as e:
                            if "time-out" in str(e).lower():
                                print(f"    Map load timeout, retrying...")
                                raise
                            else:
                                raise
                        finally:
                            # Reset timeout
                            self.client.set_timeout(timeout)
                    else:
                        print(f"\nPhase 3: Already on correct map: {current_map}")
                        self.world = current_world
                else:
                    self.world = current_world
                
                # Verify final connection
                test_map = self.world.get_map().name
                print(f"\nConnection established successfully!")
                print(f"  Final map: {test_map}")
                
                # Save original settings
                self.original_settings = self.world.get_settings()
                
                # Setup synchronous mode
                self._setup_synchronous_mode()
                
                return  # Success!
                
            except RuntimeError as e:
                last_error = e
                error_str = str(e).lower()
                
                if "time-out" in error_str:
                    print(f"    Timeout occurred: {e}")
                elif "connection" in error_str:
                    print(f"    Connection error: {e}")
                else:
                    print(f"    Error: {e}")
                
                if attempt < max_retries:
                    print(f"    Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"\n    All {max_retries} attempts failed!")
            
            except Exception as e:
                last_error = e
                print(f"    Unexpected error: {type(e).__name__}: {e}")
                if attempt < max_retries:
                    print(f"    Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        
        # All retries exhausted
        raise RuntimeError(
            f"Failed to connect to CARLA after {max_retries} attempts.\n"
            f"Last error: {last_error}\n"
            f"Troubleshooting:\n"
            f"  1. Make sure CARLA is running: ./CarlaUE4.sh -RenderOffScreen\n"
            f"  2. Wait at least 60 seconds after starting CARLA\n"
            f"  3. Check if port {port} is accessible\n"
            f"  4. Try restarting CARLA"
        )

    def _setup_synchronous_mode(self):
        """Setup synchronous simulation mode."""
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.delta_time
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)
        print(f"Synchronous mode enabled at {self.config['simulation']['tick_rate']} FPS")

    def setup(self):
        """Setup the simulation environment."""
        session_path = create_session_folder(self.config['output']['base_path'])
        self.traffic_manager = TrafficManager(self.client, self.world, self.config)
        self.sensor_manager = StereoSensorManager(self.world, self.config)
        self.data_collector = DataCollector(session_path, self.config, self.skeleton_links)
        
        print("\nSpawning ego vehicle...")
        self.ego_vehicle = self.traffic_manager.spawn_ego_vehicle()
        
        print("Setting up stereo cameras...")
        intrinsics = self.sensor_manager.setup_on_vehicle(self.ego_vehicle)
        self.data_collector.save_camera_intrinsics(intrinsics)
        
        print("Spawning traffic...")
        self.traffic_manager.spawn_traffic_vehicles()
        self.traffic_manager.spawn_pedestrians()
        
        self.data_collector.save_road_layout(self.world)
        self._apply_weather(0)
        
        print("Stabilizing simulation...")
        for i in range(50):
            self.world.tick()
            if (i + 1) % 10 == 0:
                print(f"  Stabilization tick {i + 1}/50")
        
        print("Clearing sensor queues...")
        self.sensor_manager.clear_queues()
        print("\nSetup complete!")

    def _apply_weather(self, index: int):
        """Apply weather preset."""
        presets = self.config.get('weather_presets', [])
        if not presets or index >= len(presets):
            return
        preset = presets[index]
        weather = carla.WeatherParameters(
            cloudiness=preset.get('cloudiness', 0),
            precipitation=preset.get('precipitation', 0),
            precipitation_deposits=preset.get('precipitation', 0) * 0.5,
            sun_altitude_angle=preset.get('sun_altitude_angle', 70),
            sun_azimuth_angle=preset.get('sun_azimuth_angle', 0),
            fog_density=preset.get('fog_density', 0),
            wetness=preset.get('precipitation', 0) * 0.5
        )
        self.world.set_weather(weather)
        self.current_weather_index = index
        print(f"Applied weather: {preset['name']}")

    def get_weather_data(self) -> dict:
        """Get current weather data."""
        weather = self.world.get_weather()
        presets = self.config.get('weather_presets', [])
        preset_name = presets[self.current_weather_index]['name'] if presets else 'unknown'
        return {
            'preset': str(preset_name),
            'cloudiness': float(weather.cloudiness),
            'precipitation': float(weather.precipitation),
            'sun_altitude_angle': float(weather.sun_altitude_angle),
            'sun_azimuth_angle': float(weather.sun_azimuth_angle),
            'fog_density': float(weather.fog_density),
            'wetness': float(weather.wetness)
        }

    def run(self, duration_seconds: float = None):
        """Run the data collection."""
        if duration_seconds is None:
            duration_seconds = self.config['simulation']['duration_seconds']
        tick_rate = self.config['simulation']['tick_rate']
        total_frames = int(duration_seconds * tick_rate)
        weather_presets = self.config.get('weather_presets', [])
        weather_change_interval = total_frames // (len(weather_presets) or 1)
        
        print(f"\n{'='*60}")
        print(f"Starting data collection")
        print(f"  Duration: {duration_seconds} seconds")
        print(f"  Target frames: {total_frames}")
        print(f"  Tick rate: {tick_rate} FPS")
        print(f"{'='*60}\n")
        
        collected_frames = 0
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        pending_snapshot = None
        pending_camera_transforms = None
        pending_camera_matrices = None
        pending_vehicle_snapshot = None
        
        start_time = time.time()
        
        try:
            for frame_idx in range(total_frames + 1):
                # Update jaywalking behavior
                self.traffic_manager.update_jaywalking(self.delta_time)
                
                # Capture state BEFORE tick
                camera_transforms, world_to_camera_matrices = self.sensor_manager.snapshot_camera_state()
                pedestrian_snapshot = self.traffic_manager.snapshot_pedestrian_state()
                vehicle_snapshot = self.traffic_manager.snapshot_vehicle_state()
                
                # Tick the simulation
                try:
                    self.world.tick()
                    consecutive_failures = 0
                except RuntimeError as e:
                    consecutive_failures += 1
                    print(f"Warning: Tick failed ({consecutive_failures}/{max_consecutive_failures}): {e}")
                    if consecutive_failures >= max_consecutive_failures:
                        print("Too many consecutive failures, stopping collection")
                        break
                    continue
                
                # Skip first frame (no sensor data yet)
                if frame_idx == 0:
                    pending_snapshot = pedestrian_snapshot
                    pending_camera_transforms = camera_transforms
                    pending_camera_matrices = world_to_camera_matrices
                    pending_vehicle_snapshot = vehicle_snapshot
                    continue
                
                # Get sensor data
                sensor_data = self.sensor_manager.get_synchronized_data(timeout=5.0)
                if sensor_data is None:
                    pending_snapshot = pedestrian_snapshot
                    pending_camera_transforms = camera_transforms
                    pending_camera_matrices = world_to_camera_matrices
                    pending_vehicle_snapshot = vehicle_snapshot
                    continue
                
                # Process visibility
                depth_images = {
                    'left': sensor_data.depth_left,
                    'right': sensor_data.depth_right
                }
                
                pedestrian_data = self.traffic_manager.process_pedestrian_visibility(
                    pending_snapshot,
                    pending_camera_matrices,
                    self.sensor_manager.intrinsics,
                    depth_images
                )
                
                # Change weather periodically
                if weather_presets and frame_idx > 0 and frame_idx % weather_change_interval == 0:
                    next_weather = (self.current_weather_index + 1) % len(weather_presets)
                    self._apply_weather(next_weather)
                
                # Collect frame data
                try:
                    self.data_collector.collect_frame(
                        frame_id=sensor_data.frame,
                        sensor_data=sensor_data,
                        pedestrian_data=pedestrian_data,
                        vehicle_data=pending_vehicle_snapshot,
                        weather_data=self.get_weather_data(),
                        camera_transforms=pending_camera_transforms,
                        world_to_camera_matrices=pending_camera_matrices,
                        intrinsics=self.sensor_manager.intrinsics
                    )
                    collected_frames += 1
                except Exception as e:
                    print(f"Warning: Failed to collect frame: {e}")
                
                # Update pending state
                pending_snapshot = pedestrian_snapshot
                pending_camera_transforms = camera_transforms
                pending_camera_matrices = world_to_camera_matrices
                pending_vehicle_snapshot = vehicle_snapshot
                
                # Progress reporting
                if collected_frames % 100 == 0 and collected_frames > 0:
                    visible, total = self.traffic_manager.get_visibility_stats()
                    elapsed = time.time() - start_time
                    fps = collected_frames / elapsed if elapsed > 0 else 0
                    print(f"Collected {collected_frames} frames | "
                          f"Visible pedestrians: {visible}/{total} | "
                          f"FPS: {fps:.1f}")
                
                if frame_idx % (tick_rate * 30) == 0 and frame_idx > 0:
                    progress = (frame_idx / total_frames) * 100
                    elapsed = time.time() - start_time
                    eta = (elapsed / frame_idx) * (total_frames - frame_idx) if frame_idx > 0 else 0
                    print(f"\nProgress: {progress:.1f}% ({frame_idx}/{total_frames} ticks)")
                    print(f"Elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s\n")
                    
        except KeyboardInterrupt:
            print("\n\nData collection interrupted by user")
        except Exception as e:
            print(f"\nData collection stopped due to error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            elapsed = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"Collection Summary")
            print(f"  Frames collected: {collected_frames}")
            print(f"  Time elapsed: {elapsed:.1f}s")
            print(f"  Average FPS: {collected_frames/elapsed:.1f}" if elapsed > 0 else "  Average FPS: N/A")
            print(f"{'='*60}\n")
            self.data_collector.finalize()

    def cleanup(self):
        """Cleanup simulation resources."""
        print("\nCleaning up...")
        
        # Set shorter timeout for cleanup
        try:
            if self.client:
                self.client.set_timeout(5.0)
        except:
            pass
        
        # Destroy sensors
        if self.sensor_manager:
            try:
                self.sensor_manager.destroy()
            except Exception as e:
                print(f"Warning: Could not destroy sensors: {e}")
        
        # Destroy traffic
        if self.traffic_manager:
            try:
                self.traffic_manager.destroy()
            except Exception as e:
                print(f"Warning: Could not destroy traffic: {e}")
        
        # Restore original settings
        if self.world and self.original_settings:
            try:
                self.world.apply_settings(self.original_settings)
            except Exception as e:
                print(f"Warning: Could not restore settings: {e}")
        
        print("Cleanup complete!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='CARLA Stereo Pedestrian Data Collection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python carla_data_collector.py --duration 300
  python carla_data_collector.py --host localhost --port 2000 --duration 600
  python carla_data_collector.py --config custom_config.yaml
        """
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--skeleton', type=str, default='skeleton.txt',
                        help='Path to skeleton links file (default: skeleton.txt)')
    parser.add_argument('--host', type=str, default='localhost',
                        help='CARLA server host (default: localhost)')
    parser.add_argument('--port', type=int, default=2000,
                        help='CARLA server port (default: 2000)')
    parser.add_argument('--duration', type=int, default=None,
                        help='Override simulation duration in seconds')
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("CARLA Stereo Pedestrian Data Collection System")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Config file: {args.config}")
    print(f"  Skeleton file: {args.skeleton}")
    print(f"  CARLA host: {args.host}")
    print(f"  CARLA port: {args.port}")
    
    # Load configuration
    print(f"\nLoading configuration from {args.config}")
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found: {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"ERROR: Invalid YAML in configuration file: {e}")
        sys.exit(1)
    
    # Override duration if specified
    if args.duration:
        config['simulation']['duration_seconds'] = args.duration
        print(f"  Duration override: {args.duration} seconds")
    
    # Override connection settings from command line
    if 'connection' not in config:
        config['connection'] = {}
    config['connection']['host'] = args.host
    config['connection']['port'] = args.port
    
    # Load skeleton links
    print(f"\nLoading skeleton links from {args.skeleton}")
    skeleton_links = load_skeleton_links(args.skeleton)
    print(f"  Loaded {len(skeleton_links)} skeleton links")
    
    # Create and run simulation
    sim = SimulationManager(config, skeleton_links)
    
    try:
        sim.connect(host=args.host, port=args.port)
        sim.setup()
        sim.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        sim.cleanup()
    
    print("\nDone!")


if __name__ == '__main__':
    main()
