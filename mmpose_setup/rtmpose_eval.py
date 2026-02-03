#!/usr/bin/env python3
"""
RTMPose Evaluation on CARLA Stereo Pedestrian Dataset - Streaming Version

Features:
- Streaming evaluation with disk-based result storage
- Left-right flip correction for pose predictions
- Memory-efficient incremental statistics
- Comprehensive per-joint confidence tracking

Author: Assistant
License: MIT
"""

import json
import gzip
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import warnings
from scipy.optimize import linear_sum_assignment
import cv2
import time
import gc

from dataloader import (
    CARLAStereoPedestrianDataset,
    FrameData,
    Pedestrian,
    COCO_KEYPOINTS,
    COCO_SKELETON
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Constants
# =============================================================================

# COCO keypoint pairs for left-right flipping
# Format: (left_index, right_index)
FLIP_PAIRS = [
    (1, 2),   # left_eye <-> right_eye
    (3, 4),   # left_ear <-> right_ear
    (5, 6),   # left_shoulder <-> right_shoulder
    (7, 8),   # left_elbow <-> right_elbow
    (9, 10),  # left_wrist <-> right_wrist
    (11, 12), # left_hip <-> right_hip
    (13, 14), # left_knee <-> right_knee
    (15, 16), # left_ankle <-> right_ankle
]


# =============================================================================
# Enums and Configuration
# =============================================================================

class FrameStatus(Enum):
    SUCCESS = "success"
    DETECTION_FAILED = "detection_failed"
    POSE_FAILED = "pose_failed"
    NO_IMAGE = "no_image"
    SKIPPED = "skipped"


@dataclass
class DetectorConfig:
    config_file: str = ''
    checkpoint_file: str = ''
    score_threshold: float = 0.5
    nms_threshold: float = 0.65
    person_class_id: int = 0


@dataclass
class PoseModelConfig:
    config_file: str = ''
    checkpoint_file: str = ''
    model_type: str = 'rtmpose'


@dataclass
class EvalConfig:
    pose: PoseModelConfig = field(default_factory=PoseModelConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    device: str = 'cuda:0'
    use_gt_bboxes: bool = False
    bbox_padding: float = 0.1
    min_bbox_area: float = 100.0
    max_deviation_threshold: float = 100.0
    use_visible_only_for_matching: bool = True
    min_visible_keypoints_for_match: int = 3
    enable_flip_correction: bool = True  # NEW: Enable left-right flip correction
    distance_ranges: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0, 10), (10, 20), (20, 30), (30, 50), (50, float('inf'))
    ])
    occlusion_ranges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (0, 0), (1, 3), (4, 6), (7, 10), (11, 17)
    ])
    confidence_ranges: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)
    ])
    cameras: List[str] = field(default_factory=lambda: ['left', 'right'])
    min_gt_visible_keypoints: int = 1
    clear_cache_every_n_frames: int = 50


# =============================================================================
# JSON Helpers
# =============================================================================

def convert_to_serializable(obj: Any) -> Any:
    if obj is None:
        return None
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, (str, int, float)):
        return obj
    else:
        try:
            return str(obj)
        except:
            return None


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class MatchedSkeleton:
    """Matched skeleton with full keypoint data."""
    gt_pedestrian_id: int
    prediction_index: int
    camera: str
    distance_to_camera: Optional[float]
    detection_score: Optional[float]
    
    # Bounding boxes
    pred_bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    gt_bbox: Optional[Tuple[float, float, float, float]]
    bbox_iou: Optional[float]
    
    # Full keypoint arrays
    pred_keypoints: np.ndarray  # [17, 3] with (x, y, confidence)
    
    # Flip correction
    was_flipped: bool  # Whether prediction was flipped to match GT
    
    # Aggregate metrics
    mean_deviation: float
    mean_deviation_all: float
    mean_confidence: float
    num_visible_keypoints_gt: int
    num_occluded_keypoints_gt: int
    
    def to_dict(self) -> dict:
        return {
            'gt_pedestrian_id': int(self.gt_pedestrian_id),
            'prediction_index': int(self.prediction_index),
            'camera': self.camera,
            'distance_to_camera': float(self.distance_to_camera) if self.distance_to_camera else None,
            'detection_score': float(self.detection_score) if self.detection_score else None,
            'pred_bbox': [float(x) for x in self.pred_bbox],
            'gt_bbox': [float(x) for x in self.gt_bbox] if self.gt_bbox else None,
            'bbox_iou': float(self.bbox_iou) if self.bbox_iou else None,
            'pred_keypoints': self.pred_keypoints.tolist(),
            'was_flipped': self.was_flipped,
            'mean_deviation': float(self.mean_deviation),
            'mean_deviation_all': float(self.mean_deviation_all),
            'mean_confidence': float(self.mean_confidence),
            'num_visible_keypoints_gt': int(self.num_visible_keypoints_gt),
            'num_occluded_keypoints_gt': int(self.num_occluded_keypoints_gt),
        }


@dataclass
class HallucinatedSkeleton:
    """Predicted skeleton with no GT match."""
    prediction_index: int
    camera: str
    detection_score: Optional[float]
    pred_bbox: Tuple[float, float, float, float]
    pred_keypoints: np.ndarray  # [17, 3] with (x, y, confidence)
    mean_confidence: float
    max_confidence: float
    min_confidence: float
    
    def to_dict(self) -> dict:
        return {
            'prediction_index': int(self.prediction_index),
            'camera': self.camera,
            'detection_score': float(self.detection_score) if self.detection_score else None,
            'pred_bbox': [float(x) for x in self.pred_bbox],
            'pred_keypoints': self.pred_keypoints.tolist(),
            'mean_confidence': float(self.mean_confidence),
            'max_confidence': float(self.max_confidence),
            'min_confidence': float(self.min_confidence),
        }


@dataclass
class MissingSkeleton:
    """GT skeleton with no prediction match."""
    gt_pedestrian_id: int
    camera: str
    distance_to_camera: Optional[float]
    num_visible_keypoints: int
    num_occluded_keypoints: int
    gt_bbox: Optional[Tuple[float, float, float, float]]
    
    def to_dict(self) -> dict:
        return {
            'gt_pedestrian_id': int(self.gt_pedestrian_id),
            'camera': self.camera,
            'distance_to_camera': float(self.distance_to_camera) if self.distance_to_camera else None,
            'num_visible_keypoints': int(self.num_visible_keypoints),
            'num_occluded_keypoints': int(self.num_occluded_keypoints),
            'gt_bbox': [float(x) for x in self.gt_bbox] if self.gt_bbox else None,
        }


@dataclass
class FrameResult:
    """Complete evaluation result for a single frame + camera."""
    frame_id: int
    camera: str
    timestamp: float
    status: FrameStatus = FrameStatus.SUCCESS
    error_message: Optional[str] = None
    
    num_gt_pedestrians: int = 0
    num_detections: int = 0
    num_predictions: int = 0
    num_matched: int = 0
    num_hallucinated: int = 0
    num_missing: int = 0
    num_flipped: int = 0  # NEW: How many predictions were flipped
    
    matched_skeletons: List[MatchedSkeleton] = field(default_factory=list)
    hallucinated_skeletons: List[HallucinatedSkeleton] = field(default_factory=list)
    missing_skeletons: List[MissingSkeleton] = field(default_factory=list)
    
    detection_time_ms: Optional[float] = None
    pose_time_ms: Optional[float] = None
    
    @property
    def is_success(self) -> bool:
        return self.status == FrameStatus.SUCCESS
    
    def to_dict(self) -> dict:
        return {
            'frame_id': self.frame_id,
            'camera': self.camera,
            'timestamp': self.timestamp,
            'status': self.status.value,
            'error_message': self.error_message,
            'num_gt_pedestrians': self.num_gt_pedestrians,
            'num_detections': self.num_detections,
            'num_predictions': self.num_predictions,
            'num_matched': self.num_matched,
            'num_hallucinated': self.num_hallucinated,
            'num_missing': self.num_missing,
            'num_flipped': self.num_flipped,
            'matched_skeletons': [m.to_dict() for m in self.matched_skeletons],
            'hallucinated_skeletons': [h.to_dict() for h in self.hallucinated_skeletons],
            'missing_skeletons': [m.to_dict() for m in self.missing_skeletons],
            'detection_time_ms': self.detection_time_ms,
            'pose_time_ms': self.pose_time_ms,
        }


# =============================================================================
# Flip Utilities
# =============================================================================

def flip_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Flip left-right keypoints (swap left/right body parts).
    
    Args:
        keypoints: [17, 3] array with (x, y, confidence)
    
    Returns:
        Flipped keypoints [17, 3]
    """
    flipped = keypoints.copy()
    
    for left_idx, right_idx in FLIP_PAIRS:
        flipped[left_idx] = keypoints[right_idx].copy()
        flipped[right_idx] = keypoints[left_idx].copy()
    
    return flipped


def compute_keypoint_deviation(
    gt_keypoints: np.ndarray,
    pred_keypoints: np.ndarray,
    use_visible_only: bool = True,
    min_keypoints: int = 3,
) -> Tuple[float, int]:
    """
    Compute mean deviation between GT and predicted keypoints.
    
    Returns:
        Tuple of (mean_deviation, num_valid_keypoints)
        Returns (inf, 0) if not enough valid keypoints
    """
    if use_visible_only:
        mask = gt_keypoints[:, 2] == 2
    else:
        mask = gt_keypoints[:, 2] > 0
    
    # Also check prediction validity
    pred_valid = (
        np.isfinite(pred_keypoints[:, 0]) &
        np.isfinite(pred_keypoints[:, 1]) &
        (np.abs(pred_keypoints[:, 0]) < 10000) &
        (np.abs(pred_keypoints[:, 1]) < 10000)
    )
    
    combined_mask = mask & pred_valid
    num_valid = combined_mask.sum()
    
    if num_valid < min_keypoints:
        return float('inf'), 0
    
    gt_pos = gt_keypoints[combined_mask, :2]
    pred_pos = pred_keypoints[combined_mask, :2]
    
    distances = np.linalg.norm(gt_pos - pred_pos, axis=1)
    valid_distances = distances[np.isfinite(distances)]
    
    if len(valid_distances) == 0:
        return float('inf'), 0
    
    return float(valid_distances.mean()), int(num_valid)


# =============================================================================
# RTMPose Evaluator
# =============================================================================

class RTMPoseEvaluator:
    """
    Evaluates RTMPose against CARLA ground truth with flip correction.
    """
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.detector = None
        self.pose_model = None
        self.detection_failure_count = 0
        self.pose_failure_count = 0
        
        self._validate_config()
        self._init_models()
    
    def _validate_config(self):
        if not self.config.pose.config_file or not self.config.pose.checkpoint_file:
            raise ValueError("Pose model config and checkpoint required")
        
        if not Path(self.config.pose.config_file).exists():
            raise FileNotFoundError(f"Pose config not found: {self.config.pose.config_file}")
        
        if not Path(self.config.pose.checkpoint_file).exists():
            raise FileNotFoundError(f"Pose checkpoint not found: {self.config.pose.checkpoint_file}")
        
        if not self.config.use_gt_bboxes:
            if not self.config.detector.config_file or not self.config.detector.checkpoint_file:
                raise ValueError("Detector config and checkpoint required")
            
            if not Path(self.config.detector.config_file).exists():
                raise FileNotFoundError(f"Detector config not found")
            
            if not Path(self.config.detector.checkpoint_file).exists():
                raise FileNotFoundError(f"Detector checkpoint not found")
    
    def _init_models(self):
        if not self.config.use_gt_bboxes:
            self._init_detector()
        self._init_pose_model()
    
    def _init_detector(self):
        print(f"Loading detector...")
        from mmdet.utils import register_all_modules as register_mmdet
        from mmdet.apis import init_detector
        
        register_mmdet(init_default_scope=True)
        self.detector = init_detector(
            self.config.detector.config_file,
            self.config.detector.checkpoint_file,
            device=self.config.device
        )
        print(f"✓ Detector loaded")
    
    def _init_pose_model(self):
        print(f"Loading pose model...")
        from mmpose.utils import register_all_modules as register_mmpose
        from mmpose.apis import init_model
        
        register_mmpose(init_default_scope=True)
        self.pose_model = init_model(
            self.config.pose.config_file,
            self.config.pose.checkpoint_file,
            device=self.config.device
        )
        print(f"✓ Pose model loaded")
    
    def detect_persons(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from mmengine.registry import DefaultScope
        from mmdet.apis import inference_detector
        
        with DefaultScope.overwrite_default_scope('mmdet'):
            result = inference_detector(self.detector, image)
        
        bboxes, scores = [], []
        
        if hasattr(result, 'pred_instances'):
            pred = result.pred_instances
            labels = pred.labels.cpu().numpy()
            det_scores = pred.scores.cpu().numpy()
            boxes = pred.bboxes.cpu().numpy()
            
            for label, score, box in zip(labels, det_scores, boxes):
                if label == self.config.detector.person_class_id and score >= self.config.detector.score_threshold:
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    if area >= self.config.min_bbox_area:
                        bboxes.append(box)
                        scores.append(score)
        
        return (np.array(bboxes, dtype=np.float32).reshape(-1, 4),
                np.array(scores, dtype=np.float32))
    
    def estimate_poses(self, image: np.ndarray, bboxes: np.ndarray) -> List[np.ndarray]:
        if len(bboxes) == 0:
            return []
        
        from mmengine.registry import DefaultScope
        from mmpose.apis import inference_topdown
        
        bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        
        with DefaultScope.overwrite_default_scope('mmpose'):
            results = inference_topdown(self.pose_model, image, bboxes, bbox_format='xyxy')
        
        poses = []
        for result in results:
            if not hasattr(result, 'pred_instances'):
                continue
            
            pred = result.pred_instances
            if not hasattr(pred, 'keypoints'):
                continue
            
            kps = pred.keypoints
            kp_scores = getattr(pred, 'keypoint_scores', None)
            
            if hasattr(kps, 'cpu'):
                kps = kps.cpu().numpy()
            if kp_scores is not None and hasattr(kp_scores, 'cpu'):
                kp_scores = kp_scores.cpu().numpy()
            
            kps = np.atleast_2d(np.squeeze(kps))
            
            if kps.ndim == 2 and kps.shape[0] == 17:
                if kp_scores is not None:
                    kp_scores = np.squeeze(kp_scores)
                    if kp_scores.ndim == 0:
                        kp_scores = np.full(17, float(kp_scores))
                    kp_scores = kp_scores[:17].reshape(-1, 1)
                else:
                    kp_scores = np.ones((17, 1), dtype=np.float32)
                
                pose = np.concatenate([kps[:, :2], kp_scores], axis=1)
                poses.append(pose.astype(np.float32))
        
        return poses
    
    def _expand_bboxes(self, bboxes: np.ndarray, padding: float) -> np.ndarray:
        if len(bboxes) == 0:
            return bboxes
        
        bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        widths = bboxes[:, 2] - bboxes[:, 0]
        heights = bboxes[:, 3] - bboxes[:, 1]
        
        expanded = bboxes.copy()
        expanded[:, 0] -= widths * padding / 2
        expanded[:, 1] -= heights * padding / 2
        expanded[:, 2] += widths * padding / 2
        expanded[:, 3] += heights * padding / 2
        
        return expanded
    
    @staticmethod
    def compute_bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0
    
    def _compute_deviation_matrix_with_flip(
        self,
        gt_keypoints_list: List[np.ndarray],
        pred_keypoints_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute deviation matrix considering both normal and flipped predictions.
        
        Returns:
            cost_matrix: [num_gt, num_pred] minimum deviation for each pair
            flip_matrix: [num_gt, num_pred] bool, True if flipped version was better
        """
        num_gt = len(gt_keypoints_list)
        num_pred = len(pred_keypoints_list)
        
        if num_gt == 0 or num_pred == 0:
            return np.zeros((num_gt, num_pred)), np.zeros((num_gt, num_pred), dtype=bool)
        
        cost_matrix = np.full((num_gt, num_pred), np.inf)
        flip_matrix = np.zeros((num_gt, num_pred), dtype=bool)
        
        use_visible = self.config.use_visible_only_for_matching
        min_kps = self.config.min_visible_keypoints_for_match
        
        for i, gt_kps in enumerate(gt_keypoints_list):
            for j, pred_kps in enumerate(pred_keypoints_list):
                # Normal deviation
                dev_normal, n_normal = compute_keypoint_deviation(
                    gt_kps, pred_kps, use_visible, min_kps
                )
                
                # Flipped deviation
                if self.config.enable_flip_correction:
                    pred_flipped = flip_keypoints(pred_kps)
                    dev_flipped, n_flipped = compute_keypoint_deviation(
                        gt_kps, pred_flipped, use_visible, min_kps
                    )
                else:
                    dev_flipped = float('inf')
                
                # Use better version
                if dev_flipped < dev_normal:
                    cost_matrix[i, j] = dev_flipped
                    flip_matrix[i, j] = True
                else:
                    cost_matrix[i, j] = dev_normal
                    flip_matrix[i, j] = False
        
        return cost_matrix, flip_matrix
    
    def _match_skeletons_with_flip(
        self,
        gt_keypoints_list: List[np.ndarray],
        pred_keypoints_list: List[np.ndarray],
    ) -> Tuple[List[Tuple[int, int, bool]], List[int], List[int]]:
        """
        Match GT to predictions using Hungarian algorithm with flip correction.
        
        Returns:
            matched_pairs: List of (gt_idx, pred_idx, was_flipped)
            unmatched_gt: List of unmatched GT indices
            unmatched_pred: List of unmatched prediction indices
        """
        num_gt = len(gt_keypoints_list)
        num_pred = len(pred_keypoints_list)
        
        if num_gt == 0:
            return [], [], list(range(num_pred))
        if num_pred == 0:
            return [], list(range(num_gt)), []
        
        cost_matrix, flip_matrix = self._compute_deviation_matrix_with_flip(
            gt_keypoints_list, pred_keypoints_list
        )
        
        finite_mask = np.isfinite(cost_matrix)
        if not finite_mask.any():
            return [], list(range(num_gt)), list(range(num_pred))
        
        max_valid = cost_matrix[finite_mask].max()
        large_cost = max(max_valid * 10, self.config.max_deviation_threshold * 10, 1e6)
        cost_matrix_safe = np.where(finite_mask, cost_matrix, large_cost)
        
        try:
            row_indices, col_indices = linear_sum_assignment(cost_matrix_safe)
        except ValueError:
            return [], list(range(num_gt)), list(range(num_pred))
        
        matched_pairs = []
        unmatched_gt = set(range(num_gt))
        unmatched_pred = set(range(num_pred))
        
        for gt_idx, pred_idx in zip(row_indices, col_indices):
            cost = cost_matrix[gt_idx, pred_idx]
            if np.isfinite(cost) and cost <= self.config.max_deviation_threshold:
                was_flipped = flip_matrix[gt_idx, pred_idx]
                matched_pairs.append((gt_idx, pred_idx, was_flipped))
                unmatched_gt.discard(gt_idx)
                unmatched_pred.discard(pred_idx)
        
        return matched_pairs, list(unmatched_gt), list(unmatched_pred)
    
    def _get_pedestrian_distance(self, ped: Pedestrian, camera: str) -> Optional[float]:
        nose = ped.keypoints.get('nose')
        if nose:
            depth = nose.depth_left if camera == 'left' else nose.depth_right
            if depth and depth > 0:
                return depth
        return ped.distance_to_ego
    
    def _compute_per_keypoint_deviations(
        self,
        gt_keypoints: np.ndarray,
        pred_keypoints: np.ndarray,
    ) -> Tuple[List[Optional[float]], float, float, int, int]:
        """
        Compute per-keypoint deviations.
        
        Returns:
            deviations: List of deviation per keypoint (None if GT not labeled)
            mean_visible: Mean deviation for visible keypoints
            mean_all: Mean deviation for all labeled keypoints
            num_visible: Count of visible GT keypoints
            num_occluded: Count of occluded GT keypoints
        """
        deviations = []
        devs_visible = []
        devs_all = []
        num_visible = 0
        num_occluded = 0
        
        for kp_idx in range(17):
            gt_vis = int(gt_keypoints[kp_idx, 2])
            
            if gt_vis == 2:
                num_visible += 1
            elif gt_vis == 1:
                num_occluded += 1
            
            if gt_vis > 0:
                gt_pos = gt_keypoints[kp_idx, :2]
                pred_pos = pred_keypoints[kp_idx, :2]
                dev = float(np.linalg.norm(gt_pos - pred_pos))
                deviations.append(dev)
                devs_all.append(dev)
                if gt_vis == 2:
                    devs_visible.append(dev)
            else:
                deviations.append(None)
        
        mean_visible = float(np.mean(devs_visible)) if devs_visible else 0.0
        mean_all = float(np.mean(devs_all)) if devs_all else 0.0
        
        return deviations, mean_visible, mean_all, num_visible, num_occluded
    
    def _create_matched_skeleton(
        self,
        gt_ped: Pedestrian,
        gt_kps: np.ndarray,
        pred_kps: np.ndarray,
        pred_idx: int,
        camera: str,
        det_score: Optional[float],
        pred_bbox: np.ndarray,
        was_flipped: bool,
    ) -> MatchedSkeleton:
        """Create a matched skeleton result."""
        # Apply flip if needed
        if was_flipped:
            pred_kps = flip_keypoints(pred_kps)
        
        # Compute deviations
        _, mean_dev, mean_dev_all, num_vis, num_occ = self._compute_per_keypoint_deviations(
            gt_kps, pred_kps
        )
        
        # Get GT bbox
        gt_bbox = gt_ped.get_bounding_box(camera)
        gt_bbox_tuple = tuple(float(x) for x in gt_bbox) if gt_bbox is not None else None
        
        # Compute IoU
        bbox_iou = None
        if gt_bbox is not None:
            bbox_iou = self.compute_bbox_iou(gt_bbox, pred_bbox)
        
        return MatchedSkeleton(
            gt_pedestrian_id=gt_ped.id,
            prediction_index=pred_idx,
            camera=camera,
            distance_to_camera=self._get_pedestrian_distance(gt_ped, camera),
            detection_score=det_score,
            pred_bbox=tuple(float(x) for x in pred_bbox),
            gt_bbox=gt_bbox_tuple,
            bbox_iou=bbox_iou,
            pred_keypoints=pred_kps.copy(),
            was_flipped=was_flipped,
            mean_deviation=mean_dev,
            mean_deviation_all=mean_dev_all,
            mean_confidence=float(np.mean(pred_kps[:, 2])),
            num_visible_keypoints_gt=num_vis,
            num_occluded_keypoints_gt=num_occ,
        )
    
    def _create_hallucinated_skeleton(
        self,
        pred_kps: np.ndarray,
        pred_idx: int,
        camera: str,
        det_score: Optional[float],
        pred_bbox: np.ndarray,
    ) -> HallucinatedSkeleton:
        """Create a hallucinated skeleton result."""
        confidences = pred_kps[:, 2]
        
        return HallucinatedSkeleton(
            prediction_index=pred_idx,
            camera=camera,
            detection_score=det_score,
            pred_bbox=tuple(float(x) for x in pred_bbox),
            pred_keypoints=pred_kps.copy(),
            mean_confidence=float(np.mean(confidences)),
            max_confidence=float(np.max(confidences)),
            min_confidence=float(np.min(confidences)),
        )
    
    def _create_missing_skeleton(
        self,
        gt_ped: Pedestrian,
        gt_kps: np.ndarray,
        camera: str,
    ) -> MissingSkeleton:
        """Create a missing skeleton result."""
        num_visible = int((gt_kps[:, 2] == 2).sum())
        num_occluded = int((gt_kps[:, 2] == 1).sum())
        
        gt_bbox = gt_ped.get_bounding_box(camera)
        gt_bbox_tuple = tuple(float(x) for x in gt_bbox) if gt_bbox is not None else None
        
        return MissingSkeleton(
            gt_pedestrian_id=gt_ped.id,
            camera=camera,
            distance_to_camera=self._get_pedestrian_distance(gt_ped, camera),
            num_visible_keypoints=num_visible,
            num_occluded_keypoints=num_occluded,
            gt_bbox=gt_bbox_tuple,
        )
    
    def _create_empty_frame_result(
        self,
        frame: FrameData,
        camera: str,
        gt_pedestrians: List[Pedestrian],
        gt_keypoints_list: List[np.ndarray],
        status: FrameStatus,
        error_message: Optional[str] = None,
        detection_time_ms: Optional[float] = None,
    ) -> FrameResult:
        """Create empty frame result (all GT missing)."""
        missing = [
            self._create_missing_skeleton(ped, kps, camera)
            for ped, kps in zip(gt_pedestrians, gt_keypoints_list)
        ]
        
        return FrameResult(
            frame_id=frame.annotation.frame_id,
            camera=camera,
            timestamp=frame.annotation.timestamp,
            status=status,
            error_message=error_message,
            num_gt_pedestrians=len(gt_pedestrians),
            num_missing=len(gt_pedestrians),
            missing_skeletons=missing,
            detection_time_ms=detection_time_ms,
        )
    
    def evaluate_frame(self, frame: FrameData, camera: str) -> FrameResult:
        """Evaluate a single frame."""
        image = frame.rgb_left if camera == 'left' else frame.rgb_right
        if image is None:
            return FrameResult(
                frame_id=frame.annotation.frame_id,
                camera=camera,
                timestamp=frame.annotation.timestamp,
                status=FrameStatus.NO_IMAGE,
                error_message=f"No {camera} image",
            )
        
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Get GT
        gt_pedestrians = []
        gt_keypoints_list = []
        gt_bboxes = []
        
        for ped in frame.annotation.pedestrians:
            vis_count = ped.visible_keypoints_left if camera == 'left' else ped.visible_keypoints_right
            if vis_count >= self.config.min_gt_visible_keypoints:
                gt_pedestrians.append(ped)
                gt_keypoints_list.append(ped.get_keypoints_array(camera, include_visibility=True))
                bbox = ped.get_bounding_box(camera)
                if bbox is not None:
                    gt_bboxes.append(bbox)
        
        # Detection
        t0 = time.time()
        bboxes = np.array([]).reshape(0, 4)
        scores = np.array([])
        
        if self.config.use_gt_bboxes:
            if gt_bboxes:
                bboxes = self._expand_bboxes(np.array(gt_bboxes), self.config.bbox_padding)
                scores = np.ones(len(bboxes), dtype=np.float32)
        else:
            try:
                bboxes, scores = self.detect_persons(image_bgr)
            except Exception as e:
                self.detection_failure_count += 1
                return self._create_empty_frame_result(
                    frame, camera, gt_pedestrians, gt_keypoints_list,
                    FrameStatus.DETECTION_FAILED, str(e), (time.time() - t0) * 1000
                )
        
        detection_time_ms = (time.time() - t0) * 1000
        
        if len(bboxes) == 0:
            return self._create_empty_frame_result(
                frame, camera, gt_pedestrians, gt_keypoints_list,
                FrameStatus.SUCCESS, None, detection_time_ms
            )
        
        # Pose estimation
        t1 = time.time()
        try:
            predictions = self.estimate_poses(image_bgr, bboxes)
        except Exception as e:
            self.pose_failure_count += 1
            return self._create_empty_frame_result(
                frame, camera, gt_pedestrians, gt_keypoints_list,
                FrameStatus.POSE_FAILED, str(e), detection_time_ms
            )
        
        pose_time_ms = (time.time() - t1) * 1000
        
        # Match with flip correction
        matched_pairs, unmatched_gt, unmatched_pred = self._match_skeletons_with_flip(
            gt_keypoints_list, predictions
        )
        
        # Create results
        matched_skeletons = []
        num_flipped = 0
        
        for gt_idx, pred_idx, was_flipped in matched_pairs:
            det_score = float(scores[pred_idx]) if pred_idx < len(scores) else None
            pred_bbox = bboxes[pred_idx]
            
            matched = self._create_matched_skeleton(
                gt_pedestrians[gt_idx],
                gt_keypoints_list[gt_idx],
                predictions[pred_idx],
                pred_idx,
                camera,
                det_score,
                pred_bbox,
                was_flipped,
            )
            matched_skeletons.append(matched)
            
            if was_flipped:
                num_flipped += 1
        
        hallucinated_skeletons = []
        for pred_idx in unmatched_pred:
            det_score = float(scores[pred_idx]) if pred_idx < len(scores) else None
            halluc = self._create_hallucinated_skeleton(
                predictions[pred_idx], pred_idx, camera, det_score, bboxes[pred_idx]
            )
            hallucinated_skeletons.append(halluc)
        
        missing_skeletons = []
        for gt_idx in unmatched_gt:
            missing = self._create_missing_skeleton(
                gt_pedestrians[gt_idx], gt_keypoints_list[gt_idx], camera
            )
            missing_skeletons.append(missing)
        
        return FrameResult(
            frame_id=frame.annotation.frame_id,
            camera=camera,
            timestamp=frame.annotation.timestamp,
            status=FrameStatus.SUCCESS,
            num_gt_pedestrians=len(gt_pedestrians),
            num_detections=len(bboxes),
            num_predictions=len(predictions),
            num_matched=len(matched_pairs),
            num_hallucinated=len(unmatched_pred),
            num_missing=len(unmatched_gt),
            num_flipped=num_flipped,
            matched_skeletons=matched_skeletons,
            hallucinated_skeletons=hallucinated_skeletons,
            missing_skeletons=missing_skeletons,
            detection_time_ms=detection_time_ms,
            pose_time_ms=pose_time_ms,
        )


# =============================================================================
# Visualization (Smaller nodes/lines)
# =============================================================================

def visualize_evaluation_frame(
    image: np.ndarray,
    frame_result: FrameResult,
    dataset: CARLAStereoPedestrianDataset,
    show_matched: bool = True,
    show_hallucinated: bool = True,
    show_missing: bool = True,
) -> np.ndarray:
    """
    Visualize with smaller, cleaner markers.
    """
    img = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    
    # Colors (BGR)
    GT_COLOR = (0, 200, 0)       # Green
    PRED_COLOR = (255, 100, 0)   # Blue
    HALLUC_COLOR = (0, 0, 220)   # Red
    MISSING_COLOR = (0, 200, 200) # Yellow
    FLIP_COLOR = (255, 0, 255)   # Magenta for flipped
    
    # Sizes (smaller)
    GT_RADIUS = 3
    PRED_RADIUS = 2
    LINE_THICKNESS = 1
    
    # Draw matched
    if show_matched:
        for ms in frame_result.matched_skeletons:
            # Get GT from dataset
            try:
                frame = dataset.get_frame_by_id(frame_result.frame_id)
                gt_ped = next((p for p in frame.annotation.pedestrians if p.id == ms.gt_pedestrian_id), None)
                if gt_ped:
                    gt_kps = gt_ped.get_keypoints_array(frame_result.camera, include_visibility=True)
                else:
                    gt_kps = None
            except:
                gt_kps = None
            
            pred_kps = np.array(ms.pred_keypoints)
            
            # Color based on flip status
            pred_color = FLIP_COLOR if ms.was_flipped else PRED_COLOR
            
            # Draw GT skeleton
            if gt_kps is not None:
                for i, j in COCO_SKELETON:
                    if gt_kps[i, 2] > 0 and gt_kps[j, 2] > 0:
                        pt1 = tuple(gt_kps[i, :2].astype(int))
                        pt2 = tuple(gt_kps[j, :2].astype(int))
                        cv2.line(img, pt1, pt2, GT_COLOR, LINE_THICKNESS)
                
                for idx in range(17):
                    if gt_kps[idx, 2] > 0:
                        pt = tuple(gt_kps[idx, :2].astype(int))
                        cv2.circle(img, pt, GT_RADIUS, GT_COLOR, -1)
            
            # Draw predicted skeleton
            for i, j in COCO_SKELETON:
                if pred_kps[i, 2] > 0.3 and pred_kps[j, 2] > 0.3:
                    pt1 = tuple(pred_kps[i, :2].astype(int))
                    pt2 = tuple(pred_kps[j, :2].astype(int))
                    cv2.line(img, pt1, pt2, pred_color, LINE_THICKNESS)
            
            for idx in range(17):
                if pred_kps[idx, 2] > 0.3:
                    pt = tuple(pred_kps[idx, :2].astype(int))
                    cv2.circle(img, pt, PRED_RADIUS, pred_color, -1)
            
            # Label
            if ms.pred_bbox:
                x1, y1 = int(ms.pred_bbox[0]), int(ms.pred_bbox[1])
                label = f"ID:{ms.gt_pedestrian_id} d:{ms.mean_deviation:.1f}"
                if ms.was_flipped:
                    label += " [F]"
                cv2.putText(img, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, pred_color, 1)
    
    # Draw hallucinated
    if show_hallucinated:
        for hs in frame_result.hallucinated_skeletons:
            pred_kps = np.array(hs.pred_keypoints)
            
            for i, j in COCO_SKELETON:
                if pred_kps[i, 2] > 0.3 and pred_kps[j, 2] > 0.3:
                    pt1 = tuple(pred_kps[i, :2].astype(int))
                    pt2 = tuple(pred_kps[j, :2].astype(int))
                    cv2.line(img, pt1, pt2, HALLUC_COLOR, LINE_THICKNESS)
            
            for idx in range(17):
                if pred_kps[idx, 2] > 0.3:
                    pt = tuple(pred_kps[idx, :2].astype(int))
                    cv2.circle(img, pt, PRED_RADIUS, HALLUC_COLOR, -1)
            
            # Bbox
            x1, y1, x2, y2 = [int(x) for x in hs.pred_bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), HALLUC_COLOR, 1)
            cv2.putText(img, f"HALL c:{hs.mean_confidence:.2f}", (x1, y1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, HALLUC_COLOR, 1)
    
    # Draw missing
    if show_missing:
        for ms in frame_result.missing_skeletons:
            try:
                frame = dataset.get_frame_by_id(frame_result.frame_id)
                gt_ped = next((p for p in frame.annotation.pedestrians if p.id == ms.gt_pedestrian_id), None)
                if gt_ped:
                    gt_kps = gt_ped.get_keypoints_array(frame_result.camera, include_visibility=True)
                    
                    for i, j in COCO_SKELETON:
                        if gt_kps[i, 2] > 0 and gt_kps[j, 2] > 0:
                            pt1 = tuple(gt_kps[i, :2].astype(int))
                            pt2 = tuple(gt_kps[j, :2].astype(int))
                            cv2.line(img, pt1, pt2, MISSING_COLOR, LINE_THICKNESS)
                    
                    for idx in range(17):
                        if gt_kps[idx, 2] > 0:
                            pt = tuple(gt_kps[idx, :2].astype(int))
                            thickness = -1 if gt_kps[idx, 2] == 2 else 1
                            cv2.circle(img, pt, GT_RADIUS, MISSING_COLOR, thickness)
            except:
                pass
            
            if ms.gt_bbox:
                x1, y1, x2, y2 = [int(x) for x in ms.gt_bbox]
                cv2.rectangle(img, (x1, y1), (x2, y2), MISSING_COLOR, 1)
                dist_str = f" {ms.distance_to_camera:.0f}m" if ms.distance_to_camera else ""
                cv2.putText(img, f"MISS ID:{ms.gt_pedestrian_id}{dist_str}", (x1, y1 - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, MISSING_COLOR, 1)
    
    # Info overlay
    flip_str = f" Flip:{frame_result.num_flipped}" if frame_result.num_flipped > 0 else ""
    info = (f"F:{frame_result.frame_id} {frame_result.camera} | "
            f"M:{frame_result.num_matched} H:{frame_result.num_hallucinated} "
            f"Miss:{frame_result.num_missing}{flip_str}")
    
    cv2.putText(img, info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
    cv2.putText(img, info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# =============================================================================
# Running Statistics
# =============================================================================

class RunningStats:
    """Welford's online algorithm for running statistics."""
    
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
    
    def update(self, value: float):
        if not np.isfinite(value):
            return
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
    
    @property
    def std(self) -> float:
        return np.sqrt(self.M2 / (self.n - 1)) if self.n > 1 else 0.0
    
    def to_dict(self) -> dict:
        return {
            'count': self.n,
            'mean': self.mean if self.n > 0 else 0.0,
            'std': self.std,
            'min': self.min_val if self.n > 0 else 0.0,
            'max': self.max_val if self.n > 0 else 0.0,
        }


# =============================================================================
# Incremental Statistics
# =============================================================================

class IncrementalStatistics:
    """Memory-efficient incremental statistics accumulator."""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        
        # Counters
        self.total_frames = 0
        self.successful_frames = 0
        self.detection_failed = 0
        self.pose_failed = 0
        
        self.total_gt = 0
        self.total_matched = 0
        self.total_hallucinated = 0
        self.total_missing = 0
        self.total_flipped = 0
        
        # Running stats
        self.deviation_stats = RunningStats()
        self.iou_stats = RunningStats()
        self.confidence_stats = RunningStats()
        
        # Per-keypoint stats
        self.per_keypoint = {
            kp: {'deviation': RunningStats(), 'confidence': RunningStats()}
            for kp in COCO_KEYPOINTS
        }
        
        # Per-distance
        self.per_distance: Dict[str, Dict] = defaultdict(lambda: {
            'matched': 0, 'missing': 0, 'deviation': RunningStats()
        })
    
    def _get_distance_label(self, dist: Optional[float]) -> str:
        if dist is None:
            return "unknown"
        for low, high in self.config.distance_ranges:
            if low <= dist < high:
                return f"{low}m+" if high == float('inf') else f"{low}-{high}m"
        return "unknown"
    
    def update(self, result: FrameResult, dataset: CARLAStereoPedestrianDataset):
        """Update with a frame result."""
        self.total_frames += 1
        
        if result.status == FrameStatus.SUCCESS:
            self.successful_frames += 1
        elif result.status == FrameStatus.DETECTION_FAILED:
            self.detection_failed += 1
        elif result.status == FrameStatus.POSE_FAILED:
            self.pose_failed += 1
        
        self.total_gt += result.num_gt_pedestrians
        self.total_matched += result.num_matched
        self.total_hallucinated += result.num_hallucinated
        self.total_missing += result.num_missing
        self.total_flipped += result.num_flipped
        
        # Process matched
        for ms in result.matched_skeletons:
            self.deviation_stats.update(ms.mean_deviation)
            self.confidence_stats.update(ms.mean_confidence)
            
            if ms.bbox_iou is not None:
                self.iou_stats.update(ms.bbox_iou)
            
            dist_label = self._get_distance_label(ms.distance_to_camera)
            self.per_distance[dist_label]['matched'] += 1
            self.per_distance[dist_label]['deviation'].update(ms.mean_deviation)
            
            # Per-keypoint (need GT for this)
            try:
                frame = dataset.get_frame_by_id(result.frame_id)
                gt_ped = next((p for p in frame.annotation.pedestrians if p.id == ms.gt_pedestrian_id), None)
                if gt_ped:
                    gt_kps = gt_ped.get_keypoints_array(result.camera, include_visibility=True)
                    pred_kps = np.array(ms.pred_keypoints)
                    
                    for idx, kp_name in enumerate(COCO_KEYPOINTS):
                        if gt_kps[idx, 2] > 0:
                            dev = float(np.linalg.norm(gt_kps[idx, :2] - pred_kps[idx, :2]))
                            self.per_keypoint[kp_name]['deviation'].update(dev)
                        self.per_keypoint[kp_name]['confidence'].update(pred_kps[idx, 2])
            except:
                pass
        
        # Process missing
        for ms in result.missing_skeletons:
            dist_label = self._get_distance_label(ms.distance_to_camera)
            self.per_distance[dist_label]['missing'] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        precision = self.total_matched / (self.total_matched + self.total_hallucinated) if (self.total_matched + self.total_hallucinated) > 0 else 0
        recall = self.total_matched / self.total_gt if self.total_gt > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'overall': {
                'total_frames': self.total_frames,
                'successful_frames': self.successful_frames,
                'detection_failed': self.detection_failed,
                'pose_failed': self.pose_failed,
                'total_gt': self.total_gt,
                'total_matched': self.total_matched,
                'total_hallucinated': self.total_hallucinated,
                'total_missing': self.total_missing,
                'total_flipped': self.total_flipped,
                'flip_rate': self.total_flipped / self.total_matched if self.total_matched > 0 else 0,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'mean_deviation': self.deviation_stats.mean,
                'std_deviation': self.deviation_stats.std,
                'mean_iou': self.iou_stats.mean,
                'mean_confidence': self.confidence_stats.mean,
            },
            'per_distance': {
                label: {
                    'matched': data['matched'],
                    'missing': data['missing'],
                    'recall': data['matched'] / (data['matched'] + data['missing']) if (data['matched'] + data['missing']) > 0 else 0,
                    'mean_deviation': data['deviation'].mean,
                }
                for label, data in self.per_distance.items()
            },
            'per_keypoint': {
                kp: {
                    'deviation': data['deviation'].to_dict(),
                    'confidence': {'mean': data['confidence'].mean, 'std': data['confidence'].std},
                }
                for kp, data in self.per_keypoint.items()
            },
        }
    
    def print_summary(self):
        stats = self.get_statistics()
        overall = stats['overall']
        
        print("\n" + "=" * 70)
        print("Evaluation Summary".center(70))
        print("=" * 70)
        
        print(f"\n  Frames: {overall['total_frames']} (Success: {overall['successful_frames']}, "
              f"Det Fail: {overall['detection_failed']}, Pose Fail: {overall['pose_failed']})")
        print(f"  GT: {overall['total_gt']}, Matched: {overall['total_matched']}, "
              f"Halluc: {overall['total_hallucinated']}, Missing: {overall['total_missing']}")
        print(f"  Flipped: {overall['total_flipped']} ({overall['flip_rate']:.1%} of matches)")
        print(f"\n  Precision: {overall['precision']:.4f}")
        print(f"  Recall:    {overall['recall']:.4f}")
        print(f"  F1:        {overall['f1']:.4f}")
        print(f"\n  Mean Deviation: {overall['mean_deviation']:.2f} px")
        print(f"  Mean IoU:       {overall['mean_iou']:.4f}")
        
        print("\n  Per-Distance:")
        for dist, data in sorted(stats['per_distance'].items(), 
                                  key=lambda x: float(x[0].split('-')[0].replace('m+','').replace('m','') or '999')):
            print(f"    {dist:12s}: Recall={data['recall']:.2%}, Dev={data['mean_deviation']:.1f}px")
        
        print("\n" + "=" * 70)


# =============================================================================
# Streaming Writer
# =============================================================================

class StreamingResultsWriter:
    """Write results to disk as they're generated."""
    
    def __init__(self, output_dir: str, compress: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if compress:
            self.filepath = self.output_dir / "results.jsonl.gz"
            self.file = gzip.open(self.filepath, 'wt', encoding='utf-8')
        else:
            self.filepath = self.output_dir / "results.jsonl"
            self.file = open(self.filepath, 'w', encoding='utf-8')
        
        self.count = 0
    
    def write(self, result: FrameResult):
        line = json.dumps(convert_to_serializable(result.to_dict()), cls=NumpyEncoder)
        self.file.write(line + '\n')
        self.count += 1
        
        if self.count % 100 == 0:
            self.file.flush()
    
    def close(self):
        self.file.close()
        print(f"  Written {self.count} results to {self.filepath}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# =============================================================================
# Main Evaluation
# =============================================================================

def run_evaluation(
    session_path: str,
    output_dir: str,
    pose_config: str,
    pose_checkpoint: str,
    det_config: str,
    det_checkpoint: str,
    max_frames: Optional[int] = None,
    enable_flip: bool = True,
    save_visualizations: bool = True,
    vis_interval: int = 500,
    max_vis: int = 200,
    device: str = 'cuda:0',
):
    """
    Run streaming evaluation with flip correction.
    
    Args:
        session_path: Path to CARLA session
        output_dir: Output directory for results
        pose_config: Path to pose model config
        pose_checkpoint: Path to pose model weights
        det_config: Path to detector config
        det_checkpoint: Path to detector weights
        max_frames: Maximum frames to process (None = all)
        enable_flip: Enable left-right flip correction
        save_visualizations: Save sample visualizations
        vis_interval: Save visualization every N frames
        max_vis: Maximum number of visualizations to save
        device: Device for inference
    """
    print("=" * 70)
    print("RTMPose Streaming Evaluation with Flip Correction")
    print("=" * 70)
    
    # Create config
    config = EvalConfig(
        pose=PoseModelConfig(
            config_file=pose_config,
            checkpoint_file=pose_checkpoint,
        ),
        detector=DetectorConfig(
            config_file=det_config,
            checkpoint_file=det_checkpoint,
            score_threshold=0.5,
        ),
        device=device,
        enable_flip_correction=enable_flip,
        max_deviation_threshold=100.0,
        cameras=['left', 'right'],
    )
    
    print(f"\nConfiguration:")
    print(f"  Session: {session_path}")
    print(f"  Output: {output_dir}")
    print(f"  Flip correction: {enable_flip}")
    print(f"  Max frames: {max_frames or 'all'}")
    
    # Load dataset
    print(f"\nLoading dataset...")
    dataset = CARLAStereoPedestrianDataset(
        session_path,
        load_images=True,
        load_depth=False,
        cameras=['left', 'right'],
    )
    print(f"  Loaded {len(dataset)} frames")
    
    # Initialize
    print(f"\nInitializing models...")
    evaluator = RTMPoseEvaluator(config)
    stats = IncrementalStatistics(config)
    
    # Setup output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if save_visualizations:
        vis_dir = output_path / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
    
    # Determine number of frames
    num_frames = min(len(dataset), max_frames) if max_frames else len(dataset)
    
    print(f"\nStarting evaluation of {num_frames} frames...")
    print("-" * 70)
    
    vis_count = 0
    start_time = time.time()
    
    with StreamingResultsWriter(output_dir, compress=True) as writer:
        for idx in range(num_frames):
            frame = dataset[idx]
            
            for camera in config.cameras:
                # Evaluate
                result = evaluator.evaluate_frame(frame, camera)
                
                # Write to disk
                writer.write(result)
                
                # Update stats
                stats.update(result, dataset)
                
                # Save visualization
                if save_visualizations and vis_count < max_vis:
                    should_vis = (
                        (idx % vis_interval == 0) or
                        (result.status != FrameStatus.SUCCESS) or
                        (result.num_flipped > 0) or
                        (result.num_hallucinated > 0 and result.num_hallucinated <= 3)
                    )
                    
                    if should_vis:
                        try:
                            image = frame.rgb_left if camera == 'left' else frame.rgb_right
                            if image is not None:
                                vis_img = visualize_evaluation_frame(image, result, dataset)
                                
                                # Category prefix for filename
                                if result.status != FrameStatus.SUCCESS:
                                    prefix = result.status.value
                                elif result.num_flipped > 0:
                                    prefix = "flipped"
                                elif result.num_hallucinated > 0:
                                    prefix = "halluc"
                                else:
                                    prefix = "regular"
                                
                                vis_path = vis_dir / f"{prefix}_{result.frame_id:06d}_{camera}.jpg"
                                cv2.imwrite(str(vis_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR),
                                           [cv2.IMWRITE_JPEG_QUALITY, 85])
                                vis_count += 1
                        except Exception as e:
                            pass
                
                # Free result memory
                del result
            
            # Progress
            if (idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                fps = (idx + 1) / elapsed
                eta = (num_frames - idx - 1) / fps if fps > 0 else 0
                
                print(f"  [{idx+1:6d}/{num_frames}] {fps:.1f} fps | "
                      f"Recall: {stats.total_matched/stats.total_gt:.3f} | "
                      f"Flipped: {stats.total_flipped} | "
                      f"ETA: {eta/60:.1f}min")
            
            # Memory cleanup
            if (idx + 1) % 50 == 0:
                gc.collect()
                if TORCH_AVAILABLE:
                    torch.cuda.empty_cache()
            
            del frame
    
    # Final timing
    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time/60:.1f} minutes ({num_frames/total_time:.1f} fps)")
    
    # Save statistics
    final_stats = stats.get_statistics()
    stats_path = output_path / 'statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(convert_to_serializable(final_stats), f, indent=2)
    print(f"  Statistics saved to {stats_path}")
    
    # Save config
    config_dict = {
        'pose_config': pose_config,
        'pose_checkpoint': pose_checkpoint,
        'det_config': det_config,
        'det_checkpoint': det_checkpoint,
        'enable_flip_correction': enable_flip,
        'max_deviation_threshold': config.max_deviation_threshold,
        'session_path': session_path,
        'num_frames': num_frames,
        'total_time_seconds': total_time,
    }
    config_path = output_path / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Print summary
    stats.print_summary()
    
    print(f"\nResults saved to: {output_dir}")
    print("=" * 70)
    
    return final_stats


# =============================================================================
# Results Reader (for later analysis)
# =============================================================================

class ResultsReader:
    """
    Read streaming results from disk for analysis.
    
    Provides iterator interface to avoid loading all results into memory.
    """
    
    def __init__(self, results_path: str):
        self.path = Path(results_path)
        
        if self.path.suffix == '.gz':
            self.compressed = True
        else:
            self.compressed = False
    
    def __iter__(self):
        """Iterate over results one at a time."""
        if self.compressed:
            f = gzip.open(self.path, 'rt', encoding='utf-8')
        else:
            f = open(self.path, 'r', encoding='utf-8')
        
        try:
            for line in f:
                yield json.loads(line.strip())
        finally:
            f.close()
    
    def count(self) -> int:
        """Count total results."""
        return sum(1 for _ in self)
    
    def filter(self, predicate):
        """Filter results by predicate function."""
        for result in self:
            if predicate(result):
                yield result
    
    def get_flipped(self):
        """Get results where predictions were flipped."""
        return self.filter(lambda r: r.get('num_flipped', 0) > 0)
    
    def get_failures(self):
        """Get failed frames."""
        return self.filter(lambda r: r['status'] != 'success')
    
    def get_hallucinations(self):
        """Get frames with hallucinations."""
        return self.filter(lambda r: r.get('num_hallucinated', 0) > 0)
    
    def get_by_frame_id(self, frame_id: int):
        """Get all results for a specific frame ID."""
        return self.filter(lambda r: r['frame_id'] == frame_id)


def analyze_results(results_path: str, output_path: Optional[str] = None):
    """
    Analyze saved results and generate additional reports.
    
    Args:
        results_path: Path to results.jsonl.gz
        output_path: Optional path for analysis output
    """
    print(f"Analyzing results from: {results_path}")
    
    reader = ResultsReader(results_path)
    
    # Collect analysis data
    flip_cases = []
    high_deviation = []
    hallucination_cases = []
    failure_cases = []
    
    per_keypoint_flips = defaultdict(int)
    
    for result in reader:
        # Track flips
        if result.get('num_flipped', 0) > 0:
            flip_cases.append({
                'frame_id': result['frame_id'],
                'camera': result['camera'],
                'num_flipped': result['num_flipped'],
            })
        
        # Track high deviation
        for ms in result.get('matched_skeletons', []):
            if ms['mean_deviation'] > 30:
                high_deviation.append({
                    'frame_id': result['frame_id'],
                    'camera': result['camera'],
                    'gt_id': ms['gt_pedestrian_id'],
                    'deviation': ms['mean_deviation'],
                    'was_flipped': ms['was_flipped'],
                })
        
        # Track hallucinations
        if result.get('num_hallucinated', 0) > 0:
            hallucination_cases.append({
                'frame_id': result['frame_id'],
                'camera': result['camera'],
                'num_hallucinated': result['num_hallucinated'],
                'confidence': np.mean([h['mean_confidence'] for h in result['hallucinated_skeletons']]),
            })
        
        # Track failures
        if result['status'] != 'success':
            failure_cases.append({
                'frame_id': result['frame_id'],
                'camera': result['camera'],
                'status': result['status'],
                'error': result.get('error_message'),
            })
    
    # Print analysis
    print(f"\n{'='*60}")
    print("Results Analysis")
    print(f"{'='*60}")
    
    print(f"\nFlip Correction:")
    print(f"  Total flipped predictions: {sum(f['num_flipped'] for f in flip_cases)}")
    print(f"  Frames with flips: {len(flip_cases)}")
    
    print(f"\nHigh Deviation (>30px):")
    print(f"  Total cases: {len(high_deviation)}")
    if high_deviation:
        flipped_high_dev = sum(1 for h in high_deviation if h['was_flipped'])
        print(f"  Of which flipped: {flipped_high_dev}")
    
    print(f"\nHallucinations:")
    print(f"  Frames with hallucinations: {len(hallucination_cases)}")
    if hallucination_cases:
        avg_conf = np.mean([h['confidence'] for h in hallucination_cases])
        print(f"  Average confidence: {avg_conf:.3f}")
    
    print(f"\nFailures:")
    print(f"  Total failed frames: {len(failure_cases)}")
    if failure_cases:
        by_status = defaultdict(int)
        for f in failure_cases:
            by_status[f['status']] += 1
        for status, count in by_status.items():
            print(f"    {status}: {count}")
    
    # Save detailed analysis
    if output_path:
        analysis = {
            'flip_cases': flip_cases[:100],  # Limit size
            'high_deviation': sorted(high_deviation, key=lambda x: x['deviation'], reverse=True)[:100],
            'hallucination_cases': hallucination_cases[:100],
            'failure_cases': failure_cases[:100],
            'summary': {
                'total_flipped': sum(f['num_flipped'] for f in flip_cases),
                'frames_with_flips': len(flip_cases),
                'high_deviation_count': len(high_deviation),
                'hallucination_frames': len(hallucination_cases),
                'failure_count': len(failure_cases),
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nDetailed analysis saved to: {output_path}")
    
    return analysis


# =============================================================================
# Configuration
# =============================================================================

# Paths - adjust to your setup
MMPOSE_ROOT = Path.home() / 'RTMPose' / 'mmpose'
CHECKPOINTS = MMPOSE_ROOT / 'checkpoints'
CONFIGS = MMPOSE_ROOT / 'configs'

SESSION_PATH = '/home/theta/carla/output/sessions/session_20260116_230516'
OUTPUT_DIR = './eval_output_flip'

# Model paths
POSE_CONFIG = str(CONFIGS / 'body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py')
POSE_CHECKPOINT = str(CHECKPOINTS / 'td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth')
DET_CONFIG = str(CHECKPOINTS / 'det-config.py')
DET_CHECKPOINT = str(CHECKPOINTS / 'det-weights.pth')

# Settings
MAX_FRAMES = None  # None for full dataset
ENABLE_FLIP = True
SAVE_VIS = True
VIS_INTERVAL = 500
MAX_VIS = 200
DEVICE = 'cuda:0'


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='RTMPose Streaming Evaluation')
    parser.add_argument('--session', type=str, default=SESSION_PATH, help='Session path')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR, help='Output directory')
    parser.add_argument('--max-frames', type=int, default=None, help='Max frames')
    parser.add_argument('--no-flip', action='store_true', help='Disable flip correction')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualizations')
    parser.add_argument('--analyze', type=str, default=None, help='Analyze existing results file')
    
    args = parser.parse_args()
    
    if args.analyze:
        # Analysis mode
        analyze_results(args.analyze, args.analyze.replace('.jsonl', '_analysis.json'))
    else:
        # Evaluation mode
        run_evaluation(
            session_path=args.session,
            output_dir=args.output,
            pose_config=POSE_CONFIG,
            pose_checkpoint=POSE_CHECKPOINT,
            det_config=DET_CONFIG,
            det_checkpoint=DET_CHECKPOINT,
            max_frames=args.max_frames,
            enable_flip=not args.no_flip,
            save_visualizations=not args.no_vis,
            vis_interval=VIS_INTERVAL,
            max_vis=MAX_VIS,
            device=DEVICE,
        )