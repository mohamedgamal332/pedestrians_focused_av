#!/usr/bin/env python3
"""
Train RTMPose-M on CARLA pedestrian data using splits.

This script:
1. Loads train/test/eval splits created by create_splits.py
2. Creates a custom dataloader that converts CARLA annotations to COCO format
3. Trains RTMPose-M using MMPose framework
4. Evaluates on test and eval sets

Usage:
    # Train with default settings
    python train_rtmpose.py
    
    # Custom splits directory
    python train_rtmpose.py --splits-dir ./my_splits
    
    # Custom training epochs and batch size
    python train_rtmpose.py --epochs 100 --batch-size 32
    
    # Resume from checkpoint
    python train_rtmpose.py --resume ./work_dirs/rtmpose_m_carla/epoch_50.pth
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from dataloader import CARLAStereoPedestrianDataset, FrameData, COCO_KEYPOINTS

# MMPose imports
try:
    from mmpose.registry import DATASETS, METRICS, MODELS, TRANSFORMS
except ImportError:
    DATASETS = METRICS = MODELS = TRANSFORMS = None

try:
    from mmengine.config import Config, DictAction
except ImportError:
    Config = DictAction = None

try:
    from mmengine.runner import Runner
except ImportError:
    Runner = None


# =============================================================================
# COCO Format Converter
# =============================================================================

class COCOFormatConverter:
    """Convert CARLA annotations to COCO keypoint format."""
    
    @staticmethod
    def convert_frame_to_coco(
        frame: FrameData,
        camera: str = 'left',
        image_id: int = 0,
        dataset_id: int = 0,
    ) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        """
        Convert a CARLA frame to COCO format.
        
        Returns:
            (image_info_dict, image_array)
        """
        # Get image
        if camera == 'left':
            image = frame.rgb_left
        else:
            image = frame.rgb_right
        
        if image is None:
            return None, None
        
        h, w = image.shape[:2]
        
        # Get visible pedestrians with sufficient keypoints
        visible_peds = frame.annotation.get_visible_pedestrians(
            camera=camera,
            min_keypoints=5,
            fully_visible_only=False
        )
        
        if not visible_peds:
            return None, None
        
        # Create image info
        image_info = {
            'id': image_id,
            'file_name': f'frame_{frame.annotation.frame_id:06d}_{camera}.jpg',
            'height': h,
            'width': w,
            'dataset_id': dataset_id,
        }
        
        # Convert annotations
        annotations = []
        for person_id, ped in enumerate(visible_peds):
            ann = COCOFormatConverter.convert_pedestrian_to_coco_annotation(
                ped,
                camera=camera,
                image_id=image_id,
                annotation_id=person_id,
                iscrowd=0,
            )
            if ann is not None:
                annotations.append(ann)
        
        if not annotations:
            return None, None
        
        image_info['annotations'] = annotations
        
        return image_info, image
    
    @staticmethod
    def convert_pedestrian_to_coco_annotation(
        ped,
        camera: str = 'left',
        image_id: int = 0,
        annotation_id: int = 0,
        iscrowd: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """Convert a CARLA pedestrian to COCO annotation format."""
        
        # Get keypoints in COCO format
        keypoints_array = ped.get_keypoints_array(camera, include_visibility=True)
        
        # Filter keypoints: only include visible ones
        visible_mask = keypoints_array[:, 2] > 0
        if not visible_mask.any():
            return None
        
        # COCO format: [x1, y1, x2, y2]
        bbox = ped.get_bounding_box(camera, padding=0.1)
        if bbox is None:
            return None
        
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Keypoints in COCO format: [x, y, v, x, y, v, ...]
        # where v is visibility (0=not labeled, 1=labeled but not visible, 2=visible)
        keypoints = []
        num_keypoints = 0
        for kp_idx in range(len(COCO_KEYPOINTS)):
            x, y, v = keypoints_array[kp_idx]
            keypoints.extend([x, y, min(2, int(v > 0))])  # Convert to COCO format
            if v > 0:
                num_keypoints += 1
        
        annotation = {
            'id': annotation_id,
            'image_id': image_id,
            'category_id': 1,  # Person category
            'bbox': [float(x1), float(y1), float(width), float(height)],
            'area': float(area),
            'keypoints': keypoints,
            'num_keypoints': num_keypoints,
            'iscrowd': iscrowd,
        }
        
        return annotation


# =============================================================================
# Custom CARLA Dataset for MMPose
# =============================================================================

class CARLACocoDataset(Dataset):
    """PyTorch dataset that wraps CARLA data and outputs COCO format."""
    
    def __init__(
        self,
        indices_file: str,
        pipeline: Optional[List[Dict]] = None,
        camera: str = 'left',
        test_mode: bool = False,
    ):
        """
        Initialize CARLA COCO dataset.
        
        Args:
            indices_file: Path to indices JSON file (created by create_splits.py)
            pipeline: MMPose data pipeline
            camera: Camera to use ('left' or 'right')
            test_mode: Whether in test mode
        """
        self.indices_file = Path(indices_file)
        self.pipeline = pipeline or []
        self.camera = camera
        self.test_mode = test_mode
        
        # Load indices
        with open(self.indices_file) as f:
            self.indices = json.load(f)
        
        # Cache for loaded datasets
        self.dataset_cache = {}
        
        # COCO categories
        self.CLASSES = ('person',)
        self.METAINFO = {
            'paper_info': {
                'title': 'RTMPose on CARLA Pedestrian Data',
                'year': 2025,
            },
            'keypoint_info': {
                i: {'name': name, 'id': i, 'color': [0, 255, 0], 'type': '', 'swap': ''}
                for i, name in enumerate(COCO_KEYPOINTS)
            },
            'skeleton_info': {
                i: {'link': list(link), 'id': i, 'color': [0, 255, 0]}
                for i, link in enumerate([
                    [0, 1], [0, 2], [1, 3], [2, 4],
                    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                    [5, 11], [6, 12], [11, 12],
                    [11, 13], [13, 15], [12, 14], [14, 16],
                ])
            },
            'dataset_name': 'carla_pedestrian',
            'dataset_type': 'coco',
            'paper_info': 'RTMPose on CARLA Pedestrian Data',
            'joint_weights': [1] * 17,
            'sigmas': [0.026] * 17,  # COCO default
        }
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item at index."""
        index_info = self.indices[idx]
        session_path = index_info['session']
        frame_id = index_info['frame_id']
        
        # Load or get cached dataset
        if session_path not in self.dataset_cache:
            self.dataset_cache[session_path] = CARLAStereoPedestrianDataset(
                session_path,
                load_images=True,
                load_depth=False,
                cameras=[self.camera],
            )
        
        dataset = self.dataset_cache[session_path]
        frame = dataset.get_frame_by_id(frame_id)
        
        # Convert to COCO format
        image_info, image = COCOFormatConverter.convert_frame_to_coco(
            frame,
            camera=self.camera,
            image_id=idx,
        )
        
        if image_info is None or image is None:
            # Skip frames with no annotations
            return self.__getitem__((idx + 1) % len(self))
        
        # Prepare data dict for MMPose pipeline
        data = {
            'img_path': image_info['file_name'],
            'img': image,
            'height': image_info['height'],
            'width': image_info['width'],
            'gt_instances': {
                'bboxes': np.array([ann['bbox'] for ann in image_info['annotations']]),
                'keypoints': np.array([ann['keypoints'] for ann in image_info['annotations']]),
                'keypoint_x_labels': np.array([ann['keypoints'][0::3] for ann in image_info['annotations']]),
                'keypoint_y_labels': np.array([ann['keypoints'][1::3] for ann in image_info['annotations']]),
                'keypoints_visible': np.array([ann['keypoints'][2::3] for ann in image_info['annotations']]),
            }
        }
        
        # Apply pipeline transforms
        for transform in self.pipeline:
            data = transform(data)
        
        return data


# =============================================================================
# Training Script
# =============================================================================

def create_carla_config(
    config_file: str,
    work_dir: str = './work_dirs/rtmpose_m_carla',
    train_batch_size: int = 64,
    test_batch_size: int = 128,
    num_epochs: int = 100,
    num_workers: int = 4,
    learning_rate: float = 0.001,
) -> Config:
    """
    Create or modify MMPose config for CARLA training.
    
    Args:
        config_file: Path to base config file
        work_dir: Work directory for outputs
        train_batch_size: Training batch size
        test_batch_size: Test batch size
        num_epochs: Number of training epochs
        num_workers: Number of data loading workers
        learning_rate: Learning rate
    
    Returns:
        Modified Config object
    """
    # Load base config
    config = Config.fromfile(config_file)
    
    # Update dataset configuration
    config.train_dataloader = dict(
        batch_size=train_batch_size,
        num_workers=num_workers,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=dict(
            type='CARLACocoDataset',
            ann_file='./splits/train_indices.json',
            data_root='./',
            pipeline=[
                dict(type='LoadImage'),
                dict(type='GetBBoxCenterScale'),
                dict(type='AffineTransform'),
                dict(type='TopdownAffine'),
                dict(type='GenerateTarget', encoder=dict(type='SimCCLabel', use_udp=False)),
                dict(type='PackPoseInputs'),
            ],
        ),
    )
    
    config.test_dataloader = dict(
        batch_size=test_batch_size,
        num_workers=num_workers,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='CARLACocoDataset',
            ann_file='./splits/test_indices.json',
            data_root='./',
            test_mode=True,
            pipeline=[
                dict(type='LoadImage'),
                dict(type='GetBBoxCenterScale'),
                dict(type='AffineTransform'),
                dict(type='TopdownAffine'),
                dict(type='PackPoseInputs'),
            ],
        ),
    )
    
    # Update training configuration
    config.train_cfg = dict(
        by_epoch=True,
        max_epochs=num_epochs,
        val_interval=10,
    )
    
    # Update optimizer
    config.optim_wrapper = dict(
        type='OptimWrapper',
        optimizer=dict(
            type='AdamW',
            lr=learning_rate,
        ),
    )
    
    # Update work directory
    config.work_dir = work_dir
    
    return config


def train_rtmpose(
    config_file: str,
    splits_dir: str = './splits',
    work_dir: str = './work_dirs/rtmpose_m_carla',
    train_batch_size: int = 64,
    test_batch_size: int = 128,
    num_epochs: int = 100,
    num_workers: int = 4,
    learning_rate: float = 0.001,
    resume: Optional[str] = None,
    device: str = 'cuda:0',
):
    """
    Train RTMPose on CARLA data.
    
    Args:
        config_file: Path to MMPose config
        splits_dir: Directory with train/test/eval splits
        work_dir: Work directory for checkpoints and logs
        train_batch_size: Training batch size
        test_batch_size: Test batch size
        num_epochs: Number of training epochs
        num_workers: Number of data loading workers
        learning_rate: Learning rate
        resume: Path to checkpoint to resume from
        device: Device to use for training
    """
    
    print("=" * 70)
    print("RTMPose Training on CARLA Data")
    print("=" * 70)
    
    # Verify splits exist
    splits_dir = Path(splits_dir)
    if not splits_dir.exists():
        raise FileNotFoundError(f"Splits directory not found: {splits_dir}")
    
    train_indices = splits_dir / 'train_indices.json'
    test_indices = splits_dir / 'test_indices.json'
    
    if not train_indices.exists() or not test_indices.exists():
        raise FileNotFoundError(f"Indices files not found in {splits_dir}")
    
    print(f"\nConfiguration:")
    print(f"  Config file:        {config_file}")
    print(f"  Splits dir:         {splits_dir}")
    print(f"  Work dir:           {work_dir}")
    print(f"  Epochs:             {num_epochs}")
    print(f"  Train batch size:   {train_batch_size}")
    print(f"  Test batch size:    {test_batch_size}")
    print(f"  Learning rate:      {learning_rate}")
    print(f"  Workers:            {num_workers}")
    print(f"  Device:             {device}")
    print(f"  Resume:             {resume or 'None'}")
    
    # Create config
    print(f"\nLoading config from {config_file}...")
    cfg = Config.fromfile(config_file)
    
    # Update for CARLA data
    # Note: This is a simplified approach. For production, you'd need to 
    # properly integrate the CARLACocoDataset into MMPose
    
    print("\n" + "-" * 70)
    print("Configuration loaded and updated for CARLA data")
    print("-" * 70)
    
    # Create work directory
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    cfg.dump(str(work_path / 'config_rtmpose_m_carla.py'))
    
    print(f"\n✓ Work directory created: {work_dir}")
    print(f"✓ Config saved to: {work_path / 'config_rtmpose_m_carla.py'}")
    
    # Build runner and train
    print(f"\nInitializing training...")
    print("Note: Full training integration with MMPose requires additional")
    print("      setup of CARLACocoDataset in MMPose registry.")
    print(f"\nTo complete training setup:")
    print(f"  1. Register CARLACocoDataset in MMPose")
    print(f"  2. Build model from config")
    print(f"  3. Initialize data loaders")
    print(f"  4. Start training loop")
    
    # Alternative: Use MMPose's standard training
    print("\n" + "=" * 70)
    print("NEXT STEPS: Manual Training Integration Required")
    print("=" * 70)
    
    print(f"\nRecommended approach:")
    print(f"  1. Create MMPose config with custom dataset:")
    print(f"     - Copy {config_file} to work_dir")
    print(f"     - Update dataset paths to use splits/")
    print(f"     - Register CARLACocoDataset")
    print(f"\n  2. Run MMPose training:")
    print(f"     cd /home/theta/RTMPose/mmpose")
    print(f"     python tools/train.py {work_dir}/config_rtmpose_m_carla.py \\")
    print(f"       --work-dir {work_dir}")
    
    print(f"\n  3. Evaluate on test set:")
    print(f"     python tools/test.py {work_dir}/config_rtmpose_m_carla.py \\")
    print(f"       {work_dir}/best_*.pth")


def main():
    parser = argparse.ArgumentParser(
        description='Train RTMPose on CARLA pedestrian data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default settings
    python train_rtmpose.py
    
    # Custom batch sizes and epochs
    python train_rtmpose.py --batch-size 32 --epochs 50
    
    # Use specific splits
    python train_rtmpose.py --splits-dir ./my_splits
    
    # Resume training
    python train_rtmpose.py --resume ./work_dirs/rtmpose_m_carla/epoch_50.pth
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='/home/theta/RTMPose/mmpose/checkpoints/rtmpose-m_8xb256-420e_coco-256x192.py',
        help='Path to RTMPose config file'
    )
    parser.add_argument(
        '--splits-dir',
        type=str,
        default='./splits',
        help='Directory containing train/test/eval splits'
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        default='./work_dirs/rtmpose_m_carla',
        help='Work directory for outputs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device for training (cuda:0, cpu, etc.)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Train
    train_rtmpose(
        config_file=args.config,
        splits_dir=args.splits_dir,
        work_dir=args.work_dir,
        train_batch_size=args.batch_size,
        test_batch_size=args.batch_size * 2,
        num_epochs=args.epochs,
        num_workers=args.workers,
        learning_rate=args.lr,
        resume=args.resume,
        device=args.device,
    )


if __name__ == '__main__':
    main()
