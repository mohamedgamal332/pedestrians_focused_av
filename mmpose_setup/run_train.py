#!/usr/bin/env python3
"""
Direct training script for RTMPose-M on CARLA data.

This uses MMPose's Runner directly following the MMPose tools/train.py pattern.
"""

import os
import sys
from pathlib import Path

# Make sure we can import from mmpose
sys.path.insert(0, str(Path.home() / 'RTMPose' / 'mmpose'))

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
def main():
    config_path = '/home/theta/RTMPose/work_dirs/rtmpose_m_carla/rtmpose_large.py'
    work_dir = '/home/theta/RTMPose/work_dirs/rtmpose_m_carla'
    
    print("=" * 70)
    print("RTMPose-M Training on CARLA Data")
    print("=" * 70)
    
    # Load config
    print(f"\nLoading config from: {config_path}")
    cfg = Config.fromfile(config_path)
    
    # Update work directory if specified
    if work_dir is not None:
        cfg.work_dir = work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = './work_dirs/rtmpose_m_carla'
    
    print(f"Work directory: {cfg.work_dir}")
    print(f"Dataset root: {cfg.data_root}")
    print(f"Max epochs: {cfg.train_cfg['max_epochs']}")
    print(f"Batch size: {cfg.train_dataloader['batch_size']}")
    print(f"Base learning rate: {cfg.base_lr}")
    
    # Build runner from config
    print("\nBuilding runner from config...")
    runner = Runner.from_cfg(cfg)
    
    # Start training
    print("\nStarting training...")
    print("-" * 70)
    runner.train()
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()
