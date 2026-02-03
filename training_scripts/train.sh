#!/bin/bash
# Quick start script for RTMPose-M training on CARLA data

set -e  # Exit on error

echo "========================================================================"
echo "RTMPose-M Training on CARLA Data - Quick Start"
echo "========================================================================"

# Activate conda environment
echo ""
echo "Activating conda environment..."
source /home/theta/miniconda3/etc/profile.d/conda.sh
conda activate rtmpose

# Change to project directory
cd /home/theta/RTMPose

# Show dataset information
echo ""
echo "Dataset Information:"
echo "  Train:  5,751 frames (2 sessions)"
echo "  Test:   4,500 frames (1 session)"
echo "  Eval:   4,222 frames (1 session)"
echo "  Total:  14,473 frames"

# Show model information
echo ""
echo "Model Information:"
echo "  Model:      RTMPose-Medium"
echo "  Checkpoint: 53 MB (pre-downloaded)"
echo "  Input:      256×192 pixels"
echo "  Keypoints:  COCO-17"

# Show config
echo ""
echo "Training Configuration:"
grep -E "batch_size|base_lr|max_epochs" work_dirs/rtmpose_m_carla/rtmpose_m_carla.py | head -10

# Ask for confirmation
echo ""
echo "========================================================================"
read -p "Ready to start training? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting training..."
    echo "========================================================================"
    echo ""
    
    python run_train.py
    
    echo ""
    echo "========================================================================"
    echo "Training completed!"
    echo "========================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. View TensorBoard logs:"
    echo "     tensorboard --logdir ./work_dirs/rtmpose_m_carla/tf_logs"
    echo ""
    echo "  2. Evaluate on test set:"
    echo "     cd mmpose"
    echo "     python tools/test.py ../work_dirs/rtmpose_m_carla/rtmpose_m_carla.py \\"
    echo "       ../work_dirs/rtmpose_m_carla/best_coco_AP_epoch_*.pth"
else
    echo "Training cancelled."
    exit 1
fi
