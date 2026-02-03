# Setup and Deployment Guide

## Project Structure Overview

This repository is organized into functional modules for pedestrian behavior analysis:

\\\
project_root/
 gcn_training/              # GCN training and evaluation
 gcn_models/                # GCN model architectures
 mmpose_setup/              # RTMPose integration
 data_loaders/              # Data loading utilities
 data_preprocessing/        # Data preparation scripts
 risk_scoring/              # Risk assessment framework
 plotting/                  # Visualization tools
 training_scripts/          # Automated training scripts (.bat, .sh)
 pretrained_models/         # Pretrained model utilities
 README.md                  # Main documentation
 requirements-yolopose.txt # Python dependencies
 environment-yolopose.yml  # Conda environment spec
\\\

## Setup Instructions

### 1. Clone the Repository

\\\ash
git clone https://github.com/mohamedgamal332/pedestrians_focused_av.git
cd pedestrians_focused_av
\\\

### 2. Create Python Environment

#### Option A: Using pip
\\\ash
python -m venv venv_yolopose
source venv_yolopose/bin/activate  # On Windows: venv_yolopose\\Scripts\\activate
pip install -r requirements-yolopose.txt
\\\

#### Option B: Using conda
\\\ash
conda env create -f environment-yolopose.yml
conda activate yolopose
\\\

### 3. Download Pretrained Models

\\\ash
python pretrained_models/download_pretrained.py
\\\

## Module Documentation

### GCN Training (\gcn_training/\)

Core training and evaluation scripts for Graph Convolutional Networks:

- **gcn_train.py** - Main training script
- **gcn_train_all.py** - Train multiple models sequentially
- **gcn_finetune.py** - Finetune pretrained models
- **gcn_eval.py** - Evaluation on test sets
- **gcn_infer.py** - Inference on new data
- **gcn_analyze_data.py** - Data analysis and statistics
- **gcn_fetch_augment_data.py** - Data augmentation pipeline
- **test_gcn.py** - Unit tests

**Quick Start:**
\\\ash
cd gcn_training
python gcn_train.py --model stgcn --epochs 100 --batch-size 32
\\\

### GCN Models (\gcn_models/\)

GCN architecture implementations:

- **stgcn.py** - Spatial-Temporal GCN
- **ctrgcn.py** - Channel-Time Relation GCN
- **ctrgcn_motion.py** - CTRGCN variant with motion features
- **tegcn.py** - Temporal Enhanced GCN
- **sht.py** - Spatial-Heading-Temporal model
- **stgcn_config.py** - Configuration utilities

### Data Loaders (\data_loaders/\)

Data loading and preprocessing:

- **dataloader.py** - Generic data loader
- **gcn_loader.py** - GCN-specific loader with augmentation

### Data Preprocessing (\data_preprocessing/\)

Data preparation utilities:

- **create_coco_annotations.py** - Convert to COCO format
- **create_splits.py** - Train/val/test splits
- **compute_state_vectors.py** - Compute pedestrian state vectors
- **compute_risks_from_labeled.py** - Risk computation

### Risk Scoring (\isk_scoring/\)

Pedestrian behavior risk assessment:

- **risk_score.py** - Main risk scoring script
- Supports various risk metrics and pedestrian behaviors

### MMPose Setup (\mmpose_setup/\)

RTMPose pose estimation integration:

- **run_train.py** - Orchestrated training
- **train_rtmpose.py** - RTMPose training script
- **rtmpose_run.py** - Inference runner
- **rtmpose_eval.py** - Evaluation script

### Plotting (\plotting/\)

Visualization and analysis:

- **plot_training_history.py** - Training loss/accuracy curves
- **plot_training_log.py** - Log file visualization
- **plot_workdir_metrics_per_epoch.py** - Per-epoch metrics
- **coco_graph.py** - COCO dataset visualization

### Training Scripts (\	raining_scripts/\)

Automated execution scripts:

**Windows (.bat files):**
- finetune_stgcn.bat
- finetune_ctrgcn.bat
- finetune_ctrgcn_motion.bat
- finetune_tegcn.bat
- finetune_sht.bat

**Unix/Linux (.sh files):**
- finetune_stgcn.sh
- finetune_ctrgcn.sh
- finetune_ctrgcn_motion.sh
- finetune_tegcn.sh
- finetune_sht.sh
- train.sh

## Training Workflows

### Workflow 1: Basic GCN Training

\\\ash
python gcn_training/gcn_train.py \
    --model stgcn \
    --dataset ntu60 \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001
\\\

### Workflow 2: Run All Training Scripts (Windows)

\\\ash
cd training_scripts
finetune_stgcn.bat
finetune_ctrgcn.bat
finetune_tegcn.bat
\\\

### Workflow 3: Run All Training Scripts (Unix/Linux)

\\\ash
cd training_scripts
bash finetune_stgcn.sh
bash finetune_ctrgcn.sh
bash finetune_tegcn.sh
\\\

### Workflow 4: Data Preprocessing Pipeline

\\\ash
# 1. Create COCO annotations
python data_preprocessing/create_coco_annotations.py

# 2. Create train/val/test splits
python data_preprocessing/create_splits.py

# 3. Compute state vectors
python data_preprocessing/compute_state_vectors.py

# 4. Compute risk scores
python data_preprocessing/compute_risks_from_labeled.py
\\\

### Workflow 5: Evaluation and Visualization

\\\ash
# Evaluate trained models
python gcn_training/gcn_eval.py --checkpoint work_dirs/stgcn_best.pth

# Plot training history
python plotting/plot_training_history.py --log work_dirs/training.log

# Plot per-epoch metrics
python plotting/plot_workdir_metrics_per_epoch.py --workdir work_dirs/
\\\

## File Organization for GitHub

The \project_root/\ directory is ready to be pushed to GitHub. To initialize Git:

\\\ash
cd project_root
git init
git add .
git commit -m "Initial commit: GCN and RTMPose training framework"
git remote add origin https://github.com/mohamedgamal332/pedestrians_focused_av.git
git branch -M main
git push -u origin main
\\\

## Data Structure

The project expects data to be organized as:

\\\
data/
 ntu60/
    train/
    val/
    test/
 annotations/
     coco_train.json
     coco_val.json
     coco_test.json
\\\

## Model Outputs

Training generates:

\\\
work_dirs/
 stgcn_kinetics/
    best_epoch.pth
    latest.pth
    training.log
    metrics.csv
 ctrgcn/
 [other models...]
\\\

## Important Notes

1. **GPU Support**: Most scripts support CUDA. Set CUDA_VISIBLE_DEVICES if needed
2. **Batch Scripts**: Windows .bat scripts need to be run from Command Prompt or PowerShell
3. **Pretrained Models**: Download pretrained weights first using download_pretrained.py
4. **Data Format**: Ensure data follows COCO JSON format for compatibility
5. **Dependencies**: Install all requirements before running training

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in training scripts
- Use gradient accumulation
- Enable mixed precision training

### Data Loading Issues
- Verify data paths in configuration files
- Check COCO JSON format validation
- Ensure image paths are correct

### Model Loading Errors
- Download pretrained weights
- Verify PyTorch version compatibility
- Check checkpoint file integrity

## Support and Documentation

- See [GCN Training Guide](./GCN_TRAINING_GUIDE.md) for detailed GCN training
- See [Finetune All Models](./FINETUNE_ALL_MODELS.md) for multi-model training
- See [Batch Scripts Guide](./BATCH_SCRIPTS_GUIDE.md) for script execution

