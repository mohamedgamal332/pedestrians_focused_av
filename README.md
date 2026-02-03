# Pedestrians Focused Autonomous Vehicles - GCN & RTMPose Training

This repository contains comprehensive training and evaluation code for pedestrian behavior understanding using Graph Convolutional Networks (GCN) and RTMPose pose estimation.

## Project Structure

### Core Directories

- **gcn_training/** - GCN model training scripts
  - Training, evaluation, inference, and analysis scripts
  - Support for multiple GCN architectures

- **gcn_models/** - GCN model architectures
  - CTRGCN (Channel-Time Relation Graph Convolutional Network)
  - STGCN (Spatial-Temporal Graph Convolutional Network)
  - TEGCN and SHT variants
  - Configuration and model summaries

- **mmpose_setup/** - RTMPose pose estimation setup
  - Training and evaluation scripts
  - run_train.py for orchestrated training
  - RTMPose-specific runners and evaluators

- **data_loaders/** - Data loading utilities
  - Generic dataloader for training
  - GCN-specific loader implementations

- **data_preprocessing/** - Data preparation scripts
  - COCO annotation creation
  - State vector computation
  - Risk computation from labeled data
  - Dataset split creation

- **risk_scoring/** - Risk assessment framework
  - Risk score computation
  - Risk analysis from labeled pedestrian data

- **plotting/** - Visualization and plotting utilities
  - Training history visualization
  - Metrics plotting
  - COCO graph visualization

- **training_scripts/** - Automated training execution
  - .bat files for Windows execution
  - .sh files for Unix/Linux execution
  - Finetune scripts for different model architectures

- **pretrained_models/** - Pretrained model downloads and utilities
  - Download scripts for pretrained weights
  - Pretrained model guides

## Quick Start

### 1. Environment Setup
\\\ash
# Install dependencies
pip install -r requirements-yolopose.txt

# Or use conda
conda env create -f environment-yolopose.yml
\\\

### 2. Training GCN Models

**Standard Training:**
\\\ash
python gcn_training/gcn_train.py --model stgcn --epochs 100
\\\

**Batch Training (Windows):**
\\\ash
./training_scripts/finetune_stgcn.bat
\\\

**Batch Training (Unix/Linux):**
\\\ash
bash ./training_scripts/finetune_stgcn.sh
\\\

### 3. Training RTMPose
\\\ash
python mmpose_setup/run_train.py --config configs/rtmpose_large.py
\\\

### 4. Data Preprocessing
\\\ash
# Create COCO annotations
python data_preprocessing/create_coco_annotations.py

# Compute state vectors
python data_preprocessing/compute_state_vectors.py

# Create train/test splits
python data_preprocessing/create_splits.py
\\\

### 5. Risk Scoring
\\\ash
python risk_scoring/risk_score.py --predictions predictions.npy --labels labels.npy
\\\

### 6. Visualization
\\\ash
# Plot training history
python plotting/plot_training_history.py --log training.log

# Plot metrics per epoch
python plotting/plot_workdir_metrics_per_epoch.py --workdir ./work_dirs/
\\\

## Key Features

- **Multiple GCN Architectures**: STGCN, CTRGCN, TEGCN, SHT
- **RTMPose Integration**: State-of-the-art pose estimation
- **Comprehensive Data Loading**: Support for various data formats
- **Automated Training Scripts**: Windows and Unix execution
- **Risk Assessment**: Pedestrian behavior risk scoring
- **Visualization Tools**: Training metrics and analysis plots
- **Pretrained Models**: Easy download and setup of pretrained weights

## Documentation

- [GCN Training Guide](./GCN_TRAINING_GUIDE.md) - Detailed GCN training instructions
- [GCN Augmentation Guide](./GCN_AUGMENTATION_GUIDE.md) - Data augmentation strategies
- [Finetune All Models](./FINETUNE_ALL_MODELS.md) - Multi-model finetuning workflow
- [Finetune CTRGCN](./FINETUNE_CTRGCN.md) - CTRGCN-specific finetuning
- [Batch Scripts Guide](./BATCH_SCRIPTS_GUIDE.md) - Running batch training scripts

## Model Support

### GCN Models
- STGCN (Spatial-Temporal GCN)
- CTRGCN (Channel-Time Relation GCN)
- CTRGCN Motion
- TEGCN
- SHT

### Pose Estimation
- RTMPose (Large variant)
- YOLO-Pose

## Requirements

- Python 3.8+
- PyTorch / TensorFlow
- MMPose / MMAction2
- NumPy, Pandas
- scikit-learn

See \equirements-yolopose.txt\ and \environment-yolopose.yml\ for complete dependencies.

## Authors

Created for pedestrian behavior analysis in autonomous vehicles research.

## License

Please refer to the original project license.
