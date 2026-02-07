# Pedestrian-Focused Autonomous Vehicle Research Platform

A comprehensive research platform for developing and evaluating pedestrian-aware autonomous vehicle systems. This project integrates simulation, pose estimation, trajectory prediction, and autonomous driving control with a focus on pedestrian safety and behavior understanding.

## ⚡ Quick Start

Choose your component of interest:

1. **Just exploring?** → Check out the [example frames](#) and [repository structure](#-repository-structure)
2. **Want to collect data?** → Start with [CARLA Simulation](#1-collect-training-data-with-carla)
3. **Working on pose estimation?** → See [MMPose Library](#2-run-pose-estimation)
4. **Need full autonomous control?** → Set up [Governor-Reflex System](#3-governor-reflex-system)

**⚠️ Important for Governor-Reflex users**: This component requires two external repositories:
- [Alpamayo](https://github.com/NVlabs/alpamayo) - Vision-language planning model
- [PCLA](https://github.com/MasoudJTehrani/PCLA) - Low-level control system

## 🎯 Project Overview

This repository contains a full pipeline for pedestrian-focused autonomous vehicle research:

1. **Data Collection**: Realistic pedestrian behavior simulation in CARLA with stereo vision
2. **Pose Estimation**: Multi-person pose estimation and tracking using MMPose
3. **Trajectory Prediction**: Deep learning models for predicting pedestrian movements
4. **Matching & Triangulation**: Multi-camera pedestrian matching and 3D localization
5. **Autonomous Control**: Governor-Reflex architecture for pedestrian-aware driving

## 📁 Repository Structure

```
pedestrians_focused_av/
├── CarlaSimulation/              # CARLA-based data collection system
│   ├── carla_data_collector.py   # Main data collection script
│   ├── config.yaml               # Simulation configuration
│   └── README.md                 # Detailed CARLA setup guide
│
├── MMPose-Lib-staging/           # Pose estimation and trajectory prediction
│   ├── models/                   # Neural network architectures
│   │   ├── stgcn.py             # Spatial-Temporal Graph Convolutional Network
│   │   ├── ctrgcn.py            # Channel-wise Topology Refinement GCN
│   │   └── sht.py               # Spatial Hierarchical Transformer
│   ├── scripts/                  # Training and evaluation scripts
│   ├── data/                     # Dataset utilities
│   └── save_mmpose_results.py    # Pose estimation inference
│
├── gcn_models/                   # GCN Model Architectures (NEW)
│   ├── stgcn.py                 # Spatial-Temporal GCN implementation
│   ├── ctrgcn.py                # Channel-Time Relation GCN
│   ├── ctrgcn_motion.py         # CTR-GCN with motion features
│   ├── tegcn.py                 # Temporal Enhanced GCN
│   ├── sht.py                   # Spatial-Hierarchical Transformer
│   ├── README.md                # Model documentation
│   └── MODEL_SUMMARY.md         # Model specifications
│
├── gcn_training/                 # GCN Training & Evaluation (NEW)
│   ├── gcn_train.py             # Main training script
│   ├── gcn_train_all.py         # Multi-model training
│   ├── gcn_finetune.py          # Fine-tuning script
│   ├── gcn_eval.py              # Evaluation script
│   ├── gcn_infer.py             # Inference script
│   ├── gcn_analyze_data.py      # Data analysis utilities
│   └── gcn_fetch_augment_data.py # Data augmentation
│
├── training_scripts/             # Automated Training Scripts (NEW)
│   ├── finetune_stgcn.bat/sh    # ST-GCN fine-tuning
│   ├── finetune_ctrgcn.bat/sh   # CTR-GCN fine-tuning
│   ├── finetune_tegcn.bat/sh    # TE-GCN fine-tuning
│   └── finetune_sht.bat/sh      # SHT fine-tuning
│
├── Code+Images+Vision/           # Trajectory Prediction Evaluations (NEW)
│   ├── Evaluations/             # Evaluation results and analysis
│   │   ├── docs/                # Analysis reports
│   │   └── *.png                # Visualization plots
│   ├── Outputs/                 # Model outputs and checkpoints
│   │   ├── gru/                 # GRU model results
│   │   ├── lstm/                # LSTM model results
│   │   ├── mamba/               # Mamba model results
│   │   ├── kalman/              # Kalman filter results
│   │   └── mlp/                 # MLP baseline results
│   ├── PBPL.ipynb               # Pedestrian-based position localization
│   └── Requirements and baseline results.md
│
├── Matching and Triangulation/   # Multi-camera matching and 3D reconstruction
│   └── Matcher-temporal.ipynb    # GNN-based temporal matcher
│
├── governor_reflex/              # Autonomous driving control system
│   ├── governor/                 # High-level trajectory planning
│   │   ├── main_governor.py     # Alpamayo-based planner
│   │   └── alpamayo_wrapper.py  # Model wrapper
│   ├── reflex/                   # Low-level reactive control
│   │   ├── main_reflex.py       # CaRL control system
│   │   └── pedestrian_tracker.py # Real-time tracking
│   └── utils/                    # Shared utilities
│
├── data_loaders/                 # Data Loading Utilities
│   ├── dataloader.py            # Generic data loader
│   └── gcn_loader.py            # GCN-specific loader
│
├── data_preprocessing/           # Data Preparation Scripts
│   ├── create_coco_annotations.py
│   ├── create_splits.py
│   ├── compute_state_vectors.py
│   └── compute_risks_from_labeled.py
│
├── risk_scoring/                 # Risk Assessment Framework
│   ├── risk_score.py
│   └── compute_risks_from_labeled.py
│
├── plotting/                     # Visualization Tools
│   ├── plot_training_history.py
│   ├── plot_training_log.py
│   └── coco_graph.py
│
├── pretrained_models/            # Pretrained Model Management
│   ├── download_pretrained.py
│   └── PRETRAINED_WEIGHTS_GUIDE.md
│
├── mmpose_setup/                 # RTMPose Integration
│   ├── train_rtmpose.py
│   ├── rtmpose_run.py
│   └── rtmpose_eval.py
│
├── pbplocalization.ipynb         # Pedestrian-based position localization
├── evaluate_and_visualize.py     # Comprehensive evaluation script
│
└── Documentation Files:
    ├── README.md                 # This file
    ├── PROJECT_STRUCTURE.md      # Detailed structure documentation
    ├── SETUP_AND_DEPLOYMENT.md   # Setup instructions
    ├── GITHUB_DEPLOYMENT.md      # Git deployment guide
    ├── GCN_TRAINING_GUIDE.md     # GCN training guide
    ├── GCN_AUGMENTATION_GUIDE.md # Data augmentation guide
    ├── FINETUNE_ALL_MODELS.md    # Multi-model fine-tuning
    ├── FINETUNE_CTRGCN.md        # CTR-GCN specific guide
    └── BATCH_SCRIPTS_GUIDE.md    # Batch scripts usage guide
```

## 🚀 Key Features

### CARLA Simulation System
- **Stereo Vision Setup**: Synchronized left/right RGB + depth cameras
- **Realistic Pedestrian Behaviors**: Walking, running, crossing, waiting, idle states
- **Jaywalking Simulation**: Traffic-aware crossing decisions with safety checks
- **Occlusion Detection**: Per-bone visibility analysis using depth buffers
- **Rich Annotations**: 3D skeleton data, behavior labels, and visibility states
- **Robust Connection Handling**: Automatic retry logic for CARLA initialization

### GCN-Based Trajectory Prediction (NEW)
- **Multiple GCN Architectures**:
  - **ST-GCN**: Spatial-Temporal Graph Convolutional Networks
  - **CTR-GCN**: Channel-wise Topology Refinement GCN (60.67% baseline accuracy)
  - **TE-GCN**: Temporal Enhanced GCN (custom architecture)
  - **SHT**: Spatial-Hierarchical Transformer (state-of-the-art)
- **Automated Training Pipeline**: Batch scripts for Windows and Linux
- **Data Augmentation**: Synthetic sample generation for class imbalance
- **Fine-tuning Support**: Transfer learning from pretrained models
- **Comprehensive Evaluation**: Per-class metrics, confusion matrices, F1 scores

### Trajectory Prediction with Sequence Models (NEW)
- **Multiple Architectures Evaluated**:
  - **Mamba**: Best performance (1.0113m error, 21.8% improvement)
  - **GRU**: Strong baseline (1.0244m error, 20.8% improvement)
  - **LSTM**: Classic approach (1.0303m error, 20.3% improvement)
  - **Kalman Filter**: Lightweight (1.1439m error, 11.5% improvement)
  - **MLP**: Simple baseline (1.0708m error, 13.3% improvement)
- **Multi-Horizon Prediction**: Up to 30 frames ahead (1 second @ 30fps)
- **Comprehensive Metrics**: Position, per-keypoint, bone length, correction analysis
- **Pareto Analysis**: Accuracy vs recall tradeoff optimization
- **Checkpoint System**: Progressive training with epoch-wise model saving

### Pose Estimation & Risk Assessment
- **Real-time Inference**: Integration with MMPose for pose estimation
- **Risk Scoring**: Compute risk scores from pedestrian poses and trajectories
- **Behavior Classification**: 5 classes (Walking, Running, Crossing, Waiting, Idle)

### Multi-Camera Matching
- **Temporal Consistency**: Graph Neural Network-based matching across frames
- **Hungarian Algorithm**: Optimal assignment for multi-person tracking
- **3D Triangulation**: Reconstruct pedestrian positions from stereo views

### Governor-Reflex Control Architecture
- **Governor (Planning)**: Long-term trajectory planning using Alpamayo vision-language model from NVIDIA
  - Processes multi-camera inputs (front wide, front tele, cross left, cross right)
  - Generates 64 waypoints at 10Hz (6.4 second horizon)
  - Incorporates pedestrian behavior and risk assessment
  - Runs in separate conda environment with Alpamayo model
- **Reflex (Control)**: Low-level reactive control with CaRL/PCLA
  - Real-time pedestrian tracking and trajectory management
  - CARLA simulation integration with 4-camera setup
  - Egomotion buffer for vehicle state history
  - Route injection into PCLA control system
- **Inter-Process Communication**: File-based communication between Governor and Reflex
- **Pedestrian Integration**: Real-time pedestrian tracking and trajectory adjustment
- **A/B Testing Support**: Toggle pedestrian information for ablation studies
- **Mock Mode**: Built-in mock Alpamayo wrapper for testing without full model

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CARLA Simulator 0.9.15
- CUDA-capable GPU (recommended)
- Conda/Mamba package manager
- Git (for cloning external repositories)

### Component-Specific Setup

#### 1. CARLA Data Collection
```bash
cd CarlaSimulation
pip install -r requirements.txt

# Install CARLA Python API
pip install /path/to/carla/PythonAPI/carla/dist/carla-0.9.15-*.egg
```

#### 2. MMPose Library
```bash
cd MMPose-Lib-staging
pip install -r requirements.txt
pip install mmpose mmdet mmengine
```

#### 3. Governor-Reflex System

**⚠️ Important**: The Governor-Reflex system requires two external repositories to be installed:

##### Required External Repositories

1. **Alpamayo** (NVIDIA's vision-language planning model)
   ```bash
   # Clone Alpamayo repository
   git clone https://github.com/NVlabs/alpamayo.git
   cd alpamayo
   # Follow installation instructions in the Alpamayo repository
   ```

2. **PCLA** (Predictive Control with Learned Action Priors)
   ```bash
   # Clone PCLA repository
   git clone https://github.com/MasoudJTehrani/PCLA.git
   cd PCLA
   # Follow installation instructions in the PCLA repository
   ```

##### Setup Instructions

```bash
# Create separate conda environments for Governor and Reflex

# 1. Governor Environment (Alpamayo-based planning)
conda create -n alpo python=3.8
conda activate alpo
cd /path/to/alpamayo
# Install Alpamayo dependencies (follow their README)
pip install torch torchvision
# Install additional governor dependencies
pip install pyyaml numpy

# 2. Reflex Environment (CaRL/PCLA control)
conda create -n PCLA python=3.8
conda activate PCLA
cd /path/to/PCLA
# Install PCLA dependencies (follow their README)
pip install carla pyyaml numpy

# 3. Update configuration paths
cd /path/to/pedestrians_focused_av/governor_reflex
# Edit config.yaml to set:
#   - paths.model_path: /path/to/alpamayo/models
#   - paths.pcla_dir: /path/to/PCLA
```

**Configuration Notes**:
- Update `governor_reflex/config.yaml` with correct paths to Alpamayo models and PCLA directory
- The Governor process runs in the `alpo` conda environment
- The Reflex process runs in the `PCLA` conda environment
- Both processes communicate through shared runtime files

## 📖 Usage

### 1. Collect Training Data with CARLA

```bash
cd CarlaSimulation

# Option 1: Manual CARLA start
./CarlaUE4.sh -RenderOffScreen -nosound &
sleep 60
./run_collection.sh 300

# Option 2: Auto-start script
export CARLA_ROOT=/path/to/carla
./start_and_collect.sh 300
```

**Output**: 
- Stereo RGB images (left/right)
- Depth maps (16-bit PNG)
- Per-frame JSON annotations with skeleton data
- Behavior labels and visibility states

### 2. Run Pose Estimation

```bash
cd MMPose-Lib-staging
python save_mmpose_results.py
```

### 3. Match and Triangulate

Open and run:
```bash
jupyter notebook "Matching and Triangulation/Matcher-temporal.ipynb"
```

### 4. Train GCN Models for Behavior Classification (NEW)

#### Quick Start with Batch Scripts

**Windows:**
```bash
cd training_scripts
finetune_stgcn.bat
finetune_ctrgcn.bat
finetune_tegcn.bat
finetune_sht.bat
```

**Linux/Mac:**
```bash
cd training_scripts
chmod +x finetune_*.sh
./finetune_stgcn.sh
./finetune_ctrgcn.sh
./finetune_tegcn.sh
./finetune_sht.sh
```

#### Manual Training

```bash
# Train individual models
python gcn_training/gcn_train.py --model stgcn --epochs 50 --batch_size 32
python gcn_training/gcn_train.py --model ctrgcn --epochs 50 --batch_size 32

# Train all models
python gcn_training/gcn_train_all.py --epochs 50

# Generate augmented data for class imbalance
python gcn_training/gcn_fetch_augment_data.py --synthetic_samples 2000

# Fine-tune with augmented data
python gcn_training/gcn_finetune.py \
    --model ctrgcn \
    --pretrained_path work_dirs/gcn_training/ctrgcn/best_model.pth \
    --augmented_data_dir ./gcn_per_pedestrian_augmented \
    --epochs 30

# Evaluate models
python gcn_training/gcn_eval.py --model ctrgcn \
    --checkpoint work_dirs/gcn_training/ctrgcn/best_model.pth
```

See detailed guides:
- [GCN Training Guide](./GCN_TRAINING_GUIDE.md)
- [Data Augmentation Guide](./GCN_AUGMENTATION_GUIDE.md)
- [Fine-tune All Models](./FINETUNE_ALL_MODELS.md)
- [Batch Scripts Guide](./BATCH_SCRIPTS_GUIDE.md)

### 5. Train Trajectory Prediction Models (NEW)

```bash
# Train LSTM-based trajectory predictor
cd MMPose-Lib-staging/scripts
python train_lstm_trajectory.py

# Train Mamba-based trajectory predictor
python train_mamba_trajectory.py

# Train Transformer-based trajectory predictor
python train_transformer_trajectory.py

# Evaluate and visualize all models
cd ../..
python evaluate_and_visualize.py
```

**Evaluation Results** (see `Code+Images+Vision/Evaluations/`):
- **Best Model**: Mamba (1.0113m error, 21.8% improvement over baseline)
- **Baseline Performance**: 1.2932m mean error
- **Pareto Operating Points**:
  - High accuracy: 0.8667m @ 74% recall
  - High recall: 1.2596m @ 100% recall

### 6. Run Governor-Reflex System

**Prerequisites**: 
- CARLA simulator must be running
- Both Alpamayo and PCLA repositories must be installed
- Configuration file must be updated with correct paths

```bash
cd governor_reflex

# First, start CARLA in a separate terminal
cd /path/to/carla
./CarlaUE4.sh -RenderOffScreen -nosound

# Terminal 1: Start Reflex (reactive control with CaRL/PCLA)
conda activate PCLA
export PYTHONPATH=$PYTHONPATH:/path/to/PCLA
python reflex/main_reflex.py

# Terminal 2: Start Governor (high-level planning with Alpamayo)
conda activate alpo
export PYTHONPATH=$PYTHONPATH:/path/to/alpamayo
python governor/main_governor.py

# Or use the provided shell scripts
./run_reflex.sh    # Starts the Reflex process
./run_governor.sh  # Starts the Governor process
```

**How it works**:
1. **Reflex** manages CARLA simulation, captures camera images, and tracks pedestrians
2. **Reflex** periodically requests trajectory plans from Governor via shared files
3. **Governor** processes camera images and pedestrian data through Alpamayo model
4. **Governor** generates trajectory waypoints and returns them to Reflex
5. **Reflex** injects trajectories into CaRL/PCLA for low-level control execution

### 6. Pedestrian-Based Localization (PBPL)

Explore the PBPL (Pedestrian-Based Position Localization) system for trajectory prediction and correction:
```bash
# Run comprehensive evaluation
python evaluate_and_visualize.py

# Or explore interactively
jupyter notebook pbplocalization.ipynb
jupyter notebook Code+Images+Vision/PBPL.ipynb
```

The PBPL system uses LSTM/GRU/Mamba models to predict future pedestrian trajectories and correct triangulation errors. See [Requirements and baseline results](./Code+Images+Vision/Requirements%20and%20baseline%20results.md) for detailed objectives and baseline metrics.

### 7. Run Governor-Reflex System

## 📊 Data Format

### CARLA Annotations
Each frame includes:
```json
{
  "frame_id": 12345,
  "timestamp": 123.456,
  "pedestrians": [
    {
      "id": 200,
      "behavior": "waiting_to_cross",
      "speed": 0.0,
      "jaywalking": {
        "is_jaywalker": true,
        "jaywalking_state": "waiting",
        "is_safe_to_cross": false
      },
      "skeleton": {
        "crl_Head__C": {
          "world": {"location": {...}},
          "cameras": {
            "left": {
              "pixel": [960, 540],
              "bone_depth": 15.5,
              "visibility_state": "visible"
            }
          }
        }
      }
    }
  ]
}
```

### GCN Training Data Format (NEW)

**Input Shape**: `[N, C, T, V]`
- **N**: Number of samples
- **C**: Channels (2 for 2D coordinates, 3 for 3D)
- **T**: Temporal frames (default: 30 frames @ 30fps = 1 second)
- **V**: Vertices/Keypoints (17 for COCO skeleton)

**Labels**: Integer class indices [0-4]
- 0: Walking
- 1: Running  
- 2: Crossing
- 3: Waiting_To_Cross
- 4: Idle

**Data Files**:
- `gcn_per_pedestrian/data.npy` - Input skeleton sequences
- `gcn_per_pedestrian/labels.npy` - Behavior class labels
- `gcn_per_pedestrian_augmented/` - Augmented dataset with synthetic samples

### Trajectory Prediction Output Format (NEW)

Model checkpoints and training outputs are saved in structured directories:

```
Code+Images+Vision/Outputs/
├── gru/
│   ├── checkpoints/          # Epoch-wise model checkpoints
│   │   ├── epoch_010.pt
│   │   └── ...
│   ├── gru_history.json      # Training history
│   ├── gru_staged_model.pt   # Best model
│   ├── checkpoint_summary.json
│   └── plots/                # Training visualizations
├── lstm/
├── mamba/
├── kalman/
└── mlp/
```

**Evaluation Metrics** (see `Code+Images+Vision/Evaluations/`):
- Mean/median position error
- Per-keypoint accuracy
- Bone length preservation
- Direction and speed errors
- Confidence vs error correlation
- Pareto frontier analysis

### Pedestrian Behaviors
| Behavior | Description | Speed Range |
|----------|-------------|-------------|
| WALKING | Normal walking pace | 0.8-1.8 m/s |
| RUNNING | Fast movement | 2.5-4.5 m/s |
| WAITING_TO_CROSS | Stopped at road edge | ~0 m/s |
| CROSSING | Actively crossing road | 1.0-2.3 m/s |
| IDLE | Stationary | <0.1 m/s |

### Visibility States
- **VISIBLE**: Bone in frame and not occluded
- **OCCLUDED**: Bone in frame but blocked
- **OUT_OF_FRAME**: Bone projects outside image
- **BEHIND_CAMERA**: Bone is behind camera plane

## 📈 Evaluation Results (NEW)

### Trajectory Prediction Performance

Comprehensive evaluation of sequence-based models for pedestrian trajectory prediction:

| Model | Mean Error | Improvement | Parameters | Best For |
|-------|------------|-------------|------------|----------|
| **Mamba** | 1.0113m | 21.8% | 502,468 | Best accuracy |
| **GRU** | 1.0244m | 20.8% | 448,196 | Speed/accuracy balance |
| **LSTM** | 1.0303m | 20.3% | 316,100 | Reliable baseline |
| **Kalman** | 1.1439m | 11.5% | 35,174 | Resource-constrained |
| **MLP** | 1.0708m | 13.3% | 85,700 | Simple baseline |
| **Baseline** | 1.2932m | - | - | Triangulation only |

**Key Findings**:
- All learning-based models outperform baseline triangulation
- Mamba architecture achieves best performance with 21.8% error reduction
- Lightweight Kalman filter still provides 11.5% improvement
- Pareto-optimal operating points available for accuracy/recall tradeoff

### GCN Behavior Classification

Performance of GCN models on pedestrian behavior recognition:

| Model | Before Fine-tuning | After Fine-tuning | Improvement |
|-------|-------------------|-------------------|-------------|
| **ST-GCN** | 18.07% | 70-80% | ~60% points |
| **CTR-GCN** | 60.67% | 75-85% | ~20% points |
| **TE-GCN** | 0.00% | 50-70% | ~60% points |
| **SHT** | N/A | 80-90%* | - |

*With Hyperformer pretrained weights

**Class Distribution Challenge**:
- Original dataset heavily imbalanced (60% Walking, 0% Waiting_To_Cross)
- Data augmentation with 2000 synthetic samples addresses imbalance
- Fine-tuning with augmented data improves minority class recognition

### Pareto Operating Points

Two optimal configurations for different deployment scenarios:

**High Accuracy Mode**:
- Confidence threshold: 0.3
- Min keypoints: 12
- Recall: 74.0%
- Mean error: 0.8667m
- Use case: Safety-critical applications

**High Recall Mode**:
- Confidence threshold: 0.3
- Min keypoints: 4
- Recall: 100.0%
- Mean error: 1.2596m
- Use case: Must-detect scenarios

### Visualizations

See `Code+Images+Vision/Evaluations/` for comprehensive visualizations:
- Loss component analysis per architecture
- Horizon-based error progression
- Threshold analysis for Pareto optimization
- Per-joint error heatmaps
- Direction and speed error distributions
- Training history and convergence plots

## 🔬 Research Applications

This platform supports research in:

- **Pedestrian Behavior Modeling**: Realistic jaywalking and crossing behaviors with 5-class classification
- **Graph-Based Action Recognition**: Multiple GCN architectures for skeleton-based behavior analysis
- **Data Augmentation Techniques**: Synthetic sample generation to address class imbalance
- **Pose-based Trajectory Prediction**: Multi-horizon forecasting using sequence models (LSTM, GRU, Mamba)
- **Trajectory Correction**: Blending predicted and detected positions to minimize triangulation errors
- **Multi-Modal Perception**: Combining RGB, depth, and skeleton data
- **Occlusion Handling**: Robust tracking under partial visibility
- **Safety-Critical Scenarios**: Testing AV responses to unpredictable pedestrians
- **Ablation Studies**: Evaluating impact of pedestrian information on AV performance
- **Pareto Optimization**: Balancing accuracy and recall for different deployment scenarios
- **Benchmark Comparisons**: State space models vs RNNs vs classical filters for trajectory prediction

## 🎓 Model Architectures

### GCN Models for Behavior Classification (NEW)

The repository includes multiple GCN architectures for pedestrian behavior recognition:

1. **ST-GCN (Spatial-Temporal GCN)**
   - Graph convolutions on skeleton topology
   - Temporal convolutions across frames
   - Efficient and fast inference
   - Can use MMACTION2 pretrained weights
   - Expected accuracy: 60-80% (with pretraining)

2. **CTR-GCN (Channel-wise Topology Refinement)**
   - Learnable graph topology
   - Channel-wise feature refinement
   - State-of-the-art on action recognition
   - Baseline: 60.67% → Fine-tuned: 75-85%

3. **TE-GCN (Temporal Enhanced GCN)**
   - Custom temporal enhancement mechanisms
   - Taylor expansion for temporal modeling
   - Expected accuracy: 50-75%

4. **SHT (Spatial Hierarchical Transformer)**
   - Hierarchical attention mechanism
   - Multi-scale spatial features
   - Long-range temporal dependencies
   - Expected accuracy: 80-90% (with pretraining)

**Behavior Classes**: Walking, Running, Crossing, Waiting_To_Cross, Idle

See [gcn_models/README.md](./gcn_models/README.md) and [gcn_models/MODEL_SUMMARY.md](./gcn_models/MODEL_SUMMARY.md) for detailed specifications.

### Trajectory Prediction Models (NEW)

Sequence-based models for multi-horizon pedestrian trajectory forecasting:

1. **Mamba-based Model** ★ Best Performance
   - State space model architecture
   - Mean error: 1.0113m (21.8% improvement over baseline)
   - 502,468 parameters
   - Best for accuracy-critical applications

2. **GRU (Gated Recurrent Unit)**
   - Lightweight recurrent architecture
   - Mean error: 1.0244m (20.8% improvement)
   - 448,196 parameters
   - Good balance of speed and accuracy

3. **LSTM (Long Short-Term Memory)**
   - Classic sequence modeling
   - Mean error: 1.0303m (20.3% improvement)
   - 316,100 parameters
   - Reliable and well-tested

4. **Kalman Filter**
   - Traditional filtering approach
   - Mean error: 1.1439m (11.5% improvement)
   - 35,174 parameters (most lightweight)
   - Best for resource-constrained deployments

5. **MLP Baseline**
   - Simple feedforward network
   - Mean error: 1.0708m (13.3% improvement)
   - 85,700 parameters
   - Baseline for comparison

**Key Features**:
- Multi-horizon prediction (1-30 frames, ~1 second)
- Position, keypoint, and bone length correction
- Confidence-based blending of predictions and detections
- Pareto-optimal operating points for accuracy/recall tradeoff

See [Code+Images+Vision/Evaluations/docs/analysis_report.md](./Code+Images+Vision/Evaluations/docs/analysis_report.md) for comprehensive evaluation results.

## ⚙️ Configuration

### CARLA Simulation (`CarlaSimulation/config.yaml`)
```yaml
simulation:
  map: "Town10HD_Opt"
  duration_seconds: 300
  tick_rate: 15

traffic:
  num_vehicles: 25
  num_pedestrians: 100
  runner_ratio: 0.25

jaywalking:
  enabled: true
  jaywalker_ratio: 0.3
  safety_time_threshold: 4.0
```

### Governor-Reflex (`governor_reflex/config.yaml`)
```yaml
# Important: Update these paths after installing external dependencies
paths:
  model_path: "/path/to/alpamayo/models"
  runtime_dir: "/path/to/runtime"
  pcla_dir: "/path/to/PCLA"

carla:
  host: "localhost"
  port: 2000
  map: "Town02"

# Camera configuration for Alpamayo (4-camera setup)
cameras:
  front_wide:
    location: [2.2, 0.0, 1.5]
    fov: 120
  front_tele:
    fov: 50
  cross_left:
    rotation: [0.0, -60.0, 0.0]
  cross_right:
    rotation: [0.0, 60.0, 0.0]

trajectory:
  num_waypoints: 64
  dt: 0.1
  horizon_seconds: 6.4

experiment:
  include_pedestrian_info: true
  mask_pedestrians_in_images: false
```

## 🐛 Troubleshooting

### CARLA Connection Issues
```bash
# Test CARLA connection
cd CarlaSimulation
python test_connection.py --wait --max-wait 120

# Check if CARLA is running
ps aux | grep CarlaUE4
netstat -tlnp | grep 2000
```

### GCN Training Issues (NEW)

**Low accuracy after training**:
```bash
# Check data quality
python gcn_training/gcn_analyze_data.py --data_dir ./gcn_per_pedestrian

# Check for class imbalance - if severe:
python gcn_training/gcn_fetch_augment_data.py --synthetic_samples 2000
python gcn_training/gcn_finetune.py --model ctrgcn --augmented_data_dir ./gcn_per_pedestrian_augmented
```

**CUDA out of memory**:
- Reduce batch size: `--batch_size 16` or `--batch_size 8`
- Use CPU: `--device cpu`
- Use gradient accumulation

**Model not converging**:
- Lower learning rate: `--lr 0.0001`
- Increase epochs: `--epochs 100`
- Use pretrained weights (ST-GCN): `--pretrained_path pretrained_weights/stgcn/...`

**"Waiting_To_Cross" class has 0% accuracy**:
- This is expected if dataset lacks this class
- Generate augmented data: `python gcn_training/gcn_fetch_augment_data.py`
- Fine-tune on augmented dataset

**Batch scripts not running**:
- Windows: Run from Command Prompt, not PowerShell
- Linux: Make executable first: `chmod +x finetune_*.sh`

### Trajectory Prediction Issues (NEW)

**High prediction errors**:
- Check confidence threshold settings
- Verify input data quality (triangulation accuracy)
- Ensure sufficient training data (>1000 sequences)
- Try different model architectures (Mamba typically best)

**Training too slow**:
- Reduce sequence length in config
- Use smaller model (MLP or Kalman)
- Enable mixed precision training
- Reduce checkpoint frequency

**Checkpoints not loading**:
- Verify checkpoint path is correct
- Check PyTorch version compatibility
- Ensure model architecture matches checkpoint

### GPU Memory Issues
- Use lower quality CARLA maps (Town03 instead of Town10HD_Opt)
- Reduce number of pedestrians/vehicles
- Use `-quality-level=low` flag for CARLA
- For GCN training: reduce batch size or use gradient checkpointing

### Import Errors

**CARLA Python API**:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/carla/PythonAPI/carla/dist/carla-0.9.15-*.egg
```

**Alpamayo Module**:
```bash
# Make sure Alpamayo is installed and in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/alpamayo
# Test import
python -c "import alpamayo_r1; print('Alpamayo OK')"
```

**PCLA Module**:
```bash
# Make sure PCLA is installed and in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/PCLA
# Test import
python -c "from PCLA import PCLA; print('PCLA OK')"
```

### Governor-Reflex Issues

**Error: "Cannot find alpamayo models"**
- Update `paths.model_path` in `governor_reflex/config.yaml`
- Ensure Alpamayo repository is properly installed
- Use mock mode for testing: set `alpamayo.use_mock: true` in config

**Error: "No PCLA instance set"**
- Ensure PCLA repository is installed and in PYTHONPATH
- Update `paths.pcla_dir` in `governor_reflex/config.yaml`
- Check that Reflex process started successfully before Governor

## 📚 Dependencies

### Core Dependencies
- **CARLA** 0.9.15: Simulation environment
- **PyTorch**: Deep learning framework
- **MMPose**: Pose estimation library
- **MMDetection**: Object detection
- **PyTorch Geometric**: Graph neural networks
- **OpenCV**: Image processing
- **NumPy, Pandas**: Data manipulation

### External Repositories (Required for Governor-Reflex)

**⚠️ These must be installed separately**:

1. **Alpamayo** - NVIDIA's vision-language planning model
   - Repository: https://github.com/NVlabs/alpamayo
   - Used by: Governor process for trajectory planning
   - Environment: `alpo` conda environment

2. **PCLA** - Predictive Control with Learned Action Priors (CaRL implementation)
   - Repository: https://github.com/MasoudJTehrani/PCLA
   - Used by: Reflex process for low-level control
   - Environment: `PCLA` conda environment

### Component-Specific Dependencies

See individual `requirements.txt` files:
- `CarlaSimulation/requirements.txt` - CARLA data collection
- `MMPose-Lib-staging/requirements.txt` - Pose estimation and trajectory models
- `requirements-yolopose.txt` - Main project dependencies
- `environment-yolopose.yml` - Conda environment specification

**Note**: Governor-Reflex does not have a requirements.txt as it depends on the external Alpamayo and PCLA repositories, which have their own dependency management.

## 📖 Documentation

This repository includes comprehensive documentation to help you get started:

### Setup & Deployment
- **[SETUP_AND_DEPLOYMENT.md](./SETUP_AND_DEPLOYMENT.md)** - Complete setup instructions for all components
- **[GITHUB_DEPLOYMENT.md](./GITHUB_DEPLOYMENT.md)** - Guide for pushing code to GitHub
- **[PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md)** - Detailed directory structure and file organization

### GCN Training & Evaluation
- **[GCN_TRAINING_GUIDE.md](./GCN_TRAINING_GUIDE.md)** - Complete guide for training GCN models
- **[GCN_AUGMENTATION_GUIDE.md](./GCN_AUGMENTATION_GUIDE.md)** - Data augmentation to handle class imbalance
- **[FINETUNE_ALL_MODELS.md](./FINETUNE_ALL_MODELS.md)** - Multi-model fine-tuning workflow
- **[FINETUNE_CTRGCN.md](./FINETUNE_CTRGCN.md)** - Detailed CTR-GCN fine-tuning guide
- **[BATCH_SCRIPTS_GUIDE.md](./BATCH_SCRIPTS_GUIDE.md)** - Using automated training scripts (.bat/.sh)

### Model Documentation
- **[gcn_models/README.md](./gcn_models/README.md)** - GCN model architectures overview
- **[gcn_models/MODEL_SUMMARY.md](./gcn_models/MODEL_SUMMARY.md)** - Model specifications and parameters
- **[pretrained_models/PRETRAINED_WEIGHTS_GUIDE.md](./pretrained_models/PRETRAINED_WEIGHTS_GUIDE.md)** - Pretrained model management

### Evaluation & Analysis
- **[Code+Images+Vision/Requirements and baseline results.md](./Code+Images+Vision/Requirements%20and%20baseline%20results.md)** - PBPL project requirements and baseline
- **[Code+Images+Vision/Evaluations/docs/analysis_report.md](./Code+Images+Vision/Evaluations/docs/analysis_report.md)** - Comprehensive trajectory prediction evaluation

### Component-Specific Guides
- **[CarlaSimulation/README.md](./CarlaSimulation/README.md)** - CARLA data collection detailed guide

## 🤝 Contributing

This is a research project. Key areas for contribution:
- Additional pedestrian behavior models
- New trajectory prediction architectures
- Improved multi-camera matching algorithms
- Enhanced visualization tools
- Performance optimizations

## 📄 License

See individual LICENSE files in component directories.

- `CarlaSimulation/LICENSE`: MIT License

## 🔗 Related Projects

### Core Dependencies
- [CARLA Simulator](https://carla.org/) - Open-source simulator for autonomous driving research
- [MMPose](https://github.com/open-mmlab/mmpose) - OpenMMLab pose estimation toolbox
- [MMDetection](https://github.com/open-mmlab/mmdetection) - OpenMMLab detection toolbox

### Required External Repositories
- [Alpamayo](https://github.com/NVlabs/alpamayo) - NVIDIA's vision-language model for trajectory planning
- [PCLA](https://github.com/MasoudJTehrani/PCLA) - Predictive Control with Learned Action Priors (CaRL)

### Related Research
- PyTorch Geometric for Graph Neural Networks
- Spatial-Temporal Graph Convolutional Networks (ST-GCN)
- Vision-language models for autonomous driving

## 📧 Contact

For questions or issues, please open a GitHub issue.

## 🙏 Acknowledgments

This project builds upon:
- **CARLA Simulator** for realistic autonomous driving simulation with pedestrian behaviors
- **OpenMMLab ecosystem** (MMPose, MMDetection) for pose estimation and object detection
- **Alpamayo** (NVIDIA) for vision-language trajectory planning capabilities
- **PCLA/CaRL** for predictive control with learned action priors
- **Graph Neural Network research** for temporal matching and trajectory prediction
- **PyTorch and PyTorch Geometric** for deep learning and graph-based models

Special thanks to the open-source communities maintaining these essential tools for autonomous driving research.

---

**Note**: This is an active research platform. Some components may require additional setup or configuration based on your specific hardware and environment.

**External Dependencies**: The Governor-Reflex system requires separate installation of [Alpamayo](https://github.com/NVlabs/alpamayo) and [PCLA](https://github.com/MasoudJTehrani/PCLA) repositories. Please follow their respective installation guides before using the Governor-Reflex components.
