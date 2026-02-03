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
├── pbplocalization.ipynb         # Pedestrian-based position localization
└── frame_*.png                   # Example visualization frames
```

## 🚀 Key Features

### CARLA Simulation System
- **Stereo Vision Setup**: Synchronized left/right RGB + depth cameras
- **Realistic Pedestrian Behaviors**: Walking, running, crossing, waiting, idle states
- **Jaywalking Simulation**: Traffic-aware crossing decisions with safety checks
- **Occlusion Detection**: Per-bone visibility analysis using depth buffers
- **Rich Annotations**: 3D skeleton data, behavior labels, and visibility states
- **Robust Connection Handling**: Automatic retry logic for CARLA initialization

### Pose Estimation & Trajectory Prediction
- **Multiple Model Architectures**:
  - STGCN: Spatial-Temporal Graph Convolutional Networks
  - CTR-GCN: Channel-wise Topology Refinement GCN
  - SHT: Spatial Hierarchical Transformer
  - LSTM and Mamba-based models
- **Risk Assessment**: Compute risk scores from pedestrian poses and trajectories
- **Real-time Inference**: Integration with MMPose for pose estimation

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

### 4. Train Trajectory Prediction Models

```bash
cd MMPose-Lib-staging/scripts

# Train different models
python train_transformer_trajectory.py
python train_lstm_trajectory.py
python train_mamba_trajectory.py
```

### 5. Run Governor-Reflex System

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

### 6. Pedestrian-Based Localization

Explore the pedestrian-based positioning and localization techniques:
```bash
jupyter notebook pbplocalization.ipynb
```

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

## 🔬 Research Applications

This platform supports research in:

- **Pedestrian Behavior Modeling**: Realistic jaywalking and crossing behaviors
- **Pose-based Trajectory Prediction**: Forecasting pedestrian movements from body poses
- **Multi-Modal Perception**: Combining RGB, depth, and skeleton data
- **Occlusion Handling**: Robust tracking under partial visibility
- **Safety-Critical Scenarios**: Testing AV responses to unpredictable pedestrians
- **Ablation Studies**: Evaluating impact of pedestrian information on AV performance

## 🎓 Model Architectures

### Trajectory Prediction Models

1. **STGCN (Spatial-Temporal GCN)**
   - Graph convolutions on skeleton topology
   - Temporal convolutions across frames
   - Efficient and fast inference

2. **CTR-GCN (Channel-wise Topology Refinement)**
   - Learnable graph topology
   - Channel-wise feature refinement
   - State-of-the-art on action recognition

3. **SHT (Spatial Hierarchical Transformer)**
   - Hierarchical attention mechanism
   - Multi-scale spatial features
   - Long-range temporal dependencies

4. **LSTM/Mamba-based Models**
   - Sequential modeling of trajectories
   - Recurrent architectures for temporal patterns

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

### GPU Memory Issues
- Use lower quality CARLA maps (Town03 instead of Town10HD_Opt)
- Reduce number of pedestrians/vehicles
- Use `-quality-level=low` flag for CARLA

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

**Note**: Governor-Reflex does not have a requirements.txt as it depends on the external Alpamayo and PCLA repositories, which have their own dependency management.

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
