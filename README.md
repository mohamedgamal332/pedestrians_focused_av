# Pedestrian-Focused Autonomous Vehicle Research Platform

A comprehensive research platform for developing and evaluating pedestrian-aware autonomous vehicle systems. This project integrates simulation, pose estimation, trajectory prediction, and autonomous driving control with a focus on pedestrian safety and behavior understanding.

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
- **Governor (Planning)**: Long-term trajectory planning using Alpamayo vision-language model
- **Reflex (Control)**: Low-level reactive control with CaRL
- **Pedestrian Integration**: Real-time pedestrian tracking and trajectory adjustment
- **A/B Testing Support**: Toggle pedestrian information for ablation studies

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CARLA Simulator 0.9.15
- CUDA-capable GPU (recommended)
- Conda/Mamba package manager

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
```bash
# Create separate conda environments
conda create -n alpo python=3.8
conda activate alpo
# Install Alpamayo dependencies...

conda create -n PCLA python=3.8
conda activate PCLA
# Install CaRL dependencies...
```

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

```bash
cd governor_reflex

# Terminal 1: Start Reflex (reactive control)
conda activate PCLA
python reflex/main_reflex.py

# Terminal 2: Start Governor (planning)
conda activate alpo
python governor/main_governor.py
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
carla:
  host: "localhost"
  port: 2000

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
Ensure CARLA Python API is in your PYTHONPATH:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/carla/PythonAPI/carla/dist/carla-0.9.15-*.egg
```

## 📚 Dependencies

Core dependencies:
- **CARLA** 0.9.15: Simulation environment
- **PyTorch**: Deep learning framework
- **MMPose**: Pose estimation library
- **MMDetection**: Object detection
- **PyTorch Geometric**: Graph neural networks
- **OpenCV**: Image processing
- **NumPy, Pandas**: Data manipulation

See individual `requirements.txt` files in each component directory.

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

- [CARLA Simulator](https://carla.org/)
- [MMPose](https://github.com/open-mmlab/mmpose)
- [MMDetection](https://github.com/open-mmlab/mmdetection)

## 📧 Contact

For questions or issues, please open a GitHub issue.

## 🙏 Acknowledgments

This project builds upon:
- CARLA Simulator for realistic autonomous driving simulation
- OpenMMLab ecosystem for pose estimation
- Graph Neural Network research for temporal matching
- Vision-language models for trajectory planning

---

**Note**: This is an active research platform. Some components may require additional setup or configuration based on your specific hardware and environment.
