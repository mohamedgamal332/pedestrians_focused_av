# Project Structure Documentation

## Complete Directory Tree

\\\
project_root/

 gcn_training/                          # GCN Training Module
    gcn_train.py                       # Main training script
    gcn_train_all.py                   # Multi-model training
    gcn_finetune.py                    # Finetuning script
    gcn_eval.py                        # Evaluation script
    gcn_infer.py                       # Inference script
    gcn_analyze_data.py                # Data analysis
    gcn_fetch_augment_data.py          # Data augmentation
    test_gcn.py                        # Unit tests

 gcn_models/                            # GCN Model Architectures
    stgcn.py                           # Spatial-Temporal GCN
    stgcn_config.py                    # STGCN configuration
    stgcn_model.py                     # Alternative STGCN impl
    ctrgcn.py                          # Channel-Time Relation GCN
    ctrgcn_motion.py                   # CTRGCN with motion
    tegcn.py                           # Temporal Enhanced GCN
    sht.py                             # Spatial-Heading-Temporal
    README.md                          # Model documentation
    MODEL_SUMMARY.md                   # Model specifications

 mmpose_setup/                          # RTMPose Integration
    run_train.py                       # Orchestrated training
    train_rtmpose.py                   # RTMPose training
    rtmpose_run.py                     # Inference runner
    rtmpose_eval.py                    # Evaluation

 data_loaders/                          # Data Loading Utilities
    dataloader.py                      # Generic loader
    gcn_loader.py                      # GCN-specific loader

 data_preprocessing/                    # Data Preparation
    create_coco_annotations.py         # COCO format converter
    create_splits.py                   # Train/val/test splits
    compute_state_vectors.py           # State vector computation
    compute_risks_from_labeled.py      # Risk computation

 risk_scoring/                          # Risk Assessment
    risk_score.py                      # Main risk scorer
    compute_risks_from_labeled.py      # Risk computation from labels

 plotting/                              # Visualization
    plot_training_history.py           # Training curves
    plot_training_log.py               # Log visualization
    plot_workdir_metrics_per_epoch.py  # Per-epoch metrics
    coco_graph.py                      # COCO visualization

 training_scripts/                      # Automated Training
    Windows Batch Files (.bat)
       finetune_stgcn.bat
       finetune_ctrgcn.bat
       finetune_ctrgcn_motion.bat
       finetune_tegcn.bat
       finetune_sht.bat
   
    Unix/Linux Shell Files (.sh)
        finetune_stgcn.sh
        finetune_ctrgcn.sh
        finetune_ctrgcn_motion.sh
        finetune_tegcn.sh
        finetune_sht.sh
        train.sh

 pretrained_models/                     # Pretrained Weights
    download_pretrained.py             # Download utility
    stgcn_kinetics.pth                 # Pretrained STGCN
    PRETRAINED_WEIGHTS_GUIDE.md        # Guide (optional)

 ROOT CONFIGURATION FILES
    README.md                          # Main documentation
    SETUP_AND_DEPLOYMENT.md            # Setup instructions
    GITHUB_DEPLOYMENT.md               # Git push guide
    PROJECT_STRUCTURE.md               # This file
    .gitignore                         # Git ignore rules
    requirements-yolopose.txt          # Python dependencies
    environment-yolopose.yml           # Conda environment
    GCN_TRAINING_GUIDE.md              # GCN guide
    GCN_AUGMENTATION_GUIDE.md          # Augmentation guide
    FINETUNE_ALL_MODELS.md             # Multi-model finetuning
    FINETUNE_CTRGCN.md                 # CTRGCN-specific guide
    BATCH_SCRIPTS_GUIDE.md             # Script execution guide

 AUTO-GENERATED DIRECTORIES (not pushed)
     data/                              # Training data
     work_dirs/                         # Training outputs
     eval_output/                       # Evaluation results
     __pycache__/                       # Python cache (ignored)
\\\

## Module Descriptions

### gcn_training/
**Purpose**: Core GCN training and evaluation functionality
**Key Scripts**:
- \gcn_train.py\: Main entry point for GCN training
- \gcn_eval.py\: Evaluate models on test datasets
- \gcn_infer.py\: Inference on new pedestrian data
- \gcn_finetune.py\: Fine-tune pretrained models
- \gcn_analyze_data.py\: Analyze dataset statistics and properties
- \gcn_fetch_augment_data.py\: Generate augmented training data

**Dependencies**: PyTorch, MMAction2, NumPy, scikit-learn

### gcn_models/
**Purpose**: GCN architecture implementations
**Model Types**:
1. **STGCN** (Spatial-Temporal GCN) - Base architecture
2. **CTRGCN** (Channel-Time Relation GCN) - Enhanced STGCN
3. **CTRGCN Motion** - CTRGCN with motion features
4. **TEGCN** (Temporal Enhanced GCN) - Temporal improvements
5. **SHT** (Spatial-Heading-Temporal) - Multi-modal variant

**Configuration**: \stgcn_config.py\ for model hyperparameters

### mmpose_setup/
**Purpose**: RTMPose pose estimation integration
**Components**:
- RTMPose-Large model training
- Pose estimation inference
- Evaluation metrics
- Integration with GCN for skeleton data

### data_loaders/
**Purpose**: Data loading and batch preparation
**Features**:
- Generic PyTorch DataLoader wrappers
- GCN-specific loading with augmentation
- Support for multiple data formats
- Batch collation utilities

### data_preprocessing/
**Purpose**: Raw data preparation and conversion
**Operations**:
- Convert video annotations to COCO JSON format
- Create train/validation/test splits
- Compute state vectors from poses
- Calculate risk scores from labeled data

### risk_scoring/
**Purpose**: Pedestrian behavior risk assessment
**Functionality**:
- Risk score computation from trajectories
- Behavior classification
- Risk metrics and statistics
- Integration with labeled datasets

### plotting/
**Purpose**: Training visualization and analysis
**Capabilities**:
- Plot training/validation curves
- Visualize per-epoch metrics
- COCO dataset exploration
- Error analysis visualization

### training_scripts/
**Purpose**: Automated model training execution
**File Types**:
- **.bat files**: Windows command batch scripts
- **.sh files**: Unix/Linux bash scripts
- Runs finetuning for specific GCN models
- Environment setup and dependency management

### pretrained_models/
**Purpose**: Pretrained weight management
**Contents**:
- Download utilities for standard pretrained weights
- STGCN Kinetics-400 pretrained checkpoint
- Model configuration guides

## File Organization Rationale

### By Functionality (Not by Model)
Files are organized by PURPOSE rather than model type:
- All training code  \gcn_training/\
- All model definitions  \gcn_models/\
- All data handling  \data_loaders/\, \data_preprocessing/\

This allows for:
-  Easy cross-model comparisons
-  Code reuse between models
-  Simplified dependency management
-  Better collaboration and updates

### By Execution Type
Separate directories for:
- Training scripts (Python)  \gcn_training/\, \mmpose_setup/\
- Automated scripts (.bat/.sh)  \	raining_scripts/\
- Analysis scripts  \plotting/\

This prevents:
-  Cluttered root directories
-  Execution confusion
-  Unnecessary file duplication

## Key File Dependencies

\\\
gcn_train.py
   imports from gcn_models/ (model architectures)
   imports from data_loaders/ (data loading)
   imports from data_preprocessing/ (data preparation)
   outputs to work_dirs/ (checkpoints, logs)

gcn_eval.py
   imports checkpoints from work_dirs/
   imports from gcn_models/
   imports from data_loaders/

risk_score.py
   imports from data_loaders/
   imports from data_preprocessing/ (if needed)
   outputs risk metrics

plot_*.py
   reads logs from work_dirs/
   generates visualization plots
\\\

## Data Flow

\\\
Raw Video Data
    
[create_coco_annotations.py]
    
COCO JSON Annotations
    
[create_splits.py]
    
Train/Val/Test Split
    
[data_loaders/]
    
Batches for Training
    
[gcn_train.py]  [gcn_models/]
    
Trained Models (work_dirs/)
    
[gcn_eval.py] + [plotting/]
    
Metrics & Visualizations
    
[risk_score.py]
    
Risk Scores & Analysis
\\\

## Getting Started

1. **Read**: Start with README.md
2. **Setup**: Follow SETUP_AND_DEPLOYMENT.md
3. **Explore**: Check GCN_TRAINING_GUIDE.md
4. **Run**: Execute training scripts in training_scripts/
5. **Analyze**: Use plotting/ scripts for results
6. **Push**: Follow GITHUB_DEPLOYMENT.md

## Contributing Guidelines

When adding new code:
1. Place in appropriate module directory
2. Follow existing naming conventions
3. Update README and relevant documentation
4. Test with existing data pipelines
5. Commit with descriptive messages

## Version Control

- **main**: Stable, deployable code
- **develop**: Integration branch (optional)
- **feature/***: Feature branches for new work

\\\ash
git checkout -b feature/new-feature
# ... make changes ...
git commit -am "Add new feature"
git push -u origin feature/new-feature
# Create Pull Request on GitHub
\\\

