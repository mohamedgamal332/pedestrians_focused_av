#!/usr/bin/env python3
"""
Inference with GCN models per pedestrian.
COCO 17 keypoints format.
Runs all available models and compares results.
"""
import torch
import numpy as np
import sys
import argparse
from pathlib import Path
from collections import defaultdict

# Add current directory to path (models are in ./models/)
sys.path.insert(0, str(Path(__file__).parent))

from models.stgcn import EnhancedSTGCN
from models.ctrgcn import EnhancedCTRGCN
from models.ctrgcn_motion import EnhancedCTRGCN_Motion
from models.tegcn import TE_GCN
try:
    from models.sht import SHT_Hyperformer
    HAS_SHT = True
except ImportError:
    HAS_SHT = False

# -----------------------------
# Model configurations with pretrained weights
# -----------------------------
MODEL_CONFIGS = {
    'stgcn': {
        'class': EnhancedSTGCN,
        'pretrained': 'pretrained_weights/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth',
        'config_file': 'mmaction2/configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py',
        'description': 'ST-GCN (Spatial Temporal Graph Convolutional Networks)'
    },
    'ctrgcn': {
        'class': EnhancedCTRGCN,
        'pretrained': None,  # Check if available
        'config_file': 'mmaction2/projects/ctrgcn/configs/ctrgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py',
        'description': 'CTR-GCN (Channel-wise Topology Refinement GCN)'
    },
    'ctrgcn_motion': {
        'class': EnhancedCTRGCN_Motion,
        'pretrained': None,  # Use trained checkpoint (work_dirs/gcn_training/ctrgcn_motion/best_model.pth)
        'config_file': None,
        'description': 'CTR-GCN with Motion stream (spatial + temporal delta branches)'
    },
    'tegcn': {
        'class': TE_GCN,
        'pretrained': None,  # Custom model, no MMACTION2 pretrained weights
        'config_file': None,
        'description': 'TE-GCN (Taylor Expansion GCN) - Custom model'
    }
}

if HAS_SHT:
    MODEL_CONFIGS['sht'] = {
        'class': SHT_Hyperformer,
        'pretrained': 'pretrained_weights/hyperformer/hyperformer_pretrained_weights/ntu60/csub/joint/runs-140-87640.pt',
        'config_file': None,
        'description': 'SHT/Hyperformer (Hypergraph Transformer) - State-of-the-art'
    }

# -----------------------------
# Parse command-line arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Run GCN inference on pedestrian data")
parser.add_argument("--data_dir", type=str, default="./gcn_per_pedestrian",
                    help="Directory containing data.npy and labels.npy")
parser.add_argument("--pretrained_path", type=str, default=None,
                    help="Path to pretrained model weights (overrides default)")
parser.add_argument("--model", type=str, default="all", 
                    choices=["all", "stgcn", "ctrgcn", "ctrgcn_motion", "tegcn"],
                    help="Model architecture to use (default: all)")
parser.add_argument("--num_classes", type=int, default=5,
                    help="Number of behavior classes")
parser.add_argument("--device", type=str, default=None,
                    help="Device to use (cuda/cpu). Auto-detected if not specified")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size for inference (default: 32)")
args = parser.parse_args()

# -----------------------------
# Settings
# -----------------------------
DATA_DIR = Path(args.data_dir)
DATA_PATH = DATA_DIR / "data.npy"
LABELS_PATH = DATA_DIR / "labels.npy"
NUM_CLASSES = args.num_classes
NUM_JOINTS = 17  # COCO format
DEVICE = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

# Determine which models to run
if args.model == "all":
    models_to_run = list(MODEL_CONFIGS.keys())
else:
    if args.model not in MODEL_CONFIGS:
        raise ValueError(f"Model '{args.model}' not available. Available: {list(MODEL_CONFIGS.keys())}")
    models_to_run = [args.model]

# Check for available pretrained weights and configs
print("=" * 80)
print("Checking available models and configs...")
print("=" * 80)
for model_name in models_to_run:
    config = MODEL_CONFIGS[model_name]
    pretrained_path = args.pretrained_path if args.pretrained_path else config['pretrained']
    
    # Check pretrained weights
    if pretrained_path:
        pretrained_exists = Path(pretrained_path).exists()
        status = "✓" if pretrained_exists else "✗"
        print(f"{status} {model_name.upper()}: Pretrained weights: {pretrained_path}")
        if not pretrained_exists:
            print(f"    Warning: Pretrained weights not found, model will use random initialization")
    else:
        print(f"✗ {model_name.upper()}: No pretrained weights specified")
    
    # Check config file
    if config['config_file']:
        config_exists = Path(config['config_file']).exists()
        status = "✓" if config_exists else "✗"
        print(f"{status} {model_name.upper()}: Config file: {config['config_file']}")
    else:
        print(f"  {model_name.upper()}: Custom model (no MMACTION2 config)")

print("=" * 80)

# -----------------------------
# Load data
# -----------------------------
print(f"\nLoading data from: {DATA_PATH}")
X = np.load(DATA_PATH)  # [N, C, T, V] where V should be 17 for COCO
try:
    y = np.load(LABELS_PATH)
    print(f"Loaded labels from: {LABELS_PATH}")
except FileNotFoundError:
    y = None
    print("No labels file found, skipping accuracy calculation")

print(f"Loaded {X.shape[0]} sequences, {X.shape[2]} frames, {X.shape[3]} joints.")

# Verify data shape
if X.shape[3] != NUM_JOINTS:
    print(f"Warning: Expected {NUM_JOINTS} joints (COCO format), but got {X.shape[3]}")
    print(f"Make sure your data is in COCO 17 keypoint format")

# Convert to torch
X_torch = torch.from_numpy(X).float().to(DEVICE)  # [N, C, T, V]

# -----------------------------
# Run inference for all models
# -----------------------------
results = {}

for model_name in models_to_run:
    print("\n" + "=" * 80)
    print(f"Running {model_name.upper()}: {MODEL_CONFIGS[model_name]['description']}")
    print("=" * 80)
    
    config = MODEL_CONFIGS[model_name]
    pretrained_path = args.pretrained_path if args.pretrained_path else config['pretrained']
    
    # Check if pretrained weights exist
    if pretrained_path and not Path(pretrained_path).exists():
        print(f"Warning: Pretrained weights not found at {pretrained_path}")
        print(f"Model will use random initialization")
        pretrained_path = None
    
    try:
        # Load model
        print(f"\nLoading {model_name} model...")
        model_class = config['class']
        if model_name == 'ctrgcn_motion':
            model = model_class(
                in_channels=X.shape[1],
                num_joints=NUM_JOINTS,
                num_classes=NUM_CLASSES,
                pretrained_spatial=None,
                pretrained_motion=None
            ).to(DEVICE)
            if pretrained_path and Path(pretrained_path).exists():
                ckpt = torch.load(pretrained_path, map_location=DEVICE)
                state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
                model.load_state_dict(state, strict=False)
                print(f"Loaded checkpoint from {pretrained_path}")
        else:
            model = model_class(
                in_channels=X.shape[1],
                num_joints=NUM_JOINTS,
                num_classes=NUM_CLASSES,
                pretrained_path=pretrained_path
            ).to(DEVICE)
        model.eval()
        print(f"Model loaded successfully!")
        
        # Run inference
        print(f"\nRunning inference with batch size {args.batch_size}...")
        all_outputs = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            num_samples = X_torch.shape[0]
            for i in range(0, num_samples, args.batch_size):
                batch_end = min(i + args.batch_size, num_samples)
                batch_X = X_torch[i:batch_end]
                
                batch_outputs = model(batch_X)  # [batch_size, num_classes]
                batch_preds = torch.argmax(batch_outputs, dim=1)
                batch_probs = torch.softmax(batch_outputs, dim=1)
                
                all_outputs.append(batch_outputs.cpu())
                all_preds.append(batch_preds.cpu())
                all_probs.append(batch_probs.cpu())
                
                if (i // args.batch_size + 1) % 100 == 0:
                    print(f"  Processed {batch_end}/{num_samples} samples...")
        
        # Concatenate all results
        outputs = torch.cat(all_outputs, dim=0)
        preds = torch.cat(all_preds, dim=0)
        probs = torch.cat(all_probs, dim=0)
        
        # Calculate accuracy if labels available
        accuracy = None
        if y is not None:
            accuracy = (preds.numpy() == y).mean()
        
        # Store results
        results[model_name] = {
            'preds': preds.numpy(),
            'probs': probs.numpy(),
            'accuracy': accuracy,
            'outputs': outputs.numpy()
        }
        
        print(f"\n✓ Completed inference on {num_samples} samples")
        if accuracy is not None:
            print(f"✓ Accuracy: {accuracy*100:.2f}%")
        
    except Exception as e:
        print(f"\n✗ Error running {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        results[model_name] = None

# -----------------------------
# Summary comparison
# -----------------------------
print("\n" + "=" * 80)
print("SUMMARY - Model Comparison")
print("=" * 80)

if y is not None:
    print(f"\n{'Model':<15} {'Accuracy':<12} {'Status'}")
    print("-" * 80)
    for model_name in models_to_run:
        if results.get(model_name) and results[model_name]['accuracy'] is not None:
            acc = results[model_name]['accuracy']
            print(f"{model_name.upper():<15} {acc*100:>10.2f}%  ✓")
        else:
            print(f"{model_name.upper():<15} {'N/A':<12} ✗")
    
    # Find best model
    valid_results = {k: v for k, v in results.items() 
                     if v and v['accuracy'] is not None}
    if valid_results:
        best_model = max(valid_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n🏆 Best Model: {best_model[0].upper()} ({best_model[1]['accuracy']*100:.2f}%)")

# Show sample predictions from first successful model
for model_name in models_to_run:
    if results.get(model_name):
        print(f"\n{model_name.upper()} - Sample predictions (first 10):")
        preds = results[model_name]['preds']
        probs = results[model_name]['probs']
        for i in range(min(10, len(preds))):
            print(f"  Sample {i}: Class {preds[i]} (confidence: {probs[i][preds[i]]:.2%})")
        break
