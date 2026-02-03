# GCN Training, Testing, and Evaluation Guide

## Overview

This guide covers training, testing, and evaluating GCN models for pedestrian behavior recognition using COCO 17 keypoint data.

## Data Preparation

First, ensure you have prepared the data using `gcn_loader.py`:

```bash
python gcn_loader.py --session_path /path/to/session --output_dir ./gcn_per_pedestrian
```

This creates:
- `gcn_per_pedestrian/data.npy`: Shape [N, C, T, V] where V=17 (COCO keypoints)
- `gcn_per_pedestrian/labels.npy`: Shape [N] with behavior class indices

## Behavior Classes

The models predict 5 behavior classes:
1. **Walking** (0)
2. **Running** (1)
3. **Crossing** (2)
4. **Waiting_To_Cross** (3)
5. **Idle** (4)

## Training

### Train a Single Model

```bash
# Train ST-GCN
python gcn_train.py --model stgcn --epochs 50 --batch_size 32 --lr 0.001

# Train CTR-GCN
python gcn_train.py --model ctrgcn --epochs 50 --batch_size 32 --lr 0.001

# Train TE-GCN
python gcn_train.py --model tegcn --epochs 50 --batch_size 32 --lr 0.001
```

### Train with Pretrained Weights (ST-GCN only)

```bash
python gcn_train.py --model stgcn \
    --pretrained_path pretrained_weights/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth \
    --epochs 50
```

### Train All Models

```bash
# Train all models sequentially
python gcn_train_all.py --epochs 50 --batch_size 32

# With pretrained weights for ST-GCN
python gcn_train_all.py --epochs 50 --pretrained
```

### Training Parameters

- `--data_dir`: Directory with data.npy and labels.npy (default: `./gcn_per_pedestrian`)
- `--model`: Model architecture (`stgcn`, `ctrgcn`, `tegcn`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay (default: 0.0001)
- `--train_split`: Train/validation split ratio (default: 0.8)
- `--output_dir`: Output directory for checkpoints (default: `./work_dirs/gcn_training`)
- `--save_interval`: Save checkpoint every N epochs (default: 5)

## Evaluation

### Evaluate a Trained Model

```bash
# Evaluate ST-GCN
python gcn_eval.py --model stgcn \
    --checkpoint work_dirs/gcn_training/stgcn/best_model.pth

# Evaluate CTR-GCN
python gcn_eval.py --model ctrgcn \
    --checkpoint work_dirs/gcn_training/ctrgcn/best_model.pth

# Evaluate TE-GCN
python gcn_eval.py --model tegcn \
    --checkpoint work_dirs/gcn_training/tegcn/best_model.pth
```

### Evaluation Output

The evaluation script provides:
- Overall accuracy
- Per-class accuracy
- Confusion matrix
- Classification report (precision, recall, F1-score)
- Class distribution in dataset

Results are saved to `evaluation_results.npz` in the checkpoint directory.

## Inference

### Run Inference on All Models

```bash
python gcn_infer.py --model all
```

### Run Inference on Specific Model

```bash
python gcn_infer.py --model stgcn --pretrained_path work_dirs/gcn_training/stgcn/best_model.pth
```

## Training Tips

### Improving Low Performance

If you see low accuracy (like STGCN 18%, TE-GCN 0%):

1. **Check Data Quality**:
   ```bash
   # Verify data distribution
   python -c "import numpy as np; y=np.load('gcn_per_pedestrian/labels.npy'); print('Class distribution:', np.bincount(y))"
   ```

2. **Use Pretrained Weights** (for ST-GCN):
   ```bash
   python gcn_train.py --model stgcn --pretrained_path pretrained_weights/stgcn/...
   ```

3. **Adjust Learning Rate**:
   - Try lower learning rate: `--lr 0.0001`
   - Try higher learning rate: `--lr 0.01`

4. **Increase Training Epochs**:
   ```bash
   python gcn_train.py --model stgcn --epochs 100
   ```

5. **Check Class Imbalance**:
   - If classes are imbalanced, consider weighted loss or data augmentation

6. **Verify Model Architecture**:
   - Ensure input shape matches: [N, C, T, V] where C=2 or 3, T=30, V=17

## Expected Performance

After training, you should see:
- **ST-GCN**: 60-80% accuracy (with pretrained weights)
- **CTR-GCN**: 60-85% accuracy
- **TE-GCN**: 50-75% accuracy (custom model, no pretrained weights)

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch_size 16`
- Use CPU: `--device cpu`

### Low Accuracy
- Check if data is normalized correctly
- Verify label encoding matches class indices
- Check for class imbalance
- Try different learning rates
- Use pretrained weights if available

### Model Not Converging
- Lower learning rate
- Increase training epochs
- Check data preprocessing
- Verify model architecture matches data shape

## File Structure

```
work_dirs/gcn_training/
├── stgcn/
│   ├── best_model.pth          # Best model checkpoint
│   ├── final_model.pth         # Final epoch checkpoint
│   ├── checkpoint_epoch_5.pth  # Periodic checkpoints
│   ├── history.npy             # Training history
│   └── evaluation_results.npz  # Evaluation results
├── ctrgcn/
│   └── ...
└── tegcn/
    └── ...
```

## Next Steps

1. Train all models: `python gcn_train_all.py --epochs 50`
2. Evaluate best models: Use `gcn_eval.py` on each best_model.pth
3. Compare results: Check evaluation_results.npz files
4. Fine-tune: Adjust hyperparameters based on validation performance
