# Fine-Tuning All Models Guide

## Available Models

1. **ST-GCN** - Baseline spatial-temporal GCN
2. **CTR-GCN** - Channel-wise topology refinement (currently 60.67% accuracy)
3. **TE-GCN** - Taylor expansion GCN (custom model)
4. **SHT/Hyperformer** - Hypergraph transformer (state-of-the-art)

## Quick Start - Batch Scripts

### Windows (.bat files)

```bash
# Fine-tune ST-GCN
finetune_stgcn.bat

# Fine-tune CTR-GCN
finetune_ctrgcn.bat

# Fine-tune TE-GCN
finetune_tegcn.bat

# Fine-tune SHT/Hyperformer
finetune_sht.bat
```

### Linux/Mac (.sh files)

```bash
# Make scripts executable
chmod +x finetune_*.sh

# Run fine-tuning
./finetune_stgcn.sh
./finetune_ctrgcn.sh
./finetune_tegcn.sh
./finetune_sht.sh
```

## Manual Fine-Tuning Commands

### ST-GCN
```bash
python gcn_finetune.py \
    --model stgcn \
    --pretrained_path work_dirs/gcn_training/stgcn/best_model.pth \
    --original_data_dir ./gcn_per_pedestrian \
    --augmented_data_dir ./gcn_per_pedestrian_augmented \
    --epochs 30 \
    --lr 0.0001
```

### CTR-GCN
```bash
python gcn_finetune.py \
    --model ctrgcn \
    --pretrained_path work_dirs/gcn_training/ctrgcn/best_model.pth \
    --original_data_dir ./gcn_per_pedestrian \
    --augmented_data_dir ./gcn_per_pedestrian_augmented \
    --epochs 30 \
    --lr 0.0001
```

### TE-GCN
```bash
python gcn_finetune.py \
    --model tegcn \
    --pretrained_path work_dirs/gcn_training/tegcn/best_model.pth \
    --original_data_dir ./gcn_per_pedestrian \
    --augmented_data_dir ./gcn_per_pedestrian_augmented \
    --epochs 30 \
    --lr 0.0001
```

### SHT/Hyperformer
```bash
python gcn_finetune.py \
    --model sht \
    --pretrained_path work_dirs/gcn_training/sht/best_model.pth \
    --original_data_dir ./gcn_per_pedestrian \
    --augmented_data_dir ./gcn_per_pedestrian_augmented \
    --epochs 30 \
    --lr 0.0001
```

## Prerequisites

### 1. Generate Augmented Dataset
All scripts check for augmented data and generate it if missing:
```bash
python gcn_fetch_augment_data.py \
    --original_data_dir ./gcn_per_pedestrian \
    --output_dir ./gcn_per_pedestrian_augmented \
    --synthetic_samples 2000
```

### 2. Train Initial Models (if not done)
```bash
# Train all models
python gcn_train_all.py --epochs 50

# Or train individually
python gcn_train.py --model stgcn --epochs 50
python gcn_train.py --model ctrgcn --epochs 50
python gcn_train.py --model tegcn --epochs 50
python gcn_train.py --model sht --epochs 50
```

## Fine-Tuning Options

### Full Fine-Tuning (Recommended)
Trains all layers with differential learning rates:
- Backbone: 0.1x learning rate
- Classifier: 1x learning rate

```bash
python gcn_finetune.py --model <model> --pretrained_path <path> --epochs 30 --lr 0.0001
```

### Freeze Backbone
Only trains classifier (faster, less flexible):
```bash
python gcn_finetune.py --model <model> --pretrained_path <path> --freeze_backbone
```

## Expected Results

| Model | Before Fine-Tuning | After Fine-Tuning |
|-------|-------------------|-------------------|
| ST-GCN | 18.07% | 70-80% |
| CTR-GCN | 60.67% | 75-85% |
| TE-GCN | 0.00% | 50-70% |
| SHT | N/A | 80-90% (if pretrained) |

## Output Locations

Fine-tuned models are saved to:
```
work_dirs/gcn_finetuning/
├── stgcn/
│   └── best_finetuned_model.pth
├── ctrgcn/
│   └── best_finetuned_model.pth
├── tegcn/
│   └── best_finetuned_model.pth
└── sht/
    └── best_finetuned_model.pth
```

## Evaluate All Fine-Tuned Models

```bash
# Evaluate ST-GCN
python gcn_eval.py --model stgcn --checkpoint work_dirs/gcn_finetuning/stgcn/best_finetuned_model.pth

# Evaluate CTR-GCN
python gcn_eval.py --model ctrgcn --checkpoint work_dirs/gcn_finetuning/ctrgcn/best_finetuned_model.pth

# Evaluate TE-GCN
python gcn_eval.py --model tegcn --checkpoint work_dirs/gcn_finetuning/tegcn/best_finetuned_model.pth

# Evaluate SHT
python gcn_eval.py --model sht --checkpoint work_dirs/gcn_finetuning/sht/best_finetuned_model.pth
```

## Complete Workflow

```bash
# 1. Generate augmented dataset
python gcn_fetch_augment_data.py --synthetic_samples 2000

# 2. Train initial models
python gcn_train_all.py --epochs 50

# 3. Fine-tune all models
finetune_stgcn.bat
finetune_ctrgcn.bat
finetune_tegcn.bat
finetune_sht.bat

# 4. Compare results
python gcn_infer.py --model all
```

## Troubleshooting

### SHT Model Not Available
If you get "SHT/Hyperformer not available":
- Install Hyperformer dependencies
- Check that `model/hyperformer.py` exists
- Verify pretrained weights are in `pretrained_weights/hyperformer/`

### Model Not Found
If a pretrained model is missing:
- Train it first: `python gcn_train.py --model <model> --epochs 50`
- Or use MMACTION2 pretrained weights (for ST-GCN)

### Class Number Mismatch
The script auto-detects number of classes from your data. If you get errors:
- Check that augmented dataset has correct number of classes
- Verify labels are in range [0, num_classes-1]
