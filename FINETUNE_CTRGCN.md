# Fine-Tune CTR-GCN Guide

## Quick Start

### Step 1: Ensure you have a trained CTR-GCN model

If you haven't trained CTR-GCN yet, train it first:

```bash
python gcn_train.py --model ctrgcn --epochs 50 --batch_size 32
```

This will create: `work_dirs/gcn_training/ctrgcn/best_model.pth`

### Step 2: Generate augmented dataset (if not done)

```bash
python gcn_fetch_augment_data.py \
    --original_data_dir ./gcn_per_pedestrian \
    --output_dir ./gcn_per_pedestrian_augmented \
    --synthetic_samples 2000
```

### Step 3: Fine-tune CTR-GCN

```bash
python gcn_finetune.py \
    --model ctrgcn \
    --pretrained_path work_dirs/gcn_training/ctrgcn/best_model.pth \
    --original_data_dir ./gcn_per_pedestrian \
    --augmented_data_dir ./gcn_per_pedestrian_augmented \
    --epochs 30 \
    --lr 0.0001 \
    --batch_size 32
```

## Fine-Tuning Options

### Full Fine-Tuning (Recommended)
Trains all layers with differential learning rates:
- Backbone: 0.1x learning rate (0.00001)
- Classifier: 1x learning rate (0.0001)

```bash
python gcn_finetune.py \
    --model ctrgcn \
    --pretrained_path work_dirs/gcn_training/ctrgcn/best_model.pth \
    --original_data_dir ./gcn_per_pedestrian \
    --augmented_data_dir ./gcn_per_pedestrian_augmented \
    --epochs 30 \
    --lr 0.0001
```

### Freeze Backbone (Faster, Less Flexible)
Only trains the classifier:

```bash
python gcn_finetune.py \
    --model ctrgcn \
    --pretrained_path work_dirs/gcn_training/ctrgcn/best_model.pth \
    --original_data_dir ./gcn_per_pedestrian \
    --augmented_data_dir ./gcn_per_pedestrian_augmented \
    --epochs 30 \
    --lr 0.0001 \
    --freeze_backbone
```

### Custom Learning Rate
Use a different learning rate:

```bash
# Lower learning rate (more conservative)
python gcn_finetune.py ... --lr 0.00005

# Higher learning rate (more aggressive)
python gcn_finetune.py ... --lr 0.0005
```

## Expected Results

**Before Fine-Tuning:**
- CTR-GCN: ~60.67% accuracy
- Missing "Waiting_To_Cross" class (0% accuracy)

**After Fine-Tuning:**
- CTR-GCN: 75-85% accuracy
- "Waiting_To_Cross": 40-60% accuracy
- Better per-class balance

## Output Files

Fine-tuned models are saved to:
```
work_dirs/gcn_finetuning/ctrgcn/
├── best_finetuned_model.pth    # Best model during fine-tuning
├── final_finetuned_model.pth   # Final epoch model
└── finetuning_history.npy      # Training history
```

## Evaluate Fine-Tuned Model

```bash
python gcn_eval.py \
    --model ctrgcn \
    --checkpoint work_dirs/gcn_finetuning/ctrgcn/best_finetuned_model.pth \
    --class_names Walking Running Crossing Waiting_To_Cross Idle
```

## Troubleshooting

### "Pretrained checkpoint not found"
Make sure you've trained CTR-GCN first:
```bash
python gcn_train.py --model ctrgcn --epochs 50
```

### "Number of classes mismatch"
The script auto-detects number of classes from your data. If you get this error:
- Check that your augmented dataset has the expected number of classes
- Verify labels are in range [0, num_classes-1]

### Low improvement after fine-tuning
- Try more epochs: `--epochs 50`
- Lower learning rate: `--lr 0.00005`
- Check if augmented data quality is good
- Verify class distribution is balanced

### CUDA Out of Memory
- Reduce batch size: `--batch_size 16`
- Use CPU: `--device cpu`

## Complete Workflow

```bash
# 1. Train initial CTR-GCN
python gcn_train.py --model ctrgcn --epochs 50

# 2. Generate augmented data
python gcn_fetch_augment_data.py --synthetic_samples 2000

# 3. Fine-tune on augmented data
python gcn_finetune.py \
    --model ctrgcn \
    --pretrained_path work_dirs/gcn_training/ctrgcn/best_model.pth \
    --augmented_data_dir ./gcn_per_pedestrian_augmented \
    --epochs 30

# 4. Evaluate
python gcn_eval.py \
    --model ctrgcn \
    --checkpoint work_dirs/gcn_finetuning/ctrgcn/best_finetuned_model.pth
```
