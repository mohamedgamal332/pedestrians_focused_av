# GCN Data Augmentation and Fine-Tuning Guide

## Problem: Class Imbalance

Your dataset shows:
- **Walking**: 9,611 samples (60.67%) ⚠️
- **Running**: 1,937 samples (12.23%)
- **Crossing**: 2,863 samples (18.07%)
- **Waiting_To_Cross**: 0 samples (0.00%) ⚠️ **MISSING CLASS**

This imbalance causes poor performance, especially for the missing class.

## Solution: Data Augmentation + Fine-Tuning

### Step 1: Generate Synthetic Data

Create synthetic samples for the missing "Waiting_To_Cross" class:

```bash
python gcn_fetch_augment_data.py \
    --original_data_dir ./gcn_per_pedestrian \
    --output_dir ./gcn_per_pedestrian_augmented \
    --synthetic_samples 2000
```

This will:
1. Load your original data
2. Generate 2000 synthetic "Waiting_To_Cross" samples by:
   - Taking walking samples
   - Making them more static (reducing motion)
   - Adding small variations to simulate waiting/standing
3. Combine with original data
4. Save to `./gcn_per_pedestrian_augmented/`

### Step 2: Verify Augmented Dataset

```bash
python gcn_analyze_data.py --data_dir ./gcn_per_pedestrian_augmented
```

You should see:
- Total samples: ~17,842 (15,842 original + 2,000 synthetic)
- "Waiting_To_Cross" class now has samples
- Better class balance

### Step 3: Fine-Tune Pre-trained Model

Fine-tune your best model on the augmented dataset:

```bash
# Fine-tune ST-GCN
python gcn_finetune.py \
    --model stgcn \
    --pretrained_path work_dirs/gcn_training/stgcn/best_model.pth \
    --original_data_dir ./gcn_per_pedestrian \
    --augmented_data_dir ./gcn_per_pedestrian_augmented \
    --epochs 30 \
    --lr 0.0001

# Fine-tune CTR-GCN
python gcn_finetune.py \
    --model ctrgcn \
    --pretrained_path work_dirs/gcn_training/ctrgcn/best_model.pth \
    --original_data_dir ./gcn_per_pedestrian \
    --augmented_data_dir ./gcn_per_pedestrian_augmented \
    --epochs 30 \
    --lr 0.0001
```

### Step 4: Alternative - Train from Scratch on Augmented Data

If you prefer to train from scratch on augmented data:

```bash
python gcn_train.py \
    --model stgcn \
    --data_dir ./gcn_per_pedestrian_augmented \
    --epochs 50 \
    --batch_size 32
```

## Fine-Tuning Options

### Option 1: Full Fine-Tuning (Recommended)
Train all layers with lower learning rate:
```bash
python gcn_finetune.py --pretrained_path ... --lr 0.0001
```

### Option 2: Freeze Backbone
Only train the classifier (faster, less flexible):
```bash
python gcn_finetune.py --pretrained_path ... --freeze_backbone
```

### Option 3: Differential Learning Rates
Backbone gets 10x lower LR than classifier:
```bash
python gcn_finetune.py --pretrained_path ... --lr 0.0001
# (Automatically uses 0.1x LR for backbone, 1x for classifier)
```

## Expected Improvements

After fine-tuning on augmented data, you should see:

1. **Better overall accuracy**: 60-80% → 75-85%
2. **Improved per-class accuracy**: Especially for minority classes
3. **No more 0% for Waiting_To_Cross**: Should achieve 40-60%+
4. **More balanced confusion matrix**: Less bias toward Walking class

## Workflow Summary

```bash
# 1. Analyze original data
python gcn_analyze_data.py

# 2. Generate augmented dataset
python gcn_fetch_augment_data.py --synthetic_samples 2000

# 3. Verify augmented data
python gcn_analyze_data.py --data_dir ./gcn_per_pedestrian_augmented

# 4. Train initial models (if not done)
python gcn_train.py --model stgcn --epochs 50

# 5. Fine-tune on augmented data
python gcn_finetune.py \
    --model stgcn \
    --pretrained_path work_dirs/gcn_training/stgcn/best_model.pth \
    --augmented_data_dir ./gcn_per_pedestrian_augmented \
    --epochs 30

# 6. Evaluate fine-tuned model
python gcn_eval.py \
    --model stgcn \
    --checkpoint work_dirs/gcn_finetuning/stgcn/best_finetuned_model.pth
```

## Tips

1. **Synthetic Sample Count**: Start with 2000, adjust based on results
   - Too few: Class still underrepresented
   - Too many: May overfit to synthetic patterns

2. **Fine-Tuning Epochs**: 20-30 epochs usually sufficient
   - Monitor validation accuracy
   - Stop if validation accuracy plateaus

3. **Learning Rate**: Use 0.0001 for fine-tuning (10x lower than training)
   - Prevents overwriting learned features
   - Allows gradual adaptation

4. **Class Weights**: Already handled automatically in training scripts
   - Weighted loss compensates for imbalance
   - Works together with data augmentation

## Troubleshooting

### Synthetic data looks unrealistic?
- Reduce `synthetic_samples` count
- Check if original walking samples are diverse enough
- Consider using crossing samples as base instead

### Fine-tuning not improving?
- Try lower learning rate (0.00005)
- Increase fine-tuning epochs
- Check if pretrained model is actually good (should be >50% accuracy)

### Still poor performance on Waiting_To_Cross?
- Increase synthetic samples (try 3000-5000)
- Use different base class (try Crossing instead of Walking)
- Consider collecting real waiting samples
