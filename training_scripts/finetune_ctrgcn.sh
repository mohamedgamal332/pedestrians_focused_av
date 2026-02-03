#!/bin/bash
# Quick script to fine-tune CTR-GCN

# Check if pretrained model exists
PRETRAINED_PATH="work_dirs/gcn_training/ctrgcn/best_model.pth"

if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "⚠️  Pretrained CTR-GCN model not found at $PRETRAINED_PATH"
    echo "Training CTR-GCN first..."
    python gcn_train.py --model ctrgcn --epochs 50 --batch_size 32
fi

# Check if augmented data exists
if [ ! -d "gcn_per_pedestrian_augmented" ]; then
    echo "⚠️  Augmented dataset not found. Generating..."
    python gcn_fetch_augment_data.py \
        --original_data_dir ./gcn_per_pedestrian \
        --output_dir ./gcn_per_pedestrian_augmented \
        --synthetic_samples 2000
fi

# Fine-tune CTR-GCN
echo "🚀 Starting CTR-GCN fine-tuning..."
python gcn_finetune.py \
    --model ctrgcn \
    --pretrained_path "$PRETRAINED_PATH" \
    --original_data_dir ./gcn_per_pedestrian \
    --augmented_data_dir ./gcn_per_pedestrian_augmented \
    --epochs 30 \
    --lr 0.0001 \
    --batch_size 32

echo "✅ Fine-tuning complete!"
echo "Evaluate with:"
echo "  python gcn_eval.py --model ctrgcn --checkpoint work_dirs/gcn_finetuning/ctrgcn/best_finetuned_model.pth"
