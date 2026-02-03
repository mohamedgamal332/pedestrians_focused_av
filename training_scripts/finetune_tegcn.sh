#!/bin/bash
# Quick script to fine-tune TE-GCN

# Check if pretrained model exists
PRETRAINED_PATH="work_dirs/gcn_training/tegcn/best_model.pth"

if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "⚠️  Pretrained TE-GCN model not found at $PRETRAINED_PATH"
    echo "Training TE-GCN first..."
    python gcn_train.py --model tegcn --epochs 50 --batch_size 32
fi

# Check if augmented data exists
if [ ! -d "gcn_per_pedestrian_augmented" ]; then
    echo "⚠️  Augmented dataset not found. Generating..."
    python gcn_fetch_augment_data.py \
        --original_data_dir ./gcn_per_pedestrian \
        --output_dir ./gcn_per_pedestrian_augmented \
        --synthetic_samples 2000
fi

# Fine-tune TE-GCN
echo "🚀 Starting TE-GCN fine-tuning..."
python gcn_finetune.py \
    --model tegcn \
    --pretrained_path "$PRETRAINED_PATH" \
    --original_data_dir ./gcn_per_pedestrian \
    --augmented_data_dir ./gcn_per_pedestrian_augmented \
    --epochs 30 \
    --lr 0.0001 \
    --batch_size 32

echo "✅ Fine-tuning complete!"
echo "Evaluate with:"
echo "  python gcn_eval.py --model tegcn --checkpoint work_dirs/gcn_finetuning/tegcn/best_finetuned_model.pth"
