#!/bin/bash
# Quick script to fine-tune ST-GCN

# Check if pretrained model exists
PRETRAINED_PATH="work_dirs/gcn_training/stgcn/best_model.pth"

if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "⚠️  Pretrained ST-GCN model not found at $PRETRAINED_PATH"
    echo "Training ST-GCN first..."
    python gcn_train.py --model stgcn --epochs 50 --batch_size 32 \
        --pretrained_path pretrained_weights/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth
fi

# Check if augmented data exists
if [ ! -d "gcn_per_pedestrian_augmented" ]; then
    echo "⚠️  Augmented dataset not found. Generating..."
    python gcn_fetch_augment_data.py \
        --original_data_dir ./gcn_per_pedestrian \
        --output_dir ./gcn_per_pedestrian_augmented \
        --synthetic_samples 2000
fi

# Fine-tune ST-GCN
echo "🚀 Starting ST-GCN fine-tuning..."
python gcn_finetune.py \
    --model stgcn \
    --pretrained_path "$PRETRAINED_PATH" \
    --original_data_dir ./gcn_per_pedestrian \
    --augmented_data_dir ./gcn_per_pedestrian_augmented \
    --epochs 30 \
    --lr 0.0001 \
    --batch_size 32

echo "✅ Fine-tuning complete!"
echo "Evaluate with:"
echo "  python gcn_eval.py --model stgcn --checkpoint work_dirs/gcn_finetuning/stgcn/best_finetuned_model.pth"
