#!/bin/bash
# Quick script to fine-tune SHT/Hyperformer

# Check if pretrained model exists (can use Hyperformer pretrained weights or trained model)
PRETRAINED_PATH="work_dirs/gcn_training/sht/best_model.pth"
HYPERFORMER_PRETRAINED="pretrained_weights/hyperformer/hyperformer_pretrained_weights/ntu60/csub/joint/runs-140-87640.pt"

# Check if we have a trained SHT model, otherwise use Hyperformer pretrained
if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "⚠️  Trained SHT model not found at $PRETRAINED_PATH"
    if [ -f "$HYPERFORMER_PRETRAINED" ]; then
        echo "Using Hyperformer pretrained weights instead..."
        PRETRAINED_PATH="$HYPERFORMER_PRETRAINED"
    else
        echo "Training SHT first (will use random initialization if no pretrained weights)..."
        python gcn_train.py --model sht --epochs 50 --batch_size 32
        PRETRAINED_PATH="work_dirs/gcn_training/sht/best_model.pth"
    fi
fi

# Check if augmented data exists
if [ ! -d "gcn_per_pedestrian_augmented" ]; then
    echo "⚠️  Augmented dataset not found. Generating..."
    python gcn_fetch_augment_data.py \
        --original_data_dir ./gcn_per_pedestrian \
        --output_dir ./gcn_per_pedestrian_augmented \
        --synthetic_samples 2000
fi

# Fine-tune SHT
echo "🚀 Starting SHT/Hyperformer fine-tuning..."
python gcn_finetune.py \
    --model sht \
    --pretrained_path "$PRETRAINED_PATH" \
    --original_data_dir ./gcn_per_pedestrian \
    --augmented_data_dir ./gcn_per_pedestrian_augmented \
    --epochs 30 \
    --lr 0.0001 \
    --batch_size 32

echo "✅ Fine-tuning complete!"
echo "Evaluate with:"
echo "  python gcn_eval.py --model sht --checkpoint work_dirs/gcn_finetuning/sht/best_finetuned_model.pth"
