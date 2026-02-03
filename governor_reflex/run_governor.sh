#!/bin/bash
# Run the Governor process

echo "=========================================="
echo "Starting Governor Process (Alpamayo)"
echo "=========================================="

# Activate alpo environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate alpo

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Change to governor_reflex directory
cd ~/trajectory-system/governor_reflex

# Run governor
python -u governor/main_governor.py "$@"
