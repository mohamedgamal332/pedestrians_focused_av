# Batch Scripts Guide - Fine-Tuning All Models

## Available Batch Scripts

### Windows (.bat files)

| Script | Model | Description |
|--------|-------|-------------|
| `finetune_stgcn.bat` | ST-GCN | Fine-tune ST-GCN with pretrained MMACTION2 weights |
| `finetune_ctrgcn.bat` | CTR-GCN | Fine-tune CTR-GCN (currently 60.67% accuracy) |
| `finetune_tegcn.bat` | TE-GCN | Fine-tune custom Taylor Expansion GCN |
| `finetune_sht.bat` | SHT/Hyperformer | Fine-tune state-of-the-art Hyperformer model |

### Linux/Mac (.sh files)

| Script | Model | Description |
|--------|-------|-------------|
| `finetune_stgcn.sh` | ST-GCN | Fine-tune ST-GCN |
| `finetune_ctrgcn.sh` | CTR-GCN | Fine-tune CTR-GCN |
| `finetune_tegcn.sh` | TE-GCN | Fine-tune TE-GCN |
| `finetune_sht.sh` | SHT/Hyperformer | Fine-tune SHT/Hyperformer |

## Quick Usage

### Windows
```bash
# Double-click or run from command prompt
finetune_stgcn.bat
finetune_ctrgcn.bat
finetune_tegcn.bat
finetune_sht.bat
```

### Linux/Mac
```bash
# Make executable (first time only)
chmod +x finetune_*.sh

# Run scripts
./finetune_stgcn.sh
./finetune_ctrgcn.sh
./finetune_tegcn.sh
./finetune_sht.sh
```

## What Each Script Does

1. **Checks for pretrained model**
   - If not found, trains the model first
   - ST-GCN uses MMACTION2 pretrained weights if available
   - SHT can use Hyperformer pretrained weights

2. **Checks for augmented dataset**
   - If not found, generates synthetic "Waiting_To_Cross" samples
   - Creates balanced dataset with ~2,000 synthetic samples

3. **Runs fine-tuning**
   - 30 epochs
   - Learning rate: 0.0001
   - Batch size: 32
   - Uses weighted loss for class imbalance

4. **Saves best model**
   - Saves to `work_dirs/gcn_finetuning/<model>/best_finetuned_model.pth`

## Prerequisites

Before running the scripts, ensure you have:

1. **Original data**: `./gcn_per_pedestrian/data.npy` and `labels.npy`
2. **Python environment**: With PyTorch and required dependencies
3. **Pretrained weights** (optional but recommended):
   - ST-GCN: `pretrained_weights/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth`
   - SHT: `pretrained_weights/hyperformer/hyperformer_pretrained_weights/ntu60/csub/joint/runs-140-87640.pt`

## Expected Output

After running each script, you'll find:

```
work_dirs/gcn_finetuning/
├── stgcn/
│   ├── best_finetuned_model.pth
│   ├── final_finetuned_model.pth
│   └── finetuning_history.npy
├── ctrgcn/
│   └── ...
├── tegcn/
│   └── ...
└── sht/
    └── ...
```

## Evaluate Results

After fine-tuning, evaluate each model:

```bash
# ST-GCN
python gcn_eval.py --model stgcn --checkpoint work_dirs/gcn_finetuning/stgcn/best_finetuned_model.pth

# CTR-GCN
python gcn_eval.py --model ctrgcn --checkpoint work_dirs/gcn_finetuning/ctrgcn/best_finetuned_model.pth

# TE-GCN
python gcn_eval.py --model tegcn --checkpoint work_dirs/gcn_finetuning/tegcn/best_finetuned_model.pth

# SHT
python gcn_eval.py --model sht --checkpoint work_dirs/gcn_finetuning/sht/best_finetuned_model.pth
```

## Troubleshooting

### Script fails with "model not found"
- Train the model first: `python gcn_train.py --model <model> --epochs 50`
- Or check if pretrained weights path is correct

### Script fails with "augmented data not found"
- The script will auto-generate it, but you can manually run:
  ```bash
  python gcn_fetch_augment_data.py --synthetic_samples 2000
  ```

### SHT script fails
- Ensure Hyperformer dependencies are installed
- Check that `model/hyperformer.py` exists
- Verify pretrained weights are in the correct location

### CUDA out of memory
- Edit the batch script and change `--batch_size 32` to `--batch_size 16`
- Or add `--device cpu` to use CPU

## Customization

To customize fine-tuning parameters, edit the batch script and modify:

```bash
python gcn_finetune.py \
    --model <model> \
    --pretrained_path <path> \
    --epochs 30 \          # Change number of epochs
    --lr 0.0001 \         # Change learning rate
    --batch_size 32 \     # Change batch size
    --freeze_backbone     # Add to freeze backbone
```
