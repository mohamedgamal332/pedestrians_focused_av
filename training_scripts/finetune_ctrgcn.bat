@echo off
REM Train CTR-GCN from scratch (no pretrained path) on augmented data

REM Generate augmented data if missing
if not exist "gcn_per_pedestrian_augmented\data.npy" (
    echo Augmented dataset not found. Generating...
    python gcn_fetch_augment_data.py --original_data_dir ./gcn_per_pedestrian --output_dir ./gcn_per_pedestrian_augmented --synthetic_samples 2000
)

REM Train CTR-GCN from scratch (no --pretrained_path)
echo Starting CTR-GCN training from scratch...
python gcn_train.py --model ctrgcn --data_dir ./gcn_per_pedestrian_augmented --epochs 50 --batch_size 32

echo Training complete!
echo Evaluate with:
echo   python gcn_eval.py --model ctrgcn --checkpoint work_dirs\gcn_training\ctrgcn\best_model.pth --data_dir ./gcn_per_pedestrian_augmented

pause
