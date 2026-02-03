@echo off
REM Train SHT/Hyperformer from scratch (no pretrained path) on augmented data

REM Generate augmented data if missing
if not exist "gcn_per_pedestrian_augmented\data.npy" (
    echo Augmented dataset not found. Generating...
    python gcn_fetch_augment_data.py --original_data_dir ./gcn_per_pedestrian --output_dir ./gcn_per_pedestrian_augmented --synthetic_samples 2000
)

REM Train SHT from scratch (no --pretrained_path)
echo Starting SHT/Hyperformer training from scratch...
python gcn_train.py --model sht --data_dir ./gcn_per_pedestrian_augmented --epochs 50 --batch_size 32

echo Training complete!
echo Evaluate with:
echo   python gcn_eval.py --model sht --checkpoint work_dirs\gcn_training\sht\best_model.pth --data_dir ./gcn_per_pedestrian_augmented

pause
