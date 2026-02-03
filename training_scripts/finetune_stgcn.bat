@echo off
REM Fine-tune ST-GCN from checkpoint_epoch_50.pth on augmented data

REM Generate augmented data if missing
if not exist "gcn_per_pedestrian_augmented\data.npy" (
    echo Augmented dataset not found. Generating...
    python gcn_fetch_augment_data.py --original_data_dir ./gcn_per_pedestrian --output_dir ./gcn_per_pedestrian_augmented --synthetic_samples 2000
)

REM Fine-tune ST-GCN from checkpoint_epoch_50.pth
echo Starting ST-GCN fine-tuning from checkpoint_epoch_50.pth...
python gcn_train.py --model stgcn --data_dir ./gcn_per_pedestrian_augmented --epochs 50 --batch_size 32 --pretrained_path "C:\Users\samso\Downloads\RTMPoseFinetuning\work_dirs\gcn_training\stgcn\checkpoint_epoch_50.pth"

echo Training complete!
echo Evaluate with:
echo   python gcn_eval.py --model stgcn --checkpoint work_dirs\gcn_training\stgcn\best_model.pth --data_dir ./gcn_per_pedestrian_augmented

pause
