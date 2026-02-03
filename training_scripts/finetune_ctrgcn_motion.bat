@echo off
REM Train CTR-GCN Motion from scratch, then fine-tune on augmented data (Windows)

REM Use project venv if present (so pip/python use it when run from Explorer or new terminal)
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

REM Avoid OpenMP duplicate libiomp5md.dll error (PyTorch + MKL on Windows)
set KMP_DUPLICATE_LIB_OK=TRUE

REM PyTorch needs typing_extensions (4.5+) and sympy (for torch._dynamo)
pip install --upgrade typing_extensions sympy --quiet

REM Train CTR-GCN Motion from scratch (no pretrained checkpoint)
echo Training CTR-GCN Motion from scratch...
python gcn_train.py --model ctrgcn_motion --epochs 50 --batch_size 32
set PRETRAINED_PATH=work_dirs\gcn_training\ctrgcn_motion\best_model.pth

REM Check if augmented data exists
if not exist "gcn_per_pedestrian_augmented\data.npy" (
    echo Warning: Augmented dataset not found. Generating...
    python gcn_fetch_augment_data.py --original_data_dir ./gcn_per_pedestrian --output_dir ./gcn_per_pedestrian_augmented --synthetic_samples 2000
)

REM Fine-tune CTR-GCN Motion
echo Starting CTR-GCN Motion fine-tuning...
python gcn_finetune.py --model ctrgcn_motion --pretrained_path "%PRETRAINED_PATH%" --original_data_dir ./gcn_per_pedestrian --augmented_data_dir ./gcn_per_pedestrian_augmented --epochs 30 --lr 0.0001 --batch_size 32

echo Fine-tuning complete!
echo Evaluate with:
echo   python gcn_eval.py --model ctrgcn_motion --checkpoint work_dirs\gcn_finetuning\ctrgcn_motion\best_finetuned_model.pth

pause
