**Train Mamba Trajectory — scripts/train_mamba_trajectory.py**

- **Purpose:**: Train a Mamba SSM model to predict pedestrian trajectory deltas (ΔC_x, ΔC_y, ΔC_z) from per-person state sequences saved by the preprocessing pipeline.

- **File:**: [scripts/train_mamba_trajectory.py](scripts/train_mamba_trajectory.py)

**Model**
- **Architecture:**: Input Linear → Mamba SSM → Output Linear Head projecting to 3 × prediction length.
- **Mamba configuration:**: `Mamba(d_model=d_model, d_state=64, d_conv=4, expand=2)`; default `d_model=128` used in the script.
- **Prediction target:**: next `PRED_LEN=8` steps of ΔC = (ΔC_x, ΔC_y, ΔC_z), produced from the last encoder hidden state.

**Data format / expected inputs**
- **Source files:**: per-scene/person state JSONs located under `hrnet_depth` (e.g. files named `*_states.json`). See [hrnet_depth](hrnet_depth) for examples.
- **Per-frame person entry:**: each `people` entry must include `person_id` and `state` where `state` is an 8-dim vector `[Cx, Cy, Cz, Vx, Vy, Vz, w, h]`.
- **Sequence construction:**: sequences are grouped by `person_id`, split at frame gaps, and require contiguous frames of length `input=16 + pred=8` to produce one sample.

**Normalization / scaling**
- **Image scaling applied:**: X coordinates and related fields are scaled by `IMG_W=1928.88`; Y coordinates scaled by `IMG_H=881.76` (from dataset statistics).
- **State normalization:**: after image scaling, per-dimension mean/std are computed and used to normalize input states and target deltas for training.
- **Evaluation:**: predictions are un-normalized back to pixel units for ADE/FDE reporting (x/y restored using `IMG_W/IMG_H`).

**Training / Loss**
- **Loss:**: MSELoss between predicted normalized ΔC and ground-truth normalized ΔC.
- **Optimizer:**: Adam with configurable `--lr`.
- **Default run:**: `python scripts/train_mamba_trajectory.py` (defaults: CPU, 20 epochs, batch size 64). Adjust `--epochs` and `--batch-size` as needed.

**Evaluation metrics**
- **ADE:**: average displacement error over prediction horizon (pixels after un-normalizing).
- **FDE:**: final displacement error at last predicted timestep (pixels after un-normalizing).

**Quick commands**
- Train 1 epoch (CPU, small batch):

```
python scripts/train_mamba_trajectory.py --epochs 1 --batch-size 8 --device cpu --save temp_mamba.pt
```

- Train 8 epochs (example):

```
python scripts/train_mamba_trajectory.py --epochs 8 --batch-size 8 --device cpu --save mamba_traj.pt
```

**Notes & suggestions**
- The current large ADE/FDE values are explained by pixel-scale units (Cx std ≈ 420 px). For metric errors, convert pixel centers to meters using camera intrinsics + depth.
- Normalizing x/y by image width/height improved training stability; consider converting Cz to meters if intrinsics are available.
- For small datasets, keep batch size small (8–16). Increase epochs (≥8) and monitor ADE/FDE to check convergence.

**Contact**
- For modifications to the Mamba parameters or input preprocessing, edit [scripts/train_mamba_trajectory.py](scripts/train_mamba_trajectory.py).
