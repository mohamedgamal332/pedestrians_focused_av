#!/usr/bin/env python3
"""
Convert CARLA MMPose dataset to GCN format per pedestrian.

Each pedestrian is treated as one sample.
Uses COCO 17 keypoint format (no neck joint needed).

Outputs:
    - data.npy : shape [N_samples, C, T, V] where V=17 (COCO keypoints)
    - labels.npy : shape [N_samples] (BehaviorState as integer)
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataloader import CARLAStereoPedestrianDataset, BehaviorState  # adjust import if needed

# -----------------------------
# Parse command-line arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Generate GCN input per pedestrian from CARLA dataset")
parser.add_argument("--session_path", type=str, default="/home/theta/carla/output/sessions/session_20260124_200032")
parser.add_argument("--output_dir", type=str, default="./gcn_per_pedestrian")
parser.add_argument("--camera", type=str, default="left", choices=["left", "right"])
parser.add_argument("--seq_length", type=int, default=30)
parser.add_argument("--min_keypoints", type=int, default=5)
parser.add_argument("--max_frames", type=int, default=None)
args = parser.parse_args()

SESSION_PATH = args.session_path
OUTPUT_DIR = Path(args.output_dir)
CAMERA = args.camera
SEQ_LENGTH = args.seq_length
MIN_KEYPOINTS = args.min_keypoints
MAX_FRAMES = args.max_frames

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
dataset = CARLAStereoPedestrianDataset(
    SESSION_PATH,
    load_images=False,
    load_depth=False,
    filter_min_visible_keypoints=MIN_KEYPOINTS,
    max_frames=MAX_FRAMES
)

print(f"Processing {len(dataset)} frames...")

# -----------------------------
# Build per-pedestrian GCN input (COCO 17 keypoints)
# -----------------------------
X_list = []
y_list = []

for start_idx in tqdm(range(0, len(dataset) - SEQ_LENGTH + 1)):
    # Gather all pedestrians present in the first frame of the sequence
    first_frame = dataset[start_idx]
    peds_in_first_frame = first_frame.annotation.get_visible_pedestrians(
        camera=CAMERA,
        min_keypoints=MIN_KEYPOINTS
    )

    for ped in peds_in_first_frame:
        # For each pedestrian, collect their keypoints over SEQ_LENGTH frames
        seq_data = np.zeros((3, SEQ_LENGTH, 17), dtype=np.float32)  # COCO 17 keypoints
        valid_sequence = True

        for t in range(SEQ_LENGTH):
            frame = dataset[start_idx + t]
            # Try to find the same pedestrian by ID
            ped_frame_list = [p for p in frame.annotation.get_visible_pedestrians(CAMERA, MIN_KEYPOINTS)
                              if p.id == ped.id]
            if not ped_frame_list:
                valid_sequence = False
                break  # pedestrian not visible in this frame
            ped_t = ped_frame_list[0]
            seq_data[:, t, :] = ped_t.get_camera_relative_positions(camera=CAMERA).T

        if valid_sequence:
            X_list.append(seq_data)
            y_list.append(list(BehaviorState).index(ped.behavior))

# -----------------------------
# Save sequences (COCO 17 format)
# -----------------------------
if X_list:
    X = np.stack(X_list, axis=0)  # [N_samples, C, T, 17]
    y = np.array(y_list, dtype=np.int64)

    # Save
    np.save(OUTPUT_DIR / "data.npy", X)
    np.save(OUTPUT_DIR / "labels.npy", y)

    print(f"\nSaved {len(X)} pedestrian sequences to {OUTPUT_DIR}")
    print(f"Data shape: {X.shape}, Labels shape: {y.shape}")
    print(f"Format: COCO 17 keypoints (no neck joint)")
else:
    print("\nNo valid pedestrian sequences found.")
