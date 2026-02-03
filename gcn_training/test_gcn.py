#!/usr/bin/env python3
"""
Test ST-GCN input by visualizing sequences as individual images per frame.

Saves one PNG per frame for each pedestrian sequence.
"""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # headless backend for Linux servers
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
DATA_DIR = Path("stgcn_test")  # folder containing data.npy and labels.npy
OUTPUT_DIR = Path("stgcn_test_images")
OUTPUT_DIR.mkdir(exist_ok=True)

# Define skeleton edges for COCO/17 keypoints (used in ST-GCN)
# (start, end) pairs of keypoint indices
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (11,12),(11,13),(13,15),(12,14),(14,16)
]

# -----------------------------
# Load data
# -----------------------------
X = np.load(DATA_DIR / "data.npy")  # [N_samples, C, T, V]
y = np.load(DATA_DIR / "labels.npy")

print(f"Loaded {X.shape[0]} sequences, each with {X.shape[2]} frames and {X.shape[3]} keypoints")

# -----------------------------
# Plot each sequence
# -----------------------------
for seq_idx in range(X.shape[0]):
    seq = X[seq_idx]  # shape [C, T, V]
    for t in range(seq.shape[1]):
        fig, ax = plt.subplots()
        x = seq[0, t, :]
        y_coord = seq[1, t, :]
        z = seq[2, t, :]  # optional: can use for color/size if needed

        # Plot keypoints
        ax.scatter(x, y_coord, c='r', s=50)

        # Draw skeleton lines
        for start, end in SKELETON:
            if start < len(x) and end < len(x):
                ax.plot([x[start], x[end]], [y_coord[start], y_coord[end]], 'b-', linewidth=2)

        ax.set_title(f"Sequence {seq_idx} Frame {t} | Label: {y[seq_idx]}")
        ax.invert_yaxis()  # match typical image coordinate system
        ax.axis('equal')
        ax.axis('off')

        out_path = OUTPUT_DIR / f"seq{seq_idx:03d}_frame{t:03d}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

print(f"Saved images to {OUTPUT_DIR}, {X.shape[0]*X.shape[1]} frames total.")
