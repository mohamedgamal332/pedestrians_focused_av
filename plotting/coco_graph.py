"""
Custom COCO-18 graph for ST-GCN using CARLA 17-keypoint sequences.

- Infers missing `neck` joint from shoulders.
- Provides adjacency matrix for Kinetics-STGCN pretrained model.
"""

from typing import List, Tuple
import numpy as np

# -----------------------------
# COCO-17 body parts (CARLA dataset)
# -----------------------------
coco17_parts = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

# -----------------------------
# COCO-18 body parts (ST-GCN Kinetics)
# -----------------------------
body_parts_18 = coco17_parts + ["neck"]

# Edges for COCO-18
edges = [
    # head
    (0,1), (0,2), (1,3), (2,4),
    (0,17),        # nose -> neck
    # torso
    (17,5), (17,6), # neck -> shoulders
    (5,6), (5,11), (6,12), (11,12),
    # left arm
    (5,7), (7,9),
    # right arm
    (6,8), (8,10),
    # left leg
    (11,13), (13,15),
    # right leg
    (12,14), (14,16)
]
# Make bidirectional
edges += [(j,i) for i,j in edges]

# -----------------------------
# Graph utilities
# -----------------------------
def get_num_node() -> int:
    return len(body_parts_18)

def self_links() -> List[Tuple[int,int]]:
    return [(i,i) for i in range(get_num_node())]

def neighbor_links() -> List[Tuple[int,int]]:
    return edges

def adjacency_matrix(normalize: bool = True) -> np.ndarray:
    V = get_num_node()
    A = np.zeros((V,V), dtype=np.float32)
    
    for i,j in self_links():
        A[i,j] = 1.0
    for i,j in neighbor_links():
        A[i,j] = 1.0

    if not normalize:
        return A
    
    deg = A.sum(axis=1)
    deg[deg==0] = 1
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm

# -----------------------------
# Add inferred neck joint to 17-keypoint sequences
# -----------------------------
def add_neck_joint(X17: np.ndarray) -> np.ndarray:
    """
    Input:
        X17: [N, C, T, 17] tensor
    Output:
        X18: [N, C, T, 18] tensor with inferred neck joint
    """
    assert X17.shape[-1] == 17, "Input must have 17 keypoints"
    
    # indices of left/right shoulders
    l_sh_idx = coco17_parts.index("left_shoulder")
    r_sh_idx = coco17_parts.index("right_shoulder")
    
    # Compute neck as midpoint between shoulders
    neck = 0.5 * (X17[..., l_sh_idx] + X17[..., r_sh_idx])
    neck = neck[..., np.newaxis]  # shape [N,C,T,1]
    
    # Concatenate as joint 17
    X18 = np.concatenate([X17, neck], axis=-1)
    return X18

# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":
    N, C, T = 2, 2, 30
    X17 = np.random.randn(N, C, T, 17).astype(np.float32)
    X18 = add_neck_joint(X17)
    A18 = adjacency_matrix()
    print("Original shape:", X17.shape)
    print("After adding neck:", X18.shape)
    print("Adjacency matrix shape:", A18.shape)
