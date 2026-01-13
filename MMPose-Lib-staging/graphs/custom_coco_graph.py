"""Custom reduced-COCO skeleton graph utilities.

This module tries to reuse mappings from `save_mmpose_results.py` when available.
It exposes functions to return adjacency and edge lists for GCNs.
"""
from typing import List, Tuple
import numpy as np

try:
    # reuse mapping if present in workspace
    from save_mmpose_results import body_parts, bones_idx
except Exception:
    # fallback minimal 12-joint mapping used in this repo
    body_parts = [
        'RightArm','LeftArm',
        'RightForeArm','LeftForeArm',
        'RightHand','LeftHand',
        'RightUpLeg','LeftUpLeg',
        'RightLeg','LeftLeg',
        'RightFoot','LeftFoot'
    ]
    # bones_idx: name -> [start_idx, end_idx, length_est]
    bones = {
        'RightArm'      : ['RightArm'       , 'RightForeArm', 0.5],
        'LeftArm'       : ['LeftArm'        , 'LeftForeArm' , 0.5],
        'RightForeArm'  : ['RightForeArm'   , 'RightHand'   , 0.5],
        'LeftForeArm'   : ['LeftForeArm'    , 'LeftHand'    , 0.5],
        'RightUpLeg'    : ['RightUpLeg'     , 'RightLeg'    , 0.5],
        'LeftUpLeg'     : ['LeftUpLeg'      , 'LeftLeg'     , 0.5],
        'RightLeg'      : ['RightLeg'       , 'RightFoot'   , 0.5],
        'LeftLeg'       : ['LeftLeg'        , 'LeftFoot'    , 0.5]
    }
    bones_idx = {i: [body_parts.index(s), body_parts.index(e), l] for i, (s, e, l) in bones.items()}


def get_num_node() -> int:
    return len(body_parts)


def self_links() -> List[Tuple[int, int]]:
    return [(i, i) for i in range(get_num_node())]


def neighbor_links() -> List[Tuple[int, int]]:
    # Use bones_idx to construct undirected neighbor edges
    edges = []
    for _, (s, e, _) in bones_idx.items():
        edges.append((s, e))
        edges.append((e, s))
    return edges


def adjacency_matrix(normalize: bool = True) -> np.ndarray:
    """Return adjacency matrix A of shape (V,V).

    If normalize True, perform symmetric normalization D^-1/2 A D^-1/2.
    """
    V = get_num_node()
    A = np.zeros((V, V), dtype=np.float32)
    for i, j in self_links():
        A[i, j] = 1.0
    for i, j in neighbor_links():
        A[i, j] = 1.0

    if not normalize:
        return A

    deg = A.sum(axis=1)
    deg[deg == 0] = 1
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm


__all__ = [
    'body_parts', 'bones_idx', 'get_num_node', 'self_links', 'neighbor_links', 'adjacency_matrix'
]
