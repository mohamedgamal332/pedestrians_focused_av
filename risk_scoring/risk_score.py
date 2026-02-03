"""Risk scoring utilities for pose tracks.

Provides a heuristic `compute_person_risk` based on distance to camera,
motion toward camera, and predicted action. Robust to skeleton inputs of
shape (T, V, 2) or (T, V, 3) where the third channel may be depth.

Usage example:
    from scripts.risk_score import compute_person_risk
    risk = compute_person_risk(skel, 'Walking')
"""
from typing import Optional, Sequence, List, Dict
import numpy as np

# Default action weights (tunable)
ACTION_WEIGHTS: Dict[str, float] = {
    "Standing": 0.2,
    "Walking": 0.5,
    "Running": 0.8,
    "Waving": 0.3,
    "Waving+Walking": 0.6,
}


def _ensure_numpy(skeleton_sequence: Sequence) -> np.ndarray:
    arr = np.asarray(skeleton_sequence, dtype=float)
    if arr.ndim != 3:
        raise ValueError('skeleton_sequence must be shape (T, V, 2|3)')
    if arr.shape[2] not in (2, 3):
        raise ValueError('last dim must be 2 (x,y) or 3 (x,y,z)')
    return arr


def _mid_hip_position(skel: np.ndarray, hip_idx0: int = 0, hip_idx1: int = 1) -> np.ndarray:
    # skel: (T, V, C)
    return (skel[:, hip_idx0, :2] + skel[:, hip_idx1, :2]) / 2.0


def compute_person_risk(
    skeleton_sequence: Sequence,
    predicted_action: str,
    action_weights: Optional[Dict[str, float]] = None,
    max_distance: float = 10.0,
    hip_idx0: int = 0,
    hip_idx1: int = 1,
    wD: float = 0.5,
    wA: float = 0.3,
    wM: float = 0.2,
) -> float:
    """Compute a heuristic risk score in [0,1] for a single person clip.

    Parameters
    - skeleton_sequence: array-like (T, V, 2) or (T, V, 3). Coordinates should
      be in camera-relative units if available (meters). If only pixel coords
      are present this still produces a normalized score but `max_distance`
      should be tuned accordingly.
    - predicted_action: action label string (used to look up weight)
    - action_weights: optional override for action weights
    - max_distance: normalization distance in same units as positions
    - hip_idx0/hip_idx1: indices for the two hip joints used to compute mid-hip
    - wD, wA, wM: weights for distance, action, motion (must sum <=1)

    Returns
    - risk_score: float in [0,1]
    """
    arr = _ensure_numpy(skeleton_sequence)
    T = arr.shape[0]
    if T < 2:
        # not enough frames to compute motion; fallback to static risk based on distance+action
        mid_hip = _mid_hip_position(arr, hip_idx0, hip_idx1)
        distances = np.linalg.norm(mid_hip, axis=1)
        distance_factor = float(np.clip(1 - distances.mean() / max_distance, 0.0, 1.0))
        aw = (action_weights or ACTION_WEIGHTS).get(predicted_action, 0.5)
        risk = wD * distance_factor + wA * aw
        return float(np.clip(risk, 0.0, 1.0))

    # 1) mid-hip 2D positions per frame
    mid_hip = _mid_hip_position(arr, hip_idx0, hip_idx1)  # (T,2)

    # 2) distance factor: closer -> higher risk
    distances = np.linalg.norm(mid_hip, axis=1)  # (T,)
    distance_factor = np.clip(1 - distances / float(max_distance), 0.0, 1.0)

    # 3) motion toward camera factor
    velocities = np.diff(mid_hip, axis=0)  # (T-1,2)
    vectors_to_camera = -mid_hip[:-1, :]  # camera at origin

    dot = np.sum(velocities * vectors_to_camera, axis=1)
    vel_norm = np.linalg.norm(velocities, axis=1) + 1e-8
    cam_norm = np.linalg.norm(vectors_to_camera, axis=1) + 1e-8
    cos_theta = dot / (vel_norm * cam_norm)
    # consider only motion that has positive projection toward camera
    motion_factor = np.clip(cos_theta, 0.0, 1.0)
    motion_factor_mean = float(np.nanmean(motion_factor))

    # 4) action factor
    aw = (action_weights or ACTION_WEIGHTS).get(predicted_action, 0.5)

    # 5) combine
    risk_score = wD * float(distance_factor.mean()) + wA * aw + wM * motion_factor_mean
    risk_score = float(np.clip(risk_score, 0.0, 1.0))
    return risk_score


def compute_scene_risks(
    skeletons: Sequence[Sequence],
    predicted_actions: Sequence[str],
    **kwargs,
) -> List[float]:
    """Compute risk per-person for a list of skeleton sequences and labels.

    skeletons: list-like of skeleton_sequence (T_i, V, 2|3)
    predicted_actions: list-like of strings, same length as skeletons
    kwargs passed to `compute_person_risk`
    """
    if len(skeletons) != len(predicted_actions):
        raise ValueError('skeletons and predicted_actions must have same length')
    return [compute_person_risk(s, a, **kwargs) for s, a in zip(skeletons, predicted_actions)]


if __name__ == '__main__':
    # basic smoke example
    import sys
    example = np.random.rand(16, 12, 2) * 10 - 5
    r = compute_person_risk(example, 'Walking')
    print('risk', r)
