"""PyTorch Dataset for skeleton clips produced from RTMPose tracks.

Exports `SkeletonDataset` which scans a tracks directory (json files produced
by `rtmpose_to_skeleton.process_scenes`) and yields clips shaped (C,T,V,1).
Normalization centers on mid-hip (avg of `RightUpLeg` and `LeftUpLeg`) and
scales by shoulder distance (avg distance between left/right arms) when possible.
"""
from pathlib import Path
import json
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from graphs.custom_coco_graph import get_num_node, body_parts


def _pad_or_trim(frames: List[np.ndarray], T: int) -> List[np.ndarray]:
    if len(frames) >= T:
        return frames[:T]
    # pad by repeating last frame
    if len(frames) == 0:
        return [np.zeros((get_num_node(), 2), dtype=np.float32) for _ in range(T)]
    out = frames[:] + [frames[-1]] * (T - len(frames))
    return out


def normalize_clip(clip: np.ndarray) -> np.ndarray:
    # clip: (T, V, C) where C is 2 or 3
    T, V, C = clip.shape
    # find mid-hip: average of RightUpLeg and LeftUpLeg if available
    try:
        r_hip = body_parts.index('RightUpLeg')
        l_hip = body_parts.index('LeftUpLeg')
        mid = (clip[:, r_hip, :2] + clip[:, l_hip, :2]) / 2.0
    except ValueError:
        mid = clip[:, 0, :2]

    # subtract mid per frame
    clip[:, :, :2] = clip[:, :, :2] - mid[:, None, :]

    # scale by shoulder distance if possible
    try:
        r_sh = body_parts.index('RightArm')
        l_sh = body_parts.index('LeftArm')
        shoulder_dist = np.linalg.norm(clip[:, r_sh, :2] - clip[:, l_sh, :2], axis=1)
        scale = np.maximum(shoulder_dist.mean(), 1e-6)
    except ValueError:
        scale = 1.0

    clip[:, :, :2] = clip[:, :, :2] / scale
    return clip


class SkeletonDataset(Dataset):
    def __init__(self, tracks_dir: str, temporal_length: int = 30, stride: int = 15, in_channels: int = 2):
        self.tracks_dir = Path(tracks_dir)
        self.T = temporal_length
        self.stride = stride
        self.in_channels = in_channels

        self.samples = []  # list of (track_json_path, track_id, start_idx)
        self._scan()

    def _scan(self):
        for p in self.tracks_dir.rglob('*_tracks.json'):
            with open(p, 'r') as f:
                data = json.load(f)
            tracks = data.get('tracks', {})
            for tid, frames in tracks.items():
                # frames is a list of dicts with 'frame_index' and 'keypoints'
                num = len(frames)
                if num == 0:
                    continue
                # create sliding windows
                for start in range(0, max(1, num - 0), self.stride):
                    self.samples.append((str(p), str(tid), int(start)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, tid, start = self.samples[idx]
        with open(p, 'r') as f:
            data = json.load(f)
        frames = data['tracks'][tid]

        # extract keypoint arrays
        kps = [np.array(f['keypoints'], dtype=np.float32) for f in frames]
        kps = _pad_or_trim(kps[start:start + self.T], self.T)
        clip = np.stack(kps, axis=0)  # (T, V, C?)

        # ensure channels
        if clip.ndim == 2:
            clip = clip[:, :, None]

        # if input requested is 2 channels and we have 3, drop third
        if self.in_channels == 2 and clip.shape[2] >= 2:
            clip = clip[:, :, :2]
        elif self.in_channels == 3 and clip.shape[2] == 2:
            # add zero channel (e.g., depth/confidence)
            z = np.zeros((clip.shape[0], clip.shape[1], 1), dtype=np.float32)
            clip = np.concatenate([clip, z], axis=2)

        clip = normalize_clip(clip)

        # convert to (C, T, V, 1)
        C = clip.shape[2]
        T = clip.shape[0]
        V = clip.shape[1]
        arr = clip.transpose(2, 0, 1)  # C, T, V
        arr = arr[:, :, :, None]
        return torch.from_numpy(arr).float()


if __name__ == '__main__':
    # quick smoke
    ds = SkeletonDataset('CSPNXT_RTMPose_ProcessedScenes', temporal_length=10)
    print(len(ds))
    if len(ds) > 0:
        x = ds[0]
        print(x.shape)
