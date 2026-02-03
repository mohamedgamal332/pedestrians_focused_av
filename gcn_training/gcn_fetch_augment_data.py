#!/usr/bin/env python3
"""
Fetch and prepare augmented skeleton dataset (10 classes, risk analysis).
Supports: synthetic generators (Waiting_To_Cross, Waving, Texting, etc.),
NTU60/Figshare download, Kinetics/UCF101 loaders, keypoint conversion (NTU 25 -> COCO 17).
Output format: COCO 17 keypoints, 30 frames, same C as input (2 or 3).
"""
import numpy as np
import argparse
from pathlib import Path
import sys
from collections import defaultdict

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

# -----------------------------
# Dataset info
# -----------------------------
DATASETS = {
    'ntu60': {
        'name': 'NTU RGB+D 60',
        'url': 'https://figshare.com/ndownloader/articles/27427188',
        'keypoints': 25,
        'note': 'Use --download_ntu60 or --ntu60_dir with .npz',
    },
    'kinetics-skeleton': {'name': 'Kinetics-Skeleton', 'keypoints': 18},
    'ucf101-skeleton': {'name': 'UCF101-Skeleton', 'keypoints': 18},
}

NTU60_FIGSHARE_ARTICLE_URL = "https://figshare.com/ndownloader/articles/27427188/versions/1"
NTU60_DEFAULT_DOWNLOAD_DIR = "data/ntu60"

# -----------------------------
# 10-class behavior names (match BehaviorState in dataloader.py)
# -----------------------------
BEHAVIOR_CLASS_NAMES = [
    'Walking', 'Running', 'Crossing', 'Waiting_To_Cross', 'Idle',
    'Waving', 'Waving_Walking', 'Texting', 'Calling', 'Talking'
]
NUM_BEHAVIOR_CLASSES = 10

# -----------------------------
# ACTION_TO_BEHAVIOR (for external dataset ingestion)
# -----------------------------
ACTION_TO_BEHAVIOR = {
    'walk': 0, 'walking': 0, 'run': 1, 'running': 1, 'cross': 2, 'crossing': 2,
    'wait': 3, 'waiting': 3, 'stand': 3, 'standing': 3, 'idle': 3, 'pause': 3, 'stop': 3,
    'wave': 5, 'waving': 5, 'waving_walking': 6, 'walk_wave': 6,
    'text': 7, 'texting': 7, 'phone': 7, 'call': 8, 'calling': 8, 'phone_call': 8,
    'talk': 9, 'talking': 9, 'conversation': 9, 'speak': 9,
}

# -----------------------------
# NTU 60 action names and mapping to our 10 classes
# -----------------------------
NTU60_ACTION_NAMES = [
    'drink water', 'eat meal', 'brush teeth', 'brush hair', 'drop', 'pickup', 'throw',
    'sitting down', 'standing up', 'clapping', 'reading', 'writing', 'tear up paper',
    'wear jacket', 'take off jacket', 'wear a shoe', 'take off a shoe', 'wear on glasses',
    'take off glasses', 'put on a hat', 'take off a hat', 'cheer up', 'hand waving',
    'kicking something', 'reach into pocket', 'hopping', 'jump up', 'make a phone call',
    'playing with phone or tablet', 'typing on a keyboard', 'pointing to something',
    'taking a selfie', 'check time', 'rub two hands together', 'nod head', 'shake head',
    'wipe face', 'salute', 'put the palms together', 'cross hands in front',
    'sneeze or cough', 'staggering', 'falling', 'touch head', 'touch chest', 'touch back',
    'touch neck', 'nausea or vomiting', 'use a fan', 'punching', 'kicking', 'pushing',
    'pat on back', 'point finger', 'hugging', 'giving something', 'touch pocket',
    'handshaking', 'walking towards', 'walking apart',
]
NTU60_ACTION_TO_BEHAVIOR = {
    0: 4, 1: 4, 2: 4, 3: 4, 4: 3, 5: 3, 6: 4, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4, 12: 4,
    13: 4, 14: 4, 15: 4, 16: 4, 17: 4, 18: 4, 19: 4, 20: 4, 21: 5, 22: 5, 23: 1, 24: 4,
    25: 1, 26: 1, 27: 8, 28: 7, 29: 7, 30: 9, 31: 7, 32: 4, 33: 4, 34: 9, 35: 9, 36: 4,
    37: 4, 38: 4, 39: 4, 40: 4, 41: 4, 42: 4, 43: 4, 44: 4, 45: 4, 46: 4, 47: 4, 48: 4,
    49: 1, 50: 1, 51: 4, 52: 9, 53: 9, 54: 9, 55: 9, 56: 4, 57: 9, 58: 0, 59: 0,
}

# -----------------------------
# Keypoint mapping: NTU 25 -> COCO 17
# -----------------------------
NTU25_TO_COCO17 = {
    0: None, 1: 0, 2: None, 3: 2, 4: 5, 5: 7, 6: 9, 7: 11, 8: 6, 9: 8, 10: 10, 11: 12,
    12: 12, 13: 14, 14: 16, 15: 11, 16: 13, 17: 15, 18: None, 19: None, 20: None, 21: None, 22: None, 23: None, 24: None,
}

# -----------------------------
# Keypoint mapping: OpenPose 18 -> COCO 17
# -----------------------------
OPENPOSE18_TO_COCO17 = {
    0: 0, 1: None, 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9, 8: 12, 9: 14, 10: 16, 11: 11, 12: 13, 13: 15, 14: 2, 15: 1, 16: 4, 17: 3,
}

# -----------------------------
# Download helpers
# -----------------------------
def download_ntu60_figshare(download_dir, skip_if_exists=True):
    if not HAS_REQUESTS:
        print("Install requests for download: pip install requests")
        return None
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    output_path = download_dir / "ntu60_figshare.zip"
    if skip_if_exists and output_path.exists():
        print(f"NTU60 already exists: {output_path}")
        return output_path
    print("Downloading NTU RGB+D 60 skeleton from Figshare (~12 GB)...")
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; GCN-augment/1.0)"})
        resp = session.get(NTU60_FIGSHARE_ARTICLE_URL, stream=True, allow_redirects=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        chunk_size = 1024 * 1024
        with open(output_path, "wb") as f, tqdm(desc=output_path.name, total=total, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f"Downloaded: {output_path}")
        return output_path
    except Exception as e:
        print(f"NTU60 download failed: {e}")
        return None


def convert_ntu60_download_to_npz(download_dir, max_samples=None):
    import zipfile
    import csv as csv_module
    download_dir = Path(download_dir)
    zip_path = download_dir / "ntu60_figshare.zip"
    if zip_path.exists():
        print("Extracting zip...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(download_dir)
    csv_files = list(download_dir.glob("**/*.csv"))
    if not csv_files:
        print("No CSV found in download dir.")
        return None
    csv_path = max(csv_files, key=lambda p: p.stat().st_size)
    print(f"Parsing CSV: {csv_path}...")
    data_by_sample = defaultdict(list)
    action_per_sample = {}
    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv_module.reader(f)
            next(reader)
            for row in tqdm(reader, desc="Reading CSV"):
                if len(row) < 4:
                    continue
                try:
                    sample_id, action_id = row[0], int(row[1]) - 1
                    if action_id < 0 or action_id > 59:
                        action_id = 0
                    if sample_id not in action_per_sample:
                        action_per_sample[sample_id] = min(59, max(0, action_id))
                    frame_idx = int(row[2])
                    rest = [float(x) for x in row[3:] if x.strip()]
                    if len(rest) >= 75:
                        coords = np.array(rest[:75], dtype=np.float32).reshape(25, 3)
                    elif len(rest) >= 25 * 3:
                        coords = np.array(rest[: 25 * 3], dtype=np.float32).reshape(25, 3)
                    else:
                        continue
                    data_by_sample[sample_id].append((frame_idx, coords))
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"CSV parse error: {e}")
        return None
    samples_x, sample_ids = [], []
    for sample_id, frames in data_by_sample.items():
        frames.sort(key=lambda t: t[0])
        coords_list = [c for _, c in frames]
        if len(coords_list) < 5:
            continue
        samples_x.append(np.stack(coords_list, axis=0))
        sample_ids.append(sample_id)
    samples_y = [action_per_sample.get(sid, 0) for sid in sample_ids]
    if max_samples and len(samples_x) > max_samples:
        idx = np.random.permutation(len(samples_x))[:max_samples]
        samples_x = [samples_x[i] for i in idx]
        samples_y = [samples_y[i] for i in idx]
    target_t = 50
    X_fixed = []
    for seq in samples_x:
        T, V, C = seq.shape
        if T >= target_t:
            indices = np.linspace(0, T - 1, target_t).astype(int)
            X_fixed.append(seq[indices])
        else:
            pad = np.tile(seq[-1:], (target_t - T, 1, 1))
            X_fixed.append(np.concatenate([seq, pad], axis=0))
    X = np.stack(X_fixed, axis=0).astype(np.float32)
    y = np.array(samples_y, dtype=np.int64)
    out_npz = download_dir / "ntu60_preprocessed.npz"
    np.savez_compressed(out_npz, x=X, y=y)
    print(f"Saved {X.shape[0]} samples to {out_npz}")
    return out_npz


# -----------------------------
# Keypoint conversion and resampling
# -----------------------------
def convert_to_coco17(keypoints, source_format='ntu25'):
    if source_format == 'coco17':
        return keypoints
    orig = keypoints.shape
    if len(orig) == 2:
        keypoints = keypoints[np.newaxis, :, :]
        squeeze = True
    else:
        squeeze = False
    T, V_source, C = keypoints.shape
    mapping = NTU25_TO_COCO17 if source_format == 'ntu25' else OPENPOSE18_TO_COCO17
    target_V = 17
    out = np.zeros((T, target_V, C), dtype=keypoints.dtype)
    for src_idx, tgt_idx in mapping.items():
        if tgt_idx is not None and src_idx < V_source:
            out[:, tgt_idx, :] = keypoints[:, src_idx, :]
    if squeeze:
        out = out.squeeze(0)
    return out


def resample_temporal(data, target_frames=30):
    if data.ndim == 3:
        data = data[np.newaxis, :, :, :]
        squeeze = True
    else:
        squeeze = False
    N, C, T, V = data.shape
    if T == target_frames:
        return data.squeeze(0) if squeeze else data
    resampled = np.zeros((N, C, target_frames, V), dtype=data.dtype)
    for n in range(N):
        for c in range(C):
            for v in range(V):
                resampled[n, c, :, v] = np.interp(
                    np.linspace(0, T - 1, target_frames),
                    np.arange(T),
                    data[n, c, :, v]
                )
    if squeeze:
        resampled = resampled.squeeze(0)
    return resampled


def _normalize_to_coco17_format(data, source_format, target_frames=30):
    if data.ndim == 3:
        data = data[np.newaxis, ...]
    N, T, V, C = data.shape
    out_list = []
    for n in range(N):
        kp = convert_to_coco17(data[n], source_format=source_format)
        kp = np.transpose(kp, (2, 0, 1))
        kp = resample_temporal(kp, target_frames=target_frames)
        out_list.append(kp)
    return np.array(out_list, dtype=np.float32)


# -----------------------------
# Load external datasets
# -----------------------------
def load_ntu_skeleton_data(path, max_samples_per_behavior=None, target_frames=30, skip_unmapped=True):
    path = Path(path)
    if not path.exists():
        print(f"NTU60 path not found: {path}")
        return None, None
    if path.suffix == '.npz':
        obj = np.load(path, allow_pickle=True)
        raw = obj['x'] if 'x' in obj else obj['data']
        raw_labels = obj['y'] if 'y' in obj else obj['label']
        if raw_labels is None:
            return None, None
        raw_labels = np.atleast_1d(raw_labels).astype(int)
    else:
        npy_files = list(path.glob("**/*.npy"))[:50000]
        if not npy_files:
            return None, None
        raw_list, raw_labels_list = [], []
        for f in tqdm(npy_files, desc="NTU60 load"):
            arr = np.load(f)
            if arr.ndim == 2 and arr.shape[0] == 25:
                arr = arr[np.newaxis, :, :]
            if arr.ndim == 3 and arr.shape[1] == 25:
                raw_list.append(arr)
                try:
                    raw_labels_list.append(int(f.stem.split('A')[-1][:2]) - 1)
                except Exception:
                    raw_labels_list.append(0)
        raw = np.concatenate(raw_list, axis=0)
        raw_labels = np.array(raw_labels_list, dtype=int)
    if raw.ndim == 3:
        raw = raw[:, np.newaxis, :, :]
    behavior_labels = []
    valid_indices = []
    for i in range(len(raw)):
        aid = int(raw_labels[i]) if raw_labels[i] < 60 else 0
        bid = NTU60_ACTION_TO_BEHAVIOR.get(aid)
        if bid is None and skip_unmapped:
            continue
        if bid is None:
            bid = 4
        behavior_labels.append(bid)
        valid_indices.append(i)
    if not valid_indices:
        return None, None
    raw = raw[valid_indices]
    behavior_labels = np.array(behavior_labels, dtype=np.int64)
    if max_samples_per_behavior is not None:
        kept = []
        for b in range(NUM_BEHAVIOR_CLASSES):
            idx = np.where(behavior_labels == b)[0]
            if len(idx) > max_samples_per_behavior:
                idx = np.random.choice(idx, max_samples_per_behavior, replace=False)
            kept.extend(idx.tolist())
        raw = raw[kept]
        behavior_labels = behavior_labels[kept]
    data = _normalize_to_coco17_format(raw, 'ntu25', target_frames=target_frames)
    return data, behavior_labels


def _action_name_to_behavior(name):
    name = name.lower().replace('-', ' ').replace('_', ' ')
    for key, bid in ACTION_TO_BEHAVIOR.items():
        if key in name:
            return bid
    if 'wave' in name:
        return 5
    if 'walk' in name and 'wave' in name:
        return 6
    if 'text' in name or ('phone' in name and 'type' in name):
        return 7
    if 'call' in name or 'phone' in name:
        return 8
    if 'talk' in name or 'speak' in name:
        return 9
    if 'run' in name or 'jog' in name:
        return 1
    if 'walk' in name or 'cross' in name:
        return 0
    if 'stand' in name or 'wait' in name or 'idle' in name:
        return 3
    return 4


def load_kinetics_skeleton_data(path, max_samples_per_behavior=None, target_frames=30):
    path = Path(path)
    if not path.exists() or path.suffix != '.npz':
        return None, None
    obj = np.load(path, allow_pickle=True)
    raw = obj['x'] if 'x' in obj else obj['data']
    if 'action_names' in obj:
        raw_labels = np.array([_action_name_to_behavior(str(n)) for n in np.atleast_1d(obj['action_names'])])
    elif 'y' in obj:
        raw_labels = np.atleast_1d(obj['y']).astype(int)
        raw_labels = np.where((raw_labels >= 0) & (raw_labels <= 9), raw_labels, -1)
    elif 'label' in obj:
        raw_labels = np.atleast_1d(obj['label']).astype(int)
        raw_labels = np.where((raw_labels >= 0) & (raw_labels <= 9), raw_labels, -1)
    else:
        return None, None
    if raw.ndim == 3:
        raw = raw[:, np.newaxis, :, :]
    valid = raw_labels >= 0
    if not np.any(valid):
        return None, None
    raw, raw_labels = raw[valid], raw_labels[valid]
    if max_samples_per_behavior is not None:
        kept = []
        for b in range(NUM_BEHAVIOR_CLASSES):
            idx = np.where(raw_labels == b)[0]
            if len(idx) > max_samples_per_behavior:
                idx = np.random.choice(idx, max_samples_per_behavior, replace=False)
            kept.extend(idx.tolist())
        raw = raw[kept]
        raw_labels = raw_labels[kept]
    data = _normalize_to_coco17_format(raw, 'openpose18', target_frames=target_frames)
    return data, raw_labels.astype(np.int64)


def load_ucf101_skeleton_data(path, max_samples_per_behavior=None, target_frames=30):
    return load_kinetics_skeleton_data(path, max_samples_per_behavior, target_frames)


# -----------------------------
# Synthetic generators (10 classes)
# -----------------------------
def generate_synthetic_waiting_data(base_data, num_samples=1000):
    walking_indices = np.where(base_data['labels'] == 0)[0]
    if len(walking_indices) == 0:
        return None, None
    selected = np.random.choice(walking_indices, size=min(num_samples, len(walking_indices)), replace=True)
    out_data, out_labels = [], []
    for idx in selected:
        sample = base_data['data'][idx].copy()
        C, T, V = sample.shape
        base_tiled = np.tile(sample[:, 0:1, :], (1, T, 1))
        noise = np.random.normal(0, 0.02 * np.std(sample), sample.shape)
        out_data.append(0.85 * base_tiled + 0.15 * (sample + noise))
        out_labels.append(3)
    return np.array(out_data), np.array(out_labels)


def generate_synthetic_waving_data(base_data, num_samples=500):
    walking_indices = np.where(base_data['labels'] == 0)[0]
    if len(walking_indices) == 0:
        return None, None
    selected = np.random.choice(walking_indices, size=min(num_samples, len(walking_indices)), replace=True)
    out_data, out_labels = [], []
    t = np.linspace(0, 4 * np.pi, 30)
    wave = 0.08 * np.sin(t) + 0.04 * np.sin(2 * t)
    for idx in selected:
        sample = base_data['data'][idx].copy()
        C, T, V = sample.shape
        base_tiled = np.tile(sample[:, 0:1, :], (1, T, 1))
        sample = 0.6 * base_tiled + 0.4 * sample
        for c in range(C):
            sample[c, :, 9] += wave + np.random.normal(0, 0.01, T)
            sample[c, :, 10] += wave * 0.7 + np.random.normal(0, 0.01, T)
        out_data.append(sample)
        out_labels.append(5)
    return np.array(out_data), np.array(out_labels)


def generate_synthetic_waving_walking_data(base_data, num_samples=500):
    walking_indices = np.where(base_data['labels'] == 0)[0]
    if len(walking_indices) == 0:
        return None, None
    selected = np.random.choice(walking_indices, size=min(num_samples, len(walking_indices)), replace=True)
    out_data, out_labels = [], []
    t = np.linspace(0, 4 * np.pi, 30)
    wave = 0.06 * np.sin(t)
    for idx in selected:
        sample = base_data['data'][idx].copy()
        C, T, V = sample.shape
        for c in range(C):
            sample[c, :, 9] += wave + np.random.normal(0, 0.008, T)
            sample[c, :, 10] += wave * 0.8 + np.random.normal(0, 0.008, T)
        out_data.append(sample)
        out_labels.append(6)
    return np.array(out_data), np.array(out_labels)


def generate_synthetic_texting_data(base_data, num_samples=500):
    walking_indices = np.where(base_data['labels'] == 0)[0]
    if len(walking_indices) == 0:
        return None, None
    selected = np.random.choice(walking_indices, size=min(num_samples, len(walking_indices)), replace=True)
    out_data, out_labels = [], []
    for idx in selected:
        sample = base_data['data'][idx].copy()
        C, T, V = sample.shape
        for t in range(T):
            for c in range(C):
                mid = 0.5 * (sample[c, t, 5] + sample[c, t, 6])
                sample[c, t, 9] = 0.6 * sample[c, t, 9] + 0.4 * mid
                sample[c, t, 10] = 0.6 * sample[c, t, 10] + 0.4 * mid
        sample += np.random.normal(0, 0.015 * np.std(sample), sample.shape)
        out_data.append(sample)
        out_labels.append(7)
    return np.array(out_data), np.array(out_labels)


def generate_synthetic_calling_data(base_data, num_samples=500):
    walking_indices = np.where(base_data['labels'] == 0)[0]
    if len(walking_indices) == 0:
        return None, None
    selected = np.random.choice(walking_indices, size=min(num_samples, len(walking_indices)), replace=True)
    out_data, out_labels = [], []
    for idx in selected:
        sample = base_data['data'][idx].copy()
        C, T, V = sample.shape
        for t in range(T):
            sample[:, t, 10] = 0.5 * sample[:, t, 10] + 0.5 * sample[:, t, 4]
        sample += np.random.normal(0, 0.02 * np.std(sample), sample.shape)
        out_data.append(sample)
        out_labels.append(8)
    return np.array(out_data), np.array(out_labels)


def generate_synthetic_talking_data(base_data, num_samples=500):
    walking_indices = np.where(base_data['labels'] == 0)[0]
    if len(walking_indices) == 0:
        return None, None
    selected = np.random.choice(walking_indices, size=min(num_samples, len(walking_indices)), replace=True)
    out_data, out_labels = [], []
    for idx in selected:
        sample = base_data['data'][idx].copy()
        base_tiled = np.tile(sample[:, 0:1, :], (1, sample.shape[1], 1))
        sample = 0.7 * sample + 0.3 * base_tiled
        sample += np.random.normal(0, 0.03 * np.std(sample), sample.shape)
        out_data.append(sample)
        out_labels.append(9)
    return np.array(out_data), np.array(out_labels)


# -----------------------------
# Prepare augmented dataset
# -----------------------------
def prepare_augmented_dataset(original_data_dir, output_dir,
                              use_synthetic=True,
                              synthetic_waiting=2000,
                              synthetic_waving=500,
                              synthetic_waving_walking=500,
                              synthetic_texting=500,
                              synthetic_calling=500,
                              synthetic_talking=500,
                              ntu60_dir=None,
                              kinetics_dir=None,
                              ucf101_dir=None,
                              max_samples_per_dataset=5000,
                              target_frames=30):
    print("=" * 80)
    print("Preparing Augmented Dataset (10 classes, risk analysis)")
    print("=" * 80)

    original_data_dir = Path(original_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_data = np.load(original_data_dir / "data.npy")
    original_labels = np.load(original_data_dir / "labels.npy")
    in_channels = original_data.shape[1]

    n_classes_orig = int(original_labels.max()) + 1
    print(f"\nOriginal: {len(original_data)} samples, {n_classes_orig} classes")
    bc = np.bincount(original_labels, minlength=NUM_BEHAVIOR_CLASSES)
    for i in range(min(n_classes_orig, NUM_BEHAVIOR_CLASSES)):
        print(f"  {BEHAVIOR_CLASS_NAMES[i]}: {bc[i]}")

    augmented_data = [original_data]
    augmented_labels = [original_labels]

    def _match_channels(ext_data):
        if ext_data is None or ext_data.shape[1] == in_channels:
            return ext_data
        if ext_data.shape[1] >= in_channels:
            return ext_data[:, :in_channels, :, :].copy()
        pad = np.zeros((ext_data.shape[0], in_channels - ext_data.shape[1], ext_data.shape[2], ext_data.shape[3]), dtype=ext_data.dtype)
        return np.concatenate([ext_data, pad], axis=1)

    max_per_behavior = max(1, max_samples_per_dataset // NUM_BEHAVIOR_CLASSES) if max_samples_per_dataset else None

    if ntu60_dir:
        print(f"\nLoading NTU60 from {ntu60_dir}...")
        ext_data, ext_labels = load_ntu_skeleton_data(ntu60_dir, max_samples_per_behavior=max_per_behavior, target_frames=target_frames)
        if ext_data is not None and len(ext_data) > 0:
            augmented_data.append(_match_channels(ext_data))
            augmented_labels.append(ext_labels)
            print(f"  Added {len(ext_data)} NTU60 samples")
    if kinetics_dir:
        print(f"\nLoading Kinetics from {kinetics_dir}...")
        ext_data, ext_labels = load_kinetics_skeleton_data(kinetics_dir, max_samples_per_behavior=max_per_behavior, target_frames=target_frames)
        if ext_data is not None and len(ext_data) > 0:
            augmented_data.append(_match_channels(ext_data))
            augmented_labels.append(ext_labels)
            print(f"  Added {len(ext_data)} Kinetics samples")
    if ucf101_dir:
        print(f"\nLoading UCF101 from {ucf101_dir}...")
        ext_data, ext_labels = load_ucf101_skeleton_data(ucf101_dir, max_samples_per_behavior=max_per_behavior, target_frames=target_frames)
        if ext_data is not None and len(ext_data) > 0:
            augmented_data.append(_match_channels(ext_data))
            augmented_labels.append(ext_labels)
            print(f"  Added {len(ext_data)} UCF101 samples")

    if use_synthetic:
        base_data = {'data': original_data, 'labels': original_labels}
        for name, n, gen in [
            ("Waiting_To_Cross", synthetic_waiting, generate_synthetic_waiting_data),
            ("Waving", synthetic_waving, generate_synthetic_waving_data),
            ("Waving_Walking", synthetic_waving_walking, generate_synthetic_waving_walking_data),
            ("Texting", synthetic_texting, generate_synthetic_texting_data),
            ("Calling", synthetic_calling, generate_synthetic_calling_data),
            ("Talking", synthetic_talking, generate_synthetic_talking_data),
        ]:
            if n > 0:
                d, l = gen(base_data, num_samples=n)
                if d is not None:
                    augmented_data.append(d)
                    augmented_labels.append(l)
                    print(f"  Added {len(d)} synthetic {name}")

    final_data = np.concatenate(augmented_data, axis=0)
    final_labels = np.concatenate(augmented_labels, axis=0)
    indices = np.random.permutation(len(final_data))
    final_data = final_data[indices]
    final_labels = final_labels[indices]

    np.save(output_dir / "data.npy", final_data)
    np.save(output_dir / "labels.npy", final_labels)
    with open(output_dir / "class_names.txt", "w") as f:
        f.write("\n".join(BEHAVIOR_CLASS_NAMES))

    n_classes_final = int(final_labels.max()) + 1
    bc_final = np.bincount(final_labels, minlength=NUM_BEHAVIOR_CLASSES)
    print(f"\nFinal: {len(final_data)} samples, {n_classes_final} classes")
    for i in range(n_classes_final):
        print(f"  {BEHAVIOR_CLASS_NAMES[i]}: {bc_final[i]} ({100*bc_final[i]/len(final_labels):.2f}%)")
    print(f"\nDone. Saved to {output_dir}")
    return output_dir


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepare augmented skeleton dataset (10 classes)")
    parser.add_argument("--original_data_dir", type=str, default="./gcn_per_pedestrian")
    parser.add_argument("--output_dir", type=str, default="./gcn_per_pedestrian_augmented")
    parser.add_argument("--synthetic_samples", type=int, default=2000, help="Waiting_To_Cross (backward compat)")
    parser.add_argument("--synthetic_waiting", type=int, default=None)
    parser.add_argument("--synthetic_waving", type=int, default=500)
    parser.add_argument("--synthetic_waving_walking", type=int, default=500)
    parser.add_argument("--synthetic_texting", type=int, default=500)
    parser.add_argument("--synthetic_calling", type=int, default=500)
    parser.add_argument("--synthetic_talking", type=int, default=500)
    parser.add_argument("--no_synthetic", action="store_true")
    parser.add_argument("--download_ntu60", action="store_true", help="Auto-download NTU60 from Figshare (~12 GB)")
    parser.add_argument("--ntu60_download_dir", type=str, default=NTU60_DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--ntu60_dir", type=str, default=None)
    parser.add_argument("--kinetics_dir", type=str, default=None)
    parser.add_argument("--ucf101_dir", type=str, default=None)
    parser.add_argument("--max_samples_per_dataset", type=int, default=5000)
    parser.add_argument("--target_frames", type=int, default=30)
    args = parser.parse_args()

    synthetic_waiting = args.synthetic_waiting if args.synthetic_waiting is not None else args.synthetic_samples

    ntu60_dir = args.ntu60_dir
    if args.download_ntu60:
        download_dir = Path(args.ntu60_download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)
        if download_ntu60_figshare(download_dir, skip_if_exists=True):
            npz_path = download_dir / "ntu60_preprocessed.npz"
            if not npz_path.exists():
                convert_ntu60_download_to_npz(download_dir)
            if npz_path.exists():
                ntu60_dir = str(npz_path)
                print(f"Using downloaded NTU60: {ntu60_dir}")

    output_dir = prepare_augmented_dataset(
        args.original_data_dir,
        args.output_dir,
        use_synthetic=not args.no_synthetic,
        synthetic_waiting=synthetic_waiting,
        synthetic_waving=args.synthetic_waving,
        synthetic_waving_walking=args.synthetic_waving_walking,
        synthetic_texting=args.synthetic_texting,
        synthetic_calling=args.synthetic_calling,
        synthetic_talking=args.synthetic_talking,
        ntu60_dir=ntu60_dir,
        kinetics_dir=args.kinetics_dir,
        ucf101_dir=args.ucf101_dir,
        max_samples_per_dataset=args.max_samples_per_dataset,
        target_frames=args.target_frames,
    )
    print("\nNext: python gcn_train.py --data_dir", output_dir, "--model stgcn --epochs 50")


if __name__ == "__main__":
    main()
