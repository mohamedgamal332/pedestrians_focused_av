"""Compute per-person risk scores from labeled pose JSONs.

Searches a base dir for files ending with `_pose_labeled.json`, groups
entries by `track_id`, computes a majority `action_label` per track,
builds the skeleton sequence (T, V, 2) and uses `scripts.risk_score.compute_person_risk`
to compute a single risk value per track. Writes CSV `person_risk_scores.csv`
under the base dir with columns: scene_path,person_id,risk
"""
import json
from pathlib import Path
from collections import defaultdict, Counter
import csv
import numpy as np

from scripts.risk_score import compute_person_risk


def load_pose_labeled(path: Path):
    with open(path, 'r') as f:
        data = json.load(f)
    # data is list of frames: {'frame_index', 'people': [ {track_id, frame_index, skeleton, action_label, bbox}, ... ]}
    tracks = defaultdict(list)
    for frame in data:
        for p in frame.get('people', []):
            tid = int(p.get('track_id'))
            sk = p.get('skeleton') or []
            # ensure list-of-lists
            tracks[tid].append({'frame_index': frame.get('frame_index'), 'skeleton': sk, 'action_label': p.get('action_label')})
    return tracks


def build_skel_sequence(track_entries):
    # sort by frame_index
    track_entries = sorted(track_entries, key=lambda x: x.get('frame_index', 0))
    sks = [e['skeleton'] for e in track_entries]
    # convert to numpy array (T, V, C)
    arr = np.array(sks, dtype=float)
    if arr.ndim == 2:
        # single joint? expand
        arr = arr[:, :, None]
    # ensure at least (T, V, 2)
    if arr.shape[2] > 2:
        arr = arr[:, :, :2]
    return arr


def majority_action(track_entries):
    labels = [e.get('action_label') for e in track_entries if e.get('action_label')]
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def process_base(base_dir: Path):
    files = list(base_dir.rglob('*_pose_labeled.json'))
    if not files:
        print('No _pose_labeled.json files found under', base_dir)
        return

    out_rows = []
    for f in files:
        rel = f.relative_to(base_dir)
        scene_prefix = str(rel.parent)
        try:
            tracks = load_pose_labeled(f)
        except Exception as e:
            print('Failed to load', f, 'error:', e)
            continue

        for tid, entries in tracks.items():
            maj_action = majority_action(entries) or 'Unknown'
            sk_seq = build_skel_sequence(entries)
            try:
                risk = compute_person_risk(sk_seq, maj_action)
            except Exception:
                risk = 0.0
            # scale to 1-10
            risk_1_10 = 1.0 + float(risk) * 9.0
            # unique person id: file stem + track id
            person_id = f'{f.stem}::track_{tid}'
            out_rows.append({'scene': scene_prefix, 'person_id': person_id, 'risk': float(risk), 'risk_1_10': round(risk_1_10, 4), 'action': maj_action})

    # write CSV
    out_csv = base_dir / 'person_risk_scores.csv'
    with open(out_csv, 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=['scene', 'person_id', 'action', 'risk', 'risk_1_10'])
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print('Wrote', out_csv)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='pose_label_sample_bal')
    args = parser.parse_args()
    process_base(Path(args.base_dir))
