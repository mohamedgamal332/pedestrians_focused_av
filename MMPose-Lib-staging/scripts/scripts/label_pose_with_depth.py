"""Label poses in hrnet_depth JSONs using simple heuristics.

Usage:
    python scripts/label_pose_with_depth.py --input_dir hrnet_depth --out_dir pose_label_sample

This script finds all files under `input_dir` that end with `_with_depth.json`,
labels each person per frame using speed and hand-raise heuristics, and writes
one output JSON per input file under `out_dir` mirroring the subfolder layout.

NOTE: The heuristic assumes `keypoints_xy_depth` is a list of [x,y,z] per joint.
Adjust head/hand indices in `HEAD_IDX`, `RIGHT_HAND_IDX`, `LEFT_HAND_IDX` if
your joint ordering differs.
"""
import argparse
import json
from pathlib import Path
import math
import numpy as np

# Heuristic action classes
ACTION_CLASSES = ["Standing", "Walking", "Running", "Waving", "Waving+Walking"]

# Adjust these indices to match your skeleton mapping if needed
HEAD_IDX = 0
RIGHT_HAND_IDX = 4
LEFT_HAND_IDX = 5


def compute_speed(prev_kp, curr_kp):
    prev_kp = np.array(prev_kp)
    curr_kp = np.array(curr_kp)
    if prev_kp.shape != curr_kp.shape:
        return 0.0
    return float(np.mean(np.linalg.norm(curr_kp - prev_kp, axis=1)))


def is_hand_raised(kp):
    try:
        head_y = kp[HEAD_IDX][1]
        right_hand_y = kp[RIGHT_HAND_IDX][1]
        left_hand_y = kp[LEFT_HAND_IDX][1]
        # in image coordinates y grows downward: smaller y -> higher
        return (right_hand_y < head_y) or (left_hand_y < head_y)
    except Exception:
        return False


def label_action(prev_kp, curr_kp):
    speed = compute_speed(prev_kp, curr_kp)
    hand_up = is_hand_raised(curr_kp)

    if speed < 1.0:
        if hand_up:
            return "Waving"
        else:
            return "Standing"
    elif speed < 5.0:
        if hand_up:
            return "Waving+Walking"
        else:
            return "Walking"
    else:
        return "Running"


def centroid_of_kp(kp):
    kp = np.array(kp)
    if kp.size == 0:
        return (0.0, 0.0)
    xs = kp[:, 0]
    ys = kp[:, 1]
    return (float(xs.mean()), float(ys.mean()))


def process_file(in_path: Path, out_path: Path, match_max_dist: float = 100.0):
    with open(in_path, 'r') as f:
        frames = json.load(f)

    # tracking by greedy centroid matching across frames to preserve person identity
    next_id = 0
    prev_people = []  # list of dicts: {id, centroid, keypoints}
    outputs = []

    for fi, frame in enumerate(frames):
        frame_index = frame.get('frame_index', fi)
        people = frame.get('people', [])

        # compute centroids
        dets = []
        for p in people:
            kps = p.get('keypoints_xy_depth') or p.get('keypoints') or []
            cent = centroid_of_kp(kps)
            dets.append({'kps': kps, 'cent': cent, 'bbox': p.get('bbox')})

        assignments = {}
        used_prev = set()

        # match current detections to previous people by centroid distance
        for di, d in enumerate(dets):
            best_tid = None
            best_dist = float('inf')
            for prev in prev_people:
                if prev['id'] in used_prev:
                    continue
                dist = math.hypot(d['cent'][0] - prev['cent'][0], d['cent'][1] - prev['cent'][1])
                if dist < best_dist:
                    best_dist = dist
                    best_tid = prev['id']

            if best_tid is not None and best_dist <= match_max_dist:
                assignments[di] = best_tid
                used_prev.add(best_tid)
            else:
                assignments[di] = next_id
                next_id += 1

        # build outputs for this frame
        frame_out = []
        for di, d in enumerate(dets):
            tid = assignments[di]
            # find previous keypoints for this tid if available
            prev_kp = d['kps']
            for prev in prev_people:
                if prev['id'] == tid:
                    prev_kp = prev.get('kps', d['kps'])
                    break

            curr_kp = d['kps']
            if curr_kp is None:
                continue
            action_label = label_action(prev_kp, curr_kp)
            frame_out.append({
                'track_id': tid,
                'frame_index': frame_index,
                'skeleton': curr_kp,
                'action_label': action_label,
                'bbox': d.get('bbox')
            })

        outputs.append({'frame_index': frame_index, 'people': frame_out})

        # update prev_people
        new_prev = []
        for di, d in enumerate(dets):
            tid = assignments[di]
            new_prev.append({'id': tid, 'cent': d['cent'], 'kps': d['kps']})
        prev_people = new_prev

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(outputs, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='hrnet_depth', help='Input base dir with _with_depth.json files')
    parser.add_argument('--out_dir', default='pose_label_sample', help='Output base dir')
    parser.add_argument('--max_dist', type=float, default=100.0, help='Max centroid distance for matching')
    args = parser.parse_args()

    base = Path(args.input_dir)
    out_base = Path(args.out_dir)

    files = list(base.rglob('*_with_depth.json'))
    if not files:
        print('No files found matching *_with_depth.json under', base)
        return

    for f in files:
        rel = f.relative_to(base)
        out_path = out_base / rel.parent / (rel.stem + '_pose_labeled.json')
        print(f'Processing {f} -> {out_path}')
        try:
            process_file(f, out_path, match_max_dist=args.max_dist)
        except Exception as e:
            print('Failed processing', f, 'error:', e)

    print('Done')


if __name__ == '__main__':
    main()
