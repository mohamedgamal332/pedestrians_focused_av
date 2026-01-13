"""Convert RTMPose per-frame outputs into person-centric tracks with keypoints.

Writes per-scene+camera track JSON files with structure:
{
  "tracks": {
     "0": [{"frame_index":0, "keypoints": [[x,y,...],...], "bbox": [...]}, ...],
     "1": [...]
  }
}

Tracking is a lightweight centroid-based greedy matcher using bbox or keypoint centroid.
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import math


def centroid_from_det(det: Dict[str, Any]) -> List[float]:
    # prefer bbox center if available
    bbox = det.get('bbox') or det.get('bboxes')
    if bbox and len(bbox) >= 4:
        x = (bbox[0] + bbox[2]) / 2.0
        y = (bbox[1] + bbox[3]) / 2.0
        return [x, y]
    # fallback: mean of keypoints
    kps = det.get('keypoints')
    if kps and len(kps) > 0:
        xs = [p[0] for p in kps if isinstance(p, (list, tuple)) and len(p) >= 2 and p[0] >= 0]
        ys = [p[1] for p in kps if isinstance(p, (list, tuple)) and len(p) >= 2 and p[1] >= 0]
        if xs and ys:
            return [sum(xs) / len(xs), sum(ys) / len(ys)]
    return [0.0, 0.0]


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def track_scene_json(input_json_path: str, out_path: str, max_distance: float = 100.0):
    """Read a per-camera per-scene RTMPose JSON and output tracks.

    Args:
        input_json_path: path to RTMPose result (list of frames)
        out_path: output tracks JSON path
        max_distance: maximum centroid distance to match detections to existing tracks
    """
    with open(input_json_path, 'r') as f:
        frames = json.load(f)

    tracks: Dict[int, List[Dict]] = {}
    # store last centroid per track
    last_centroids: Dict[int, List[float]] = {}
    next_id = 0

    for frame in frames:
        fi = frame.get('frame_index', None)
        dets = frame.get('keypoints', [])  # list of persons, each is list of [x,y] pairs
        bboxes = frame.get('bboxes', [])

        # convert to unified per-detection dicts
        detections = []
        for i, kp in enumerate(dets):
            det = {'keypoints': kp}
            if i < len(bboxes):
                det['bbox'] = bboxes[i]
            detections.append(det)

        # compute centroids
        det_centroids = [centroid_from_det({'keypoints': d.get('keypoints'), 'bbox': d.get('bbox')}) for d in detections]

        assigned = set()
        # greedy match: for each detection find nearest track last centroid
        for di, c in enumerate(det_centroids):
            best_tid = None
            best_dist = float('inf')
            for tid, lc in last_centroids.items():
                d = euclidean(c, lc)
                if d < best_dist:
                    best_dist = d
                    best_tid = tid

            if best_tid is not None and best_dist <= max_distance and best_tid not in assigned:
                assigned.add(best_tid)
                tracks.setdefault(best_tid, []).append({
                    'frame_index': fi,
                    'keypoints': detections[di].get('keypoints'),
                    'bbox': detections[di].get('bbox', None)
                })
                last_centroids[best_tid] = det_centroids[di]
            else:
                # new track
                tid = next_id
                next_id += 1
                tracks[tid] = [{
                    'frame_index': fi,
                    'keypoints': detections[di].get('keypoints'),
                    'bbox': detections[di].get('bbox', None)
                }]
                last_centroids[tid] = det_centroids[di]

    out = {'tracks': tracks}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)


def process_scenes(base_dir: str, out_dir: str, camera_names: List[str] = None, max_distance: float = 100.0):
    """Process all scene camera JSONs under base_dir into tracks under out_dir.

    base_dir layout: Scene_X/.../Right/scene_right.json
    """
    base = Path(base_dir)
    out_base = Path(out_dir)
    for scene in base.iterdir():
        if not scene.is_dir():
            continue
        for cam in (scene).iterdir():
            if not cam.is_dir():
                continue
            # find json file inside camera folder
            jsons = list(cam.glob("*.json"))
            if not jsons:
                continue
            for j in jsons:
                rel = j.relative_to(base)
                out_path = out_base / rel.parent / (rel.stem + '_tracks.json')
                print(f"Tracking {j} -> {out_path}")
                track_scene_json(str(j), str(out_path), max_distance=max_distance)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Base dir with per-scene per-camera JSONs')
    parser.add_argument('--out', required=True, help='Output tracks base dir')
    parser.add_argument('--max-distance', type=float, default=100.0)
    args = parser.parse_args()
    process_scenes(args.input, args.out, max_distance=args.max_distance)
