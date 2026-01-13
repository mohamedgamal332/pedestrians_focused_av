import argparse
import json
from pathlib import Path
from typing import List, Optional


def estimate_depth_from_bbox(bbox: Optional[List[float]], scale: float, min_depth: float) -> float:
    """Estimate a synthetic depth from a bbox height using a simple inverse rule.

    depth = scale / bbox_height, clamped to min_depth. If bbox missing or invalid, return -1.
    """
    if not bbox or len(bbox) < 4:
        return -1.0
    x1, y1, x2, y2 = bbox[:4]
    h = y2 - y1
    if h <= 0:
        return -1.0
    depth = scale / h
    if depth < min_depth:
        depth = min_depth
    return float(depth)


def process_file(in_path: Path, out_path: Path, scale: float, min_depth: float) -> None:
    data = json.loads(in_path.read_text())
    out_frames = []

    for frame in data:
        frame_idx = frame.get("frame_index")
        keypoints_all = frame.get("keypoints", [])
        bboxes_all = frame.get("bboxes", [])

        people = []
        # keypoints_all is expected to be a list of persons, each a list of keypoint [x,y]
        for i, person_kps in enumerate(keypoints_all):
            bbox = None
            if i < len(bboxes_all):
                bbox = bboxes_all[i]

            depth = estimate_depth_from_bbox(bbox, scale, min_depth)

            kps_with_depth = []
            for kp in person_kps:
                if not isinstance(kp, (list, tuple)) or len(kp) < 2:
                    kps_with_depth.append([-1.0, -1.0, -1.0])
                    continue
                x, y = kp[0], kp[1]
                if x is None or y is None:
                    kps_with_depth.append([-1.0, -1.0, -1.0])
                    continue
                # If keypoint coords are invalid (e.g. -1), propagate missing
                if (isinstance(x, (int, float)) and x < 0) or (isinstance(y, (int, float)) and y < 0):
                    kps_with_depth.append([-1.0, -1.0, -1.0])
                else:
                    kps_with_depth.append([float(x), float(y), depth])

            people.append({
                "bbox": bbox,
                "keypoints_xy_depth": kps_with_depth,
            })

        out_frames.append({"frame_index": frame_idx, "people": people})

    out_path.write_text(json.dumps(out_frames, indent=2))


def find_input_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]

    candidates = []
    for p in path.rglob("*.json"):
        name = p.name.lower()
        parts = [part.lower() for part in p.parts]
        if "_left" in name or "_right" in name:
            candidates.append(p)
            continue
        if any(part == "left" or part == "right" for part in parts):
            candidates.append(p)

    if candidates:
        seen = set()
        ordered = []
        for p in candidates:
            if p not in seen:
                seen.add(p)
                ordered.append(p)
        return ordered

    # fallback: any .json files if none matched
    return list(path.rglob("*.json"))


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic depth per keypoint (x,y,d) from saved MMPose JSON outputs.")
    # make input optional; default to repo's hrnet_ProcessedScenes when omitted
    default_hrnet_dir = Path(__file__).resolve().parent.parent / "hrnet_ProcessedScenes"
    parser.add_argument("input", nargs="?", default=str(default_hrnet_dir),
                        help="Input JSON file or directory containing per-scene JSON files. (default: hrnet_ProcessedScenes)")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory. If omitted, writes alongside input with suffix '_with_depth'.")
    parser.add_argument("--scale", type=float, default=1000.0, help="Scale factor for depth estimation (depth = scale / bbox_height).")
    parser.add_argument("--min-depth", type=float, default=0.1, help="Minimum depth value to clamp to.")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input path does not exist: {in_path}")

    files = find_input_files(in_path)
    if not files:
        raise SystemExit(f"No JSON files found in {in_path}")

    out_base = Path(args.output_dir) if args.output_dir else None

    for f in files:
        if out_base:
            out_base.mkdir(parents=True, exist_ok=True)
            out_file = out_base / (f.stem + "_with_depth.json")
        else:
            out_file = f.with_name(f.stem + "_with_depth.json")

        print(f"Processing {f} -> {out_file}")
        try:
            process_file(f, out_file, args.scale, args.min_depth)
        except Exception as e:
            print(f"Failed processing {f}: {e}")


if __name__ == "__main__":
    main()
