import json
from pathlib import Path
from typing import List, Optional


def mean_ignore_invalid(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def compute_center_from_keypoints(kps: List[List[float]]) -> (Optional[float], Optional[float], Optional[float]):
    xs = []
    ys = []
    ds = []
    for kp in kps:
        # expect [x, y, d]
        if not isinstance(kp, (list, tuple)) or len(kp) < 3:
            continue
        x, y, d = kp[0], kp[1], kp[2]
        # treat negative or None as invalid
        if x is None or y is None or d is None:
            continue
        try:
            if (isinstance(x, (int, float)) and x >= 0) and (isinstance(y, (int, float)) and y >= 0) and (isinstance(d, (int, float)) and d >= 0):
                xs.append(float(x))
                ys.append(float(y))
                ds.append(float(d))
        except Exception:
            continue
    if not xs or not ys or not ds:
        return None, None, None
    return sum(xs) / len(xs), sum(ys) / len(ys), sum(ds) / len(ds)


def bbox_size_from_bbox(bbox: Optional[List[float]]) -> (Optional[float], Optional[float]):
    if not bbox or len(bbox) < 4:
        return None, None
    x1, y1, x2, y2 = bbox[:4]
    try:
        w = float(x2) - float(x1)
        h = float(y2) - float(y1)
    except Exception:
        return None, None
    return w, h


def process_file(in_path: Path, out_path: Path) -> None:
    data = json.loads(in_path.read_text())
    out_frames = []

    prev_centers = []  # list of (Cx, Cy, Cz) for previous frame, by person index

    for frame in data:
        frame_idx = frame.get("frame_index")
        people = frame.get("people", [])

        out_people = []

        for i, person in enumerate(people):
            person_id = f"person_{i}"
            bbox = person.get("bbox")
            kps = person.get("keypoints_xy_depth") or []

            Cx, Cy, Cz = compute_center_from_keypoints(kps)
            w, h = bbox_size_from_bbox(bbox)

            # velocity compared to previous frame same person index
            if prev_centers and i < len(prev_centers):
                prev = prev_centers[i]
                if prev is None or Cx is None or Cy is None or Cz is None:
                    Vx = Vy = Vz = 0.0
                else:
                    Vx = (Cx - prev[0])
                    Vy = (Cy - prev[1])
                    Vz = (Cz - prev[2])
            else:
                Vx = Vy = Vz = 0.0

            state = [Cx if Cx is not None else None,
                     Cy if Cy is not None else None,
                     Cz if Cz is not None else None,
                     Vx,
                     Vy,
                     Vz,
                     w if w is not None else None,
                     h if h is not None else None]

            out_people.append({
                "person_id": person_id,
                "state": state,
                "bbox": bbox,
            })

        # prepare prev_centers for next iteration
        new_prev = []
        for person in out_people:
            s = person["state"]
            if s[0] is None or s[1] is None or s[2] is None:
                new_prev.append(None)
            else:
                new_prev.append((s[0], s[1], s[2]))

        prev_centers = new_prev

        out_frames.append({"frame_index": frame_idx, "people": out_people})

    out_path.write_text(json.dumps(out_frames, indent=2))


def find_input_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return list(path.rglob("*_with_depth.json")) + list(path.rglob("*.json"))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute state vectors S_t per person per frame from depth-augmented JSONs.")
    default_dir = Path(__file__).resolve().parent.parent / "hrnet_depth"
    parser.add_argument("input", nargs="?", default=str(default_dir), help="Input file or directory (default: hrnet_depth)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input path does not exist: {in_path}")

    files = find_input_files(in_path)
    if not files:
        raise SystemExit(f"No JSON files found in {in_path}")

    for f in files:
        out_file = f.with_name(f.stem + "_states.json")
        print(f"Processing {f} -> {out_file}")
        try:
            process_file(f, out_file)
        except Exception as e:
            print(f"Failed processing {f}: {e}")


if __name__ == "__main__":
    main()
