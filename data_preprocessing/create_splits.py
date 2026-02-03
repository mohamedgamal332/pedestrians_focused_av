#!/usr/bin/env python3
"""
Create train/test/eval splits by pooling ALL frames from ALL sessions
and shuffling them together.

Updates:
- Enforces strict filtering (frames with 0 valid peds are dropped).
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm

# Import from your dataloader file
from dataloader import CARLAStereoPedestrianDataset

def create_splits(
    sessions_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.6,
    test_ratio: float = 0.2,
    eval_ratio: float = 0.2,
    camera: str = 'left',
    random_seed: int = 42,
    min_keypoints: int = 0,
    max_distance: float = None,
    verbose: bool = True,
):
    
    # 1. Setup
    assert abs(train_ratio + test_ratio + eval_ratio - 1.0) < 0.001, "Ratios must sum to 1.0"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir = Path(sessions_dir)
    
    # === CRITICAL FIX HERE ===
    # We enforce filter_min_pedestrians=1. 
    # This ensures that if all pedestrians in a frame are filtered out 
    # (due to max_distance or min_keypoints), the frame itself is dropped.
    dataset_args = {
        'load_images': False,
        'load_depth': False,
        'filter_min_visible_keypoints': min_keypoints,
        'filter_max_distance': max_distance,
        'filter_min_pedestrians': 1  # <--- Forces skipping of "bad" frames
    }

    # 2. Discovery & Aggregation
    dataset_cache = {} 
    all_frames = [] 

    sessions = sorted([d for d in sessions_dir.glob("session_*") if d.is_dir()])
    
    if verbose: print(f"Scanning {len(sessions)} sessions for valid frames...")
    
    missing_files_count = 0
    
    for s_path in sessions:
        try:
            # Initialize dataset (filtering happens here automatically)
            ds = CARLAStereoPedestrianDataset(s_path, **dataset_args)
            dataset_cache[str(s_path)] = ds
            
            # Iterate over filtered frame IDs
            for fid in ds.frame_ids:
                f_id_int = int(fid)
                
                # --- NEW: Physical File Check ---
                # We assume the standard CARLA format: frame_001234.png
                img_filename = f"frame_{f_id_int:06d}.png"
                img_path = s_path / f"rgb_{camera}" / img_filename
                
                if not img_path.exists():
                    # Optional: Check for non-padded filenames if your data is different
                    # img_path_alt = s_path / f"rgb_{camera}" / f"frame_{f_id_int}.png"
                    # if img_path_alt.exists(): img_path = img_path_alt
                    # else:
                    
                    missing_files_count += 1
                    if missing_files_count <= 10:
                        print(f"Warning: Image missing on disk, skipping: {img_path}")
                    elif missing_files_count == 11:
                        print("Warning: More images missing... suppressing output.")
                    continue
                # --------------------------------
                
                all_frames.append({
                    'session_path': str(s_path),
                    'frame_id': f_id_int
                })
        except Exception as e:
            print(f"Skipping session {s_path.name}: {e}")

    total_frames = len(all_frames)
    if missing_files_count > 0:
        print(f"Total missing images skipped: {missing_files_count}")

    if total_frames == 0:
        raise RuntimeError("No valid frames found! Try relaxing filters or checking file paths.")

    # 3. Shuffle & Split
    np.random.seed(random_seed)
    np.random.shuffle(all_frames)
    
    n_train = int(total_frames * train_ratio)
    n_test = int(total_frames * test_ratio)
    
    splits = {
        'train': all_frames[:n_train],
        'test': all_frames[n_train:n_train + n_test],
        'eval': all_frames[n_train + n_test:]
    }

    if verbose:
        print(f"\nTotal Valid Frames: {total_frames}")
        print(f"  train: {len(splits['train'])}")
        print(f"  test:  {len(splits['test'])}")
        print(f"  eval:  {len(splits['eval'])}")

    # 4. Generate Outputs
    for split_name, frames in splits.items():
        if len(frames) == 0:
            continue
            
        if verbose: print(f"\nProcessing {split_name.upper()} split...")

        # Create Indices
        create_split_indices(
            frames,
            output_dir / f'{split_name}_indices.json',
            split_name
        )
        
        # Create Annotations
        create_coco_annotations(
            frames,
            dataset_cache,
            output_dir / f'{split_name}_annotations.json',
            camera,
            verbose
        )

    meta = {k: len(v) for k, v in splits.items()}
    with open(output_dir / 'splits_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    return splits


def create_split_indices(frame_list: List[Dict], output_file: Path, split_name: str):
    indices_out = []
    for idx, item in enumerate(frame_list):
        indices_out.append({
            'global_index': idx,
            'session': item['session_path'],
            'frame_id': item['frame_id']
        })
    
    with open(output_file, 'w') as f:
        json.dump(indices_out, f, indent=2)


def create_coco_annotations(
    frame_list: List[Dict], 
    dataset_cache: Dict[str, CARLAStereoPedestrianDataset], 
    output_file: Path, 
    camera: str, 
    verbose: bool
):
    COCO_KEYPOINTS = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    COCO_SKELETON = [
        [0, 1], [0, 2], [1, 3], [2, 4],
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
        [5, 11], [6, 12], [11, 12],
        [11, 13], [13, 15], [12, 14], [14, 16]
    ]

    first_ds = next(iter(dataset_cache.values()))
    cam_intrinsics = getattr(first_ds.intrinsics, camera)
    img_width, img_height = cam_intrinsics.width, cam_intrinsics.height
    
    coco_images = []
    coco_annotations = []
    image_id_counter = 0
    ann_id_counter = 0
    
    iterator = tqdm(frame_list, desc="Generating COCO", disable=not verbose)
    
    for item in iterator:
        s_path = item['session_path']
        f_id = item['frame_id']
        ds = dataset_cache[s_path]
        
        try:
            ann = ds._load_annotation(f_id)
        except Exception:
            continue

        coco_images.append({
            'id': image_id_counter,
            'file_name': f"{Path(s_path).name}/rgb_{camera}/frame_{f_id:06d}.png",
            'height': img_height,
            'width': img_width,
            'session': str(s_path),
            'frame_id': f_id,
        })
        
        for ped in ann.pedestrians:
            # Re-verify visibility just to be safe (though dataloader should have handled it)
            kps = ped.get_keypoints_array(camera=camera, include_visibility=True)
            kps[:, 0] = np.clip(kps[:, 0], 0, img_width - 1)
            kps[:, 1] = np.clip(kps[:, 1], 0, img_height - 1)
            
            num_visible = int((kps[:, 2] == 2).sum())
            if num_visible == 0:
                continue

            bbox_raw = ped.get_bounding_box(camera=camera, padding=0.1)
            if bbox_raw is None:
                continue

            x1 = max(0, min(bbox_raw[0], img_width - 1))
            y1 = max(0, min(bbox_raw[1], img_height - 1))
            x2 = max(0, min(bbox_raw[2], img_width - 1))
            y2 = max(0, min(bbox_raw[3], img_height - 1))
            w, h = x2 - x1, y2 - y1
            
            if w <= 1 or h <= 1: 
                continue

            coco_annotations.append({
                'id': ann_id_counter,
                'image_id': image_id_counter,
                'category_id': 1,
                'keypoints': kps.flatten().tolist(),
                'bbox': [float(x1), float(y1), float(w), float(h)],
                'area': float(w * h),
                'iscrowd': 0,
                'num_keypoints': num_visible,
                'pedestrian_id': ped.id,
                'behavior': ped.behavior.value
            })
            ann_id_counter += 1
        
        image_id_counter += 1

    coco_data = {
        'images': coco_images,
        'annotations': coco_annotations,
        'categories': [{
            'id': 1, 
            'name': 'person', 
            'keypoints': COCO_KEYPOINTS, 
            'skeleton': COCO_SKELETON, 
            'supercategory': 'person'
        }]
    }

    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions-dir', type=str, default='/home/theta/carla/output/sessions')
    parser.add_argument('--output-dir', type=str, default='/home/theta/RTMPose/splits')
    parser.add_argument('--camera', type=str, choices=['left', 'right'], default='left')
    parser.add_argument('--splits', type=float, nargs=3, default=[0.6, 0.2, 0.2])
    parser.add_argument('--min-keypoints', type=int, default=3)
    parser.add_argument('--max-dist', type=float, default=40.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quiet', action='store_true')
    
    args = parser.parse_args()
    
    create_splits(
        sessions_dir=args.sessions_dir,
        output_dir=args.output_dir,
        train_ratio=args.splits[0],
        test_ratio=args.splits[1],
        eval_ratio=args.splits[2],
        camera=args.camera,
        random_seed=args.seed,
        min_keypoints=args.min_keypoints,
        max_distance=args.max_dist,
        verbose=not args.quiet,
    )

if __name__ == '__main__':
    main()