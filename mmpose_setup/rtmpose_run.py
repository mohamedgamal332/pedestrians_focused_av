#!/usr/bin/env python3
"""
RTMPose Evaluation Run Script

Streaming evaluation with flip correction, disk-based storage,
and comprehensive analysis.

Usage:
    # Full evaluation
    python rtmpose_run.py
    
    # Quick test (100 frames)
    python rtmpose_run.py --max-frames 100
    
    # Without flip correction
    python rtmpose_run.py --no-flip
    
    # Analyze existing results
    python rtmpose_run.py --analyze ./eval_output/results.jsonl.gz
"""

from pathlib import Path
from typing import Optional, Dict, Any
import json
import numpy as np
import cv2
import argparse
from collections import defaultdict

# Import from evaluation module
from rtmpose_eval import (
    EvalConfig,
    PoseModelConfig,
    DetectorConfig,
    RTMPoseEvaluator,
    IncrementalStatistics,
    StreamingResultsWriter,
    ResultsReader,
    FrameStatus,
    visualize_evaluation_frame,
    convert_to_serializable,
    COCO_KEYPOINTS,
)
from dataloader import CARLAStereoPedestrianDataset

import sys

class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)

# =============================================================================
# Configuration
# =============================================================================

# Paths - adjust to your setup
MMPOSE_ROOT = Path.home() / 'RTMPose' / 'mmpose'
CHECKPOINTS = MMPOSE_ROOT / 'checkpoints'
CONFIGS = MMPOSE_ROOT / 'configs'

session = 'session_20260124_204300'

# Session path
SESSION_PATH = Path.home() / 'carla' / 'output' / 'sessions' / session
# Output directory
OUTPUT_DIR = f'./eval_output/{session}'

# Available pose models
POSE_MODELS = {
    'rtmpose-m': {
        'config': CONFIGS / 'body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py',
        'checkpoint': CHECKPOINTS / 'rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth',
    },
    'rtmpose-x': {
        'config': CONFIGS / 'body_2d_keypoint/rtmpose/coco/rtmpose-x_8xb256-420e_coco-384x288.py',
        'checkpoint': CHECKPOINTS / 'rtmpose-x_simcc-coco_pt-body7_420e-384x288-829f8b67_20230504.pth',
    },
    'vitpose-h': {
        'config': CONFIGS / 'body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py',
        'checkpoint': CHECKPOINTS / 'td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth',
    },
}

# Select pose model
SELECTED_MODEL = 'rtmpose-m'

# Detector
DETECTOR_CONFIG = CHECKPOINTS / 'det-config.py'
DETECTOR_CHECKPOINT = CHECKPOINTS / 'det-weights.pth'

# Evaluation settings
MAX_FRAMES = None  # None = all frames
DEVICE = 'cuda:0'
CAMERAS = ['left', 'right']
ENABLE_FLIP_CORRECTION = True
MAX_DEVIATION_THRESHOLD = 100.0  # pixels

# Visualization settings
SAVE_VISUALIZATIONS = True
VIS_INTERVAL = 500  # Save every N frames
MAX_VISUALIZATIONS = 200

# Logging
LOG_INTERVAL = 100
GC_INTERVAL = 50


# =============================================================================
# Main Evaluation Function
# =============================================================================

def run_evaluation(
    session_path: str,
    output_dir: str,
    pose_model: str = SELECTED_MODEL,
    max_frames: Optional[int] = None,
    enable_flip: bool = True,
    save_vis: bool = True,
    device: str = 'cuda:0',
) -> Dict[str, Any]:
    """
    Run streaming evaluation.
    
    Args:
        session_path: Path to CARLA session directory
        output_dir: Output directory for results
        pose_model: Name of pose model ('rtmpose-m', 'rtmpose-x', 'vitpose-h')
        max_frames: Maximum frames to evaluate (None = all)
        enable_flip: Enable left-right flip correction
        save_vis: Save visualization images
        device: Device for inference
    
    Returns:
        Final statistics dictionary
    """
    import time
    import gc
    
    try:
        import torch
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False
    
    print("=" * 70)
    print("RTMPose Streaming Evaluation")
    print("=" * 70)
    
    # Get model paths
    if pose_model not in POSE_MODELS:
        raise ValueError(f"Unknown pose model: {pose_model}. Choose from {list(POSE_MODELS.keys())}")
    
    model_info = POSE_MODELS[pose_model]
    pose_config = str(model_info['config'])
    pose_checkpoint = str(model_info['checkpoint'])
    
    print(f"\nConfiguration:")
    print(f"  Pose model:       {pose_model}")
    print(f"  Session:          {session_path}")
    print(f"  Output:           {output_dir}")
    print(f"  Flip correction:  {enable_flip}")
    print(f"  Max frames:       {max_frames or 'all'}")
    print(f"  Device:           {device}")
    
    # Create config
    config = EvalConfig(
        pose=PoseModelConfig(
            config_file=pose_config,
            checkpoint_file=pose_checkpoint,
            model_type=pose_model.split('-')[0],
        ),
        detector=DetectorConfig(
            config_file=str(DETECTOR_CONFIG),
            checkpoint_file=str(DETECTOR_CHECKPOINT),
            score_threshold=0.5,
        ),
        device=device,
        enable_flip_correction=enable_flip,
        max_deviation_threshold=MAX_DEVIATION_THRESHOLD,
        cameras=CAMERAS,
    )
    
    # Load dataset
    print(f"\nLoading dataset...")
    dataset = CARLAStereoPedestrianDataset(
        session_path,
        load_images=True,
        load_depth=False,
        cameras=CAMERAS,
    )
    print(f"  Loaded {len(dataset)} frames")
    
    # Initialize evaluator and statistics
    print(f"\nInitializing models...")
    evaluator = RTMPoseEvaluator(config)
    stats = IncrementalStatistics(config)
    
    # Setup output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if save_vis:
        vis_dir = output_path / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
    
    # Determine frame count
    num_frames = min(len(dataset), max_frames) if max_frames else len(dataset)
    
    print(f"\nStarting evaluation of {num_frames} frames...")
    print("-" * 70)
    
    vis_count = 0
    start_time = time.time()
    
    # Open streaming writer
    with StreamingResultsWriter(output_dir, compress=True) as writer:
        
        for idx in range(num_frames):
            frame = dataset[idx]
            
            for camera in config.cameras:
                # Evaluate frame
                result = evaluator.evaluate_frame(frame, camera)
                
                # Stream to disk
                writer.write(result)
                
                # Update incremental statistics
                stats.update(result, dataset)
                
                # Save visualization
                if save_vis and vis_count < MAX_VISUALIZATIONS:
                    should_vis = (
                        (idx % VIS_INTERVAL == 0) or
                        (result.status != FrameStatus.SUCCESS) or
                        (result.num_flipped > 0) or
                        (result.num_hallucinated > 0)
                    )
                    
                    if should_vis:
                        _save_visualization(
                            frame, result, dataset, vis_dir, vis_count
                        )
                        vis_count += 1
                
                # Free memory
                del result
            
            # Progress logging
            if (idx + 1) % LOG_INTERVAL == 0:
                _log_progress(idx + 1, num_frames, start_time, stats)
            
            # Memory cleanup
            if (idx + 1) % GC_INTERVAL == 0:
                gc.collect()
                if TORCH_AVAILABLE:
                    import torch
                    torch.cuda.empty_cache()
            
            del frame
    
    # Final timing
    total_time = time.time() - start_time
    print(f"\n  Completed in {total_time/60:.1f} minutes ({num_frames/total_time:.1f} fps)")
    
    # Get final statistics
    final_stats = stats.get_statistics()
    
    # Save outputs
    _save_outputs(output_path, final_stats, config, session_path, num_frames, total_time)
    
    # Print summary
    stats.print_summary()
    
    print(f"\nResults saved to: {output_dir}")
    print("=" * 70)
    
    return final_stats


def _save_visualization(frame, result, dataset, vis_dir, vis_count):
    """Save a visualization image."""
    try:
        image = frame.rgb_left if result.camera == 'left' else frame.rgb_right
        if image is None:
            return
        
        vis_img = visualize_evaluation_frame(image, result, dataset)
        
        # Determine category prefix
        if result.status != FrameStatus.SUCCESS:
            prefix = result.status.value
        elif result.num_flipped > 0:
            prefix = "flipped"
        elif result.num_hallucinated > 0:
            prefix = "halluc"
        elif result.num_missing > 0:
            prefix = "missing"
        else:
            prefix = "good"
        
        vis_path = vis_dir / f"{prefix}_{result.frame_id:06d}_{result.camera}.jpg"
        cv2.imwrite(
            str(vis_path),
            cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
    except Exception as e:
        pass  # Skip visualization errors silently


def _log_progress(current, total, start_time, stats):
    """Log progress with metrics."""
    import time
    elapsed = time.time() - start_time
    fps = current / elapsed if elapsed > 0 else 0
    eta = (total - current) / fps / 60 if fps > 0 else 0
    
    recall = stats.total_matched / stats.total_gt if stats.total_gt > 0 else 0
    
    print(f"  [{current:6d}/{total}] {fps:.1f} fps | "
          f"Recall: {recall:.3f} | "
          f"Flip: {stats.total_flipped} | "
          f"Hall: {stats.total_hallucinated} | "
          f"ETA: {eta:.1f}min")


def _save_outputs(output_path, stats, config, session_path, num_frames, total_time):
    """Save statistics, config, and summary."""
    # Statistics
    stats_path = output_path / 'statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(convert_to_serializable(stats), f, indent=2)
    print(f"  Statistics saved to {stats_path}")
    
    # Config
    config_dict = {
        'pose_config': config.pose.config_file,
        'pose_checkpoint': config.pose.checkpoint_file,
        'det_config': config.detector.config_file,
        'det_checkpoint': config.detector.checkpoint_file,
        'enable_flip_correction': config.enable_flip_correction,
        'max_deviation_threshold': config.max_deviation_threshold,
        'session_path': str(session_path),
        'num_frames': num_frames,
        'total_time_seconds': total_time,
        'cameras': config.cameras,
    }
    config_path = output_path / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"  Config saved to {config_path}")
    
    # Human-readable summary
    summary_path = output_path / 'summary.txt'
    _write_summary(summary_path, stats, config_dict)
    print(f"  Summary saved to {summary_path}")


def _write_summary(path, stats, config):
    """Write human-readable summary."""
    overall = stats['overall']
    
    with open(path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RTMPose Evaluation Summary\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Pose model: {Path(config['pose_checkpoint']).stem}\n")
        f.write(f"  Flip correction: {config['enable_flip_correction']}\n")
        f.write(f"  Frames evaluated: {config['num_frames']}\n")
        f.write(f"  Time: {config['total_time_seconds']/60:.1f} minutes\n\n")
        
        f.write("Results:\n")
        f.write(f"  Precision: {overall['precision']:.4f}\n")
        f.write(f"  Recall:    {overall['recall']:.4f}\n")
        f.write(f"  F1:        {overall['f1']:.4f}\n\n")
        
        f.write(f"  Mean Deviation: {overall['mean_deviation']:.2f} px\n")
        f.write(f"  Mean IoU:       {overall['mean_iou']:.4f}\n\n")
        
        f.write(f"  Total GT:          {overall['total_gt']}\n")
        f.write(f"  Matched:           {overall['total_matched']}\n")
        f.write(f"  Hallucinated:      {overall['total_hallucinated']}\n")
        f.write(f"  Missing:           {overall['total_missing']}\n")
        f.write(f"  Flipped:           {overall['total_flipped']} ({overall['flip_rate']:.1%})\n")


# =============================================================================
# Analysis Functions
# =============================================================================

def recompute_statistics(
    results_path: str,
    session_path: str,
    output_dir: Optional[str] = None,
):
    """
    Recompute statistics from a compressed results file.
    """
    print("\nRecomputing statistics from results file")
    print("-" * 60)

    results_path = Path(results_path)
    if output_dir is None:
        output_dir = results_path.parent

    output_dir = Path(output_dir)

    # Load dataset (needed for distance + GT stats)
    dataset = CARLAStereoPedestrianDataset(
        session_path,
        load_images=False,
        load_depth=False,
        cameras=CAMERAS,
    )

    # Load config.json if it exists (preferred)
    config_path = output_dir / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)

        config = EvalConfig(
            pose=PoseModelConfig(
                config_file=cfg['pose_config'],
                checkpoint_file=cfg['pose_checkpoint'],
                model_type=Path(cfg['pose_config']).stem.split('_')[0],
            ),
            detector=DetectorConfig(
                config_file=cfg['det_config'],
                checkpoint_file=cfg['det_checkpoint'],
            ),
            device='cpu',
            enable_flip_correction=cfg['enable_flip_correction'],
            max_deviation_threshold=cfg['max_deviation_threshold'],
            cameras=cfg['cameras'],
        )
    else:
        raise RuntimeError("config.json not found — cannot safely recompute stats")

    stats = IncrementalStatistics(config)
    reader = ResultsReader(results_path)

    count = 0
    for result in reader:
        fr = frame_result_from_dict(result)
        stats.update(fr, dataset)
        count += 1

        if count % 1000 == 0:
            print(f"  Processed {count} results")


    final_stats = stats.get_statistics()

    # Save outputs
    _save_outputs(
        output_dir,
        final_stats,
        config,
        session_path,
        final_stats['overall']['total_frames'],
        final_stats['overall']['total_time_seconds']
        if 'total_time_seconds' in final_stats['overall']
        else 0.0,
    )

    stats.print_summary()

    print(f"\nStatistics successfully recomputed from {results_path}")


def analyze_results(results_path: str, dataset_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze saved results for detailed insights.
    
    Args:
        results_path: Path to results.jsonl.gz
        dataset_path: Optional path to dataset for additional analysis
    
    Returns:
        Analysis dictionary
    """
    print(f"\nAnalyzing results from: {results_path}")
    print("-" * 60)
    
    reader = ResultsReader(results_path)
    
    # Collect data
    total_results = 0
    flip_cases = []
    high_deviation_cases = []
    hallucination_frames = []
    failure_frames = []
    
    per_keypoint_deviation = defaultdict(list)
    per_distance_stats = defaultdict(lambda: {'matched': 0, 'missing': 0, 'deviations': []})
    confidence_by_flip = {'flipped': [], 'normal': []}
    
    for result in reader:
        total_results += 1
        
        # Track flips
        if result.get('num_flipped', 0) > 0:
            flip_cases.append({
                'frame_id': result['frame_id'],
                'camera': result['camera'],
                'num_flipped': result['num_flipped'],
                'matched': [ms for ms in result['matched_skeletons'] if ms['was_flipped']],
            })
        
        # Track high deviation
        for ms in result.get('matched_skeletons', []):
            if ms['mean_deviation'] > 30:
                high_deviation_cases.append({
                    'frame_id': result['frame_id'],
                    'camera': result['camera'],
                    'gt_id': ms['gt_pedestrian_id'],
                    'deviation': ms['mean_deviation'],
                    'was_flipped': ms['was_flipped'],
                    'distance': ms.get('distance_to_camera'),
                })
            
            # Confidence by flip status
            if ms['was_flipped']:
                confidence_by_flip['flipped'].append(ms['mean_confidence'])
            else:
                confidence_by_flip['normal'].append(ms['mean_confidence'])
        
        # Track hallucinations
        if result.get('num_hallucinated', 0) > 0:
            hallucination_frames.append({
                'frame_id': result['frame_id'],
                'camera': result['camera'],
                'count': result['num_hallucinated'],
                'avg_confidence': np.mean([h['mean_confidence'] for h in result['hallucinated_skeletons']]),
            })
        
        # Track failures
        if result['status'] != 'success':
            failure_frames.append({
                'frame_id': result['frame_id'],
                'camera': result['camera'],
                'status': result['status'],
                'error': result.get('error_message', '')[:100],
            })
    
    # Compute analysis
    analysis = {
        'total_results': total_results,
        
        'flip_analysis': {
            'total_flipped_predictions': sum(f['num_flipped'] for f in flip_cases),
            'frames_with_flips': len(flip_cases),
            'flip_rate': sum(f['num_flipped'] for f in flip_cases) / total_results if total_results > 0 else 0,
            'sample_cases': flip_cases[:20],
        },
        
        'high_deviation_analysis': {
            'count_over_30px': len(high_deviation_cases),
            'flipped_high_dev': sum(1 for h in high_deviation_cases if h['was_flipped']),
            'normal_high_dev': sum(1 for h in high_deviation_cases if not h['was_flipped']),
            'worst_cases': sorted(high_deviation_cases, key=lambda x: x['deviation'], reverse=True)[:20],
        },
        
        'hallucination_analysis': {
            'frames_with_hallucinations': len(hallucination_frames),
            'total_hallucinations': sum(h['count'] for h in hallucination_frames),
            'avg_confidence': np.mean([h['avg_confidence'] for h in hallucination_frames]) if hallucination_frames else 0,
            'sample_frames': sorted(hallucination_frames, key=lambda x: x['count'], reverse=True)[:20],
        },
        
        'failure_analysis': {
            'total_failures': len(failure_frames),
            'by_type': dict(defaultdict(int, {f['status']: 1 for f in failure_frames})),
            'sample_failures': failure_frames[:20],
        },
        
        'confidence_analysis': {
            'flipped_mean_confidence': np.mean(confidence_by_flip['flipped']) if confidence_by_flip['flipped'] else 0,
            'normal_mean_confidence': np.mean(confidence_by_flip['normal']) if confidence_by_flip['normal'] else 0,
        },
    }
    
    # Print summary
    print(f"\nAnalysis Summary:")
    print(f"  Total frame-camera results: {total_results}")
    
    print(f"\n  Flip Correction:")
    fa = analysis['flip_analysis']
    print(f"    Flipped predictions: {fa['total_flipped_predictions']}")
    print(f"    Frames with flips:   {fa['frames_with_flips']}")
    
    print(f"\n  High Deviation (>30px):")
    hda = analysis['high_deviation_analysis']
    print(f"    Total cases:    {hda['count_over_30px']}")
    print(f"    Flipped:        {hda['flipped_high_dev']}")
    print(f"    Normal:         {hda['normal_high_dev']}")
    
    print(f"\n  Hallucinations:")
    ha = analysis['hallucination_analysis']
    print(f"    Frames:         {ha['frames_with_hallucinations']}")
    print(f"    Total:          {ha['total_hallucinations']}")
    print(f"    Avg confidence: {ha['avg_confidence']:.3f}")
    
    print(f"\n  Failures:")
    print(f"    Total: {analysis['failure_analysis']['total_failures']}")
    
    print(f"\n  Confidence (Flip vs Normal):")
    ca = analysis['confidence_analysis']
    print(f"    Flipped mean: {ca['flipped_mean_confidence']:.3f}")
    print(f"    Normal mean:  {ca['normal_mean_confidence']:.3f}")
    
    # Save analysis
    output_path = Path(results_path).parent / 'analysis.json'
    with open(output_path, 'w') as f:
        json.dump(convert_to_serializable(analysis), f, indent=2)
    print(f"\n  Analysis saved to: {output_path}")
    
    return analysis

def frame_result_from_dict(d: Dict[str, Any]):
    """
    Reconstruct FrameResult object from serialized dict.
    """
    from rtmpose_eval import (
        FrameResult,
        FrameStatus,
        MatchedSkeleton,
        HallucinatedSkeleton,
        MissingSkeleton,
    )

    matched = [
        MatchedSkeleton(
            gt_pedestrian_id=ms['gt_pedestrian_id'],
            prediction_index=ms['prediction_index'],
            camera=ms['camera'],
            distance_to_camera=ms['distance_to_camera'],
            detection_score=ms['detection_score'],
            pred_bbox=tuple(ms['pred_bbox']),
            gt_bbox=tuple(ms['gt_bbox']) if ms['gt_bbox'] else None,
            bbox_iou=ms['bbox_iou'],
            pred_keypoints=np.array(ms['pred_keypoints']),
            was_flipped=ms['was_flipped'],
            mean_deviation=ms['mean_deviation'],
            mean_deviation_all=ms['mean_deviation_all'],
            mean_confidence=ms['mean_confidence'],
            num_visible_keypoints_gt=ms['num_visible_keypoints_gt'],
            num_occluded_keypoints_gt=ms['num_occluded_keypoints_gt'],
        )
        for ms in d.get('matched_skeletons', [])
    ]

    hallucinated = [
        HallucinatedSkeleton(
            prediction_index=hs['prediction_index'],
            camera=hs['camera'],
            detection_score=hs['detection_score'],
            pred_bbox=tuple(hs['pred_bbox']),
            pred_keypoints=np.array(hs['pred_keypoints']),
            mean_confidence=hs['mean_confidence'],
            max_confidence=hs['max_confidence'],
            min_confidence=hs['min_confidence'],
        )
        for hs in d.get('hallucinated_skeletons', [])
    ]

    missing = [
        MissingSkeleton(
            gt_pedestrian_id=ms['gt_pedestrian_id'],
            camera=ms['camera'],
            distance_to_camera=ms['distance_to_camera'],
            num_visible_keypoints=ms['num_visible_keypoints'],
            num_occluded_keypoints=ms['num_occluded_keypoints'],
            gt_bbox=tuple(ms['gt_bbox']) if ms['gt_bbox'] else None,
        )
        for ms in d.get('missing_skeletons', [])
    ]

    return FrameResult(
        frame_id=d['frame_id'],
        camera=d['camera'],
        timestamp=d['timestamp'],
        status=FrameStatus(d['status']),
        num_gt_pedestrians=d['num_gt_pedestrians'],
        num_detections=d['num_detections'],
        num_predictions=d['num_predictions'],
        num_matched=d['num_matched'],
        num_hallucinated=d['num_hallucinated'],
        num_missing=d['num_missing'],
        num_flipped=d['num_flipped'],
        matched_skeletons=matched,
        hallucinated_skeletons=hallucinated,
        missing_skeletons=missing,
    )


def inspect_frame(
    results_path: str,
    dataset_path: str,
    frame_id: int,
    camera: str = 'left',
    output_path: Optional[str] = None,
):
    """
    Inspect a specific frame in detail.
    
    Shows GT keypoints, predicted keypoints, deviations, etc.
    """
    print(f"\nInspecting Frame {frame_id} ({camera})")
    print("-" * 60)
    
    # Load result
    reader = ResultsReader(results_path)
    result = None
    for r in reader.get_by_frame_id(frame_id):
        if r['camera'] == camera:
            result = r
            break
    
    if result is None:
        print(f"  Result not found for frame {frame_id} {camera}")
        return
    
    # Load dataset for GT
    dataset = CARLAStereoPedestrianDataset(dataset_path, load_images=True)
    frame = dataset.get_frame_by_id(frame_id)
    
    print(f"\n  Status: {result['status']}")
    print(f"  GT pedestrians: {result['num_gt_pedestrians']}")
    print(f"  Detections: {result['num_detections']}")
    print(f"  Matched: {result['num_matched']}")
    print(f"  Hallucinated: {result['num_hallucinated']}")
    print(f"  Missing: {result['num_missing']}")
    print(f"  Flipped: {result['num_flipped']}")
    
    # Matched details
    if result['matched_skeletons']:
        print(f"\n  Matched Skeletons:")
        for i, ms in enumerate(result['matched_skeletons']):
            print(f"\n    [{i}] GT ID: {ms['gt_pedestrian_id']}")
            print(f"        Distance: {ms['distance_to_camera']:.1f}m" if ms['distance_to_camera'] else "        Distance: N/A")
            print(f"        Deviation: {ms['mean_deviation']:.2f}px")
            print(f"        IoU: {ms['bbox_iou']:.3f}" if ms['bbox_iou'] else "        IoU: N/A")
            print(f"        Was flipped: {ms['was_flipped']}")
            print(f"        Mean confidence: {ms['mean_confidence']:.3f}")
            
            # Get GT keypoints for comparison
            gt_ped = next((p for p in frame.annotation.pedestrians if p.id == ms['gt_pedestrian_id']), None)
            if gt_ped:
                gt_kps = gt_ped.get_keypoints_array(camera, include_visibility=True)
                pred_kps = np.array(ms['pred_keypoints'])
                
                print(f"\n        Per-keypoint deviations:")
                for kp_idx, kp_name in enumerate(COCO_KEYPOINTS):
                    gt_vis = int(gt_kps[kp_idx, 2])
                    if gt_vis > 0:
                        dev = np.linalg.norm(gt_kps[kp_idx, :2] - pred_kps[kp_idx, :2])
                        conf = pred_kps[kp_idx, 2]
                        vis_str = "V" if gt_vis == 2 else "O"
                        print(f"          {kp_name:20s}: {dev:6.1f}px  conf:{conf:.2f}  [{vis_str}]")
    
    # Hallucinated details
    if result['hallucinated_skeletons']:
        print(f"\n  Hallucinated Skeletons:")
        for i, hs in enumerate(result['hallucinated_skeletons']):
            print(f"\n    [{i}] Detection score: {hs['detection_score']:.3f}" if hs['detection_score'] else f"\n    [{i}]")
            print(f"        Mean confidence: {hs['mean_confidence']:.3f}")
            print(f"        Bbox: {hs['pred_bbox']}")
    
    # Missing details
    if result['missing_skeletons']:
        print(f"\n  Missing Skeletons:")
        for i, ms in enumerate(result['missing_skeletons']):
            print(f"\n    [{i}] GT ID: {ms['gt_pedestrian_id']}")
            print(f"        Distance: {ms['distance_to_camera']:.1f}m" if ms['distance_to_camera'] else "        Distance: N/A")
            print(f"        Visible keypoints: {ms['num_visible_keypoints']}")
            print(f"        Occluded keypoints: {ms['num_occluded_keypoints']}")
    
    # Save visualization
    if output_path:
        image = frame.rgb_left if camera == 'left' else frame.rgb_right
        if image is not None:
            # Convert result dict back to object-like for visualization
            # (simplified version - just save the raw annotated image)
            from rtmpose_eval import FrameResult, MatchedSkeleton, HallucinatedSkeleton, MissingSkeleton
            
            # Reconstruct frame result object
            matched = [
                MatchedSkeleton(
                    gt_pedestrian_id=ms['gt_pedestrian_id'],
                    prediction_index=ms['prediction_index'],
                    camera=ms['camera'],
                    distance_to_camera=ms['distance_to_camera'],
                    detection_score=ms['detection_score'],
                    pred_bbox=tuple(ms['pred_bbox']),
                    gt_bbox=tuple(ms['gt_bbox']) if ms['gt_bbox'] else None,
                    bbox_iou=ms['bbox_iou'],
                    pred_keypoints=np.array(ms['pred_keypoints']),
                    was_flipped=ms['was_flipped'],
                    mean_deviation=ms['mean_deviation'],
                    mean_deviation_all=ms['mean_deviation_all'],
                    mean_confidence=ms['mean_confidence'],
                    num_visible_keypoints_gt=ms['num_visible_keypoints_gt'],
                    num_occluded_keypoints_gt=ms['num_occluded_keypoints_gt'],
                )
                for ms in result['matched_skeletons']
            ]
            
            hallucinated = [
                HallucinatedSkeleton(
                    prediction_index=hs['prediction_index'],
                    camera=hs['camera'],
                    detection_score=hs['detection_score'],
                    pred_bbox=tuple(hs['pred_bbox']),
                    pred_keypoints=np.array(hs['pred_keypoints']),
                    mean_confidence=hs['mean_confidence'],
                    max_confidence=hs['max_confidence'],
                    min_confidence=hs['min_confidence'],
                )
                for hs in result['hallucinated_skeletons']
            ]
            
            missing = [
                MissingSkeleton(
                    gt_pedestrian_id=ms['gt_pedestrian_id'],
                    camera=ms['camera'],
                    distance_to_camera=ms['distance_to_camera'],
                    num_visible_keypoints=ms['num_visible_keypoints'],
                    num_occluded_keypoints=ms['num_occluded_keypoints'],
                    gt_bbox=tuple(ms['gt_bbox']) if ms['gt_bbox'] else None,
                )
                for ms in result['missing_skeletons']
            ]
            
            fr = FrameResult(
                frame_id=result['frame_id'],
                camera=result['camera'],
                timestamp=result['timestamp'],
                status=FrameStatus(result['status']),
                num_gt_pedestrians=result['num_gt_pedestrians'],
                num_detections=result['num_detections'],
                num_predictions=result['num_predictions'],
                num_matched=result['num_matched'],
                num_hallucinated=result['num_hallucinated'],
                num_missing=result['num_missing'],
                num_flipped=result['num_flipped'],
                matched_skeletons=matched,
                hallucinated_skeletons=hallucinated,
                missing_skeletons=missing,
            )
            
            vis_img = visualize_evaluation_frame(image, fr, dataset)
            cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            print(f"\n  Visualization saved to: {output_path}")


def compare_flip_vs_noflip(flip_stats_path: str, noflip_stats_path: str):
    """
    Compare evaluation results with and without flip correction.
    """
    print("\nComparing Flip vs No-Flip Results")
    print("-" * 60)
    
    with open(flip_stats_path) as f:
        flip_stats = json.load(f)
    
    with open(noflip_stats_path) as f:
        noflip_stats = json.load(f)
    
    flip = flip_stats['overall']
    noflip = noflip_stats['overall']
    
    print(f"\n{'Metric':<25} {'Flip':>12} {'No-Flip':>12} {'Diff':>12}")
    print("-" * 61)
    
    metrics = [
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('F1', 'f1'),
        ('Mean Deviation (px)', 'mean_deviation'),
        ('Mean IoU', 'mean_iou'),
        ('Matched', 'total_matched'),
        ('Hallucinated', 'total_hallucinated'),
        ('Missing', 'total_missing'),
    ]
    
    for name, key in metrics:
        v_flip = flip.get(key, 0)
        v_noflip = noflip.get(key, 0)
        
        if isinstance(v_flip, float):
            diff = v_flip - v_noflip
            print(f"{name:<25} {v_flip:>12.4f} {v_noflip:>12.4f} {diff:>+12.4f}")
        else:
            diff = v_flip - v_noflip
            print(f"{name:<25} {v_flip:>12d} {v_noflip:>12d} {diff:>+12d}")
    
    print(f"\n  Flipped predictions (flip only): {flip.get('total_flipped', 0)}")
    print(f"  Flip rate: {flip.get('flip_rate', 0):.1%}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='RTMPose Streaming Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full evaluation
    python rtmpose_run.py
    
    # Quick test with 100 frames
    python rtmpose_run.py --max-frames 100
    
    # Use different pose model
    python rtmpose_run.py --model rtmpose-m
    
    # Disable flip correction (for comparison)
    python rtmpose_run.py --no-flip --output ./eval_noflip
    
    # Analyze existing results
    python rtmpose_run.py --analyze ./eval_output/results.jsonl.gz
    
    # Inspect specific frame
    python rtmpose_run.py --inspect 1234 --camera left
    
    # Compare flip vs no-flip results
    python rtmpose_run.py --compare ./eval_flip/statistics.json ./eval_noflip/statistics.json
        """
    )
    
    # Evaluation arguments
    parser.add_argument('--session', type=str, default=SESSION_PATH,
                        help='Path to CARLA session')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help='Output directory')
    parser.add_argument('--model', type=str, default=SELECTED_MODEL,
                        choices=list(POSE_MODELS.keys()),
                        help='Pose model to use')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum frames to evaluate')
    parser.add_argument('--no-flip', action='store_true',
                        help='Disable flip correction')
    parser.add_argument('--no-vis', action='store_true',
                        help='Disable visualizations')
    parser.add_argument('--device', type=str, default=DEVICE,
                        help='Device for inference')
    
    # Analysis arguments
    parser.add_argument(
        '--recompute-stats',
        type=str,
        default=None,
        metavar='RESULTS_FILE',
        help='Recreate statistics.json and summary.txt from results.jsonl.gz')
    parser.add_argument('--analyze', type=str, default=None,
                        help='Analyze existing results file')
    parser.add_argument('--inspect', type=int, default=None,
                        help='Inspect specific frame ID')
    parser.add_argument('--camera', type=str, default='left',
                        choices=['left', 'right'],
                        help='Camera for inspection')
    parser.add_argument('--compare', type=str, nargs=2, default=None,
                        metavar=('FLIP_STATS', 'NOFLIP_STATS'),
                        help='Compare two statistics files')
    
    args = parser.parse_args()
    
    # Mode selection
    if args.analyze:
        # Analysis mode
        analyze_results(args.analyze, args.session)
        
    elif args.recompute_stats:
        recompute_statistics(
            results_path=args.recompute_stats,
            session_path=args.session,
            output_dir=args.output)   
    
    elif args.inspect is not None:
        # Inspection mode
        results_path = Path(args.output) / 'results.jsonl.gz'
        output_img = f'inspect_{args.inspect}_{args.camera}.jpg'
        inspect_frame(
            str(results_path),
            args.session,
            args.inspect,
            args.camera,
            output_img
        )
    
    elif args.compare:
        # Comparison mode
        compare_flip_vs_noflip(args.compare[0], args.compare[1])
    
    else:
        # Evaluation mode
        run_evaluation(
            session_path=args.session,
            output_dir=args.output,
            pose_model=args.model,
            max_frames=args.max_frames,
            enable_flip=not args.no_flip,
            save_vis=not args.no_vis,
            device=args.device,
        )


if __name__ == '__main__':
    main()