#!/usr/bin/env python3
"""
Comprehensive Evaluation and Visualization Script for PBPL

This script generates all visualizations and analyses from saved checkpoints:
- Individual loss component plots (position, per-keypoint, bone, correction penalty, no-harm)
- Horizon error visualizations (position, per-joint, speed, direction)
- Updated threshold analysis with correct recall calculation
- Baseline comparisons at pareto points (Low vs High Recall)

All outputs are saved to ./Evaluations directory.

Usage:
    python evaluate_and_visualize.py --input-dir Outputs --output-dir Evaluations
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Check for required packages
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')


# ==============================================================================
# CONFIGURATION
# ==============================================================================

KEYPOINT_NAMES = [
    'Nose', 'L.Eye', 'R.Eye', 'L.Ear', 'R.Ear',
    'L.Shoulder', 'R.Shoulder', 'L.Elbow', 'R.Elbow',
    'L.Wrist', 'R.Wrist', 'L.Hip', 'R.Hip',
    'L.Knee', 'R.Knee', 'L.Ankle', 'R.Ankle'
]

ARCHITECTURE_COLORS = {
    'mlp': '#2ecc71',      # Green
    'gru': '#3498db',      # Blue
    'lstm': '#e67e22',     # Orange
    'mamba': '#9b59b6',    # Purple
    'kalman': '#e74c3c'    # Red
}

LOSS_COLORS = {
    'position_loss': '#3498db',       # Blue
    'per_keypoint_loss': '#2ecc71',   # Green
    'bone_loss': '#e67e22',           # Orange
    'correction_penalty': '#9b59b6',  # Purple
    'no_harm_penalty': '#e74c3c',     # Red
    'train_loss': '#34495e'           # Dark gray
}

ARCHITECTURES = ['mlp', 'gru', 'lstm', 'mamba', 'kalman']


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_history(input_dir: Path, architecture: str) -> Optional[Dict]:
    """Load training history for an architecture."""
    history_path = input_dir / architecture / f'{architecture}_history.json'
    if not history_path.exists():
        history_path = input_dir / f'{architecture}_history.json'
    
    if history_path.exists():
        with open(history_path) as f:
            return json.load(f)
    return None


def load_comparison_results(input_dir: Path) -> Optional[Dict]:
    """Load comparison results."""
    results_path = input_dir / 'comparison_results.json'
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None


def load_threshold_analysis(input_dir: Path) -> Optional[List[Dict]]:
    """Load threshold analysis results."""
    analysis_path = input_dir / 'threshold_analysis.json'
    if analysis_path.exists():
        with open(analysis_path) as f:
            return json.load(f)
    return None


def load_per_joint_errors(input_dir: Path) -> Optional[Dict]:
    """Load per-joint error data."""
    errors_path = input_dir / 'per_joint_errors.json'
    if errors_path.exists():
        with open(errors_path) as f:
            return json.load(f)
    return None


def load_direction_speed_errors(input_dir: Path) -> Optional[Dict]:
    """Load direction and speed error data."""
    errors_path = input_dir / 'direction_speed_errors.json'
    if errors_path.exists():
        with open(errors_path) as f:
            return json.load(f)
    return None


# ==============================================================================
# LOSS COMPONENT VISUALIZATION
# ==============================================================================

def plot_loss_components(history: Dict, arch_name: str, output_dir: Path, show: bool = False):
    """Plot individual loss components for an architecture.
    
    Creates separate plots for each loss component:
    - Position loss
    - Per-keypoint loss
    - Bone loss
    - Correction penalty
    - No-harm penalty
    """
    stage1 = history.get('stage1', history)
    
    # Check which loss components are available
    loss_components = ['position_loss', 'per_keypoint_loss', 'bone_loss', 
                       'correction_penalty', 'no_harm_penalty']
    available_losses = {k: stage1.get(k, []) for k in loss_components if stage1.get(k)}
    
    if not available_losses:
        print(f"  ⚠ No individual loss components found for {arch_name}")
        return
    
    num_plots = len(available_losses) + 1  # +1 for combined plot
    num_cols = min(3, num_plots)
    num_rows = (num_plots + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 5*num_rows))
    axes = axes.flatten() if num_plots > 1 else [axes]
    
    # Plot each loss component
    for idx, (loss_name, values) in enumerate(available_losses.items()):
        ax = axes[idx]
        epochs = range(1, len(values) + 1)
        color = LOSS_COLORS.get(loss_name, 'gray')
        
        ax.plot(epochs, values, '-', color=color, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(loss_name.replace('_', ' ').title(), fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        if values:
            final_val = values[-1]
            min_val = min(values)
            ax.axhline(y=min_val, color=color, linestyle='--', alpha=0.5, 
                       label=f'Min: {min_val:.4f}')
            ax.legend(loc='upper right')
    
    # Combined plot
    ax = axes[len(available_losses)]
    for loss_name, values in available_losses.items():
        if values:
            epochs = range(1, len(values) + 1)
            color = LOSS_COLORS.get(loss_name, 'gray')
            label = loss_name.replace('_', ' ').title()
            ax.plot(epochs, values, '-', color=color, linewidth=1.5, label=label, alpha=0.7)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('All Loss Components Combined', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for idx in range(len(available_losses) + 1, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'{arch_name.upper()} - Loss Component Breakdown', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / f'{arch_name}_loss_components.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_all_architectures_loss_components(histories: Dict[str, Dict], output_dir: Path, show: bool = False):
    """Plot loss components comparison across all architectures."""
    loss_components = ['position_loss', 'per_keypoint_loss', 'bone_loss', 
                       'correction_penalty', 'no_harm_penalty']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Plot each loss component
    for idx, loss_name in enumerate(loss_components):
        ax = axes[idx]
        
        for arch, history in histories.items():
            stage1 = history.get('stage1', history)
            values = stage1.get(loss_name, [])
            
            if values:
                epochs = range(1, len(values) + 1)
                color = ARCHITECTURE_COLORS.get(arch, 'gray')
                ax.plot(epochs, values, '-', color=color, linewidth=1.5, 
                       label=arch.upper(), alpha=0.8)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(loss_name.replace('_', ' ').title(), fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Total loss comparison
    ax = axes[5]
    for arch, history in histories.items():
        stage1 = history.get('stage1', history)
        values = stage1.get('train_loss', [])
        
        if values:
            epochs = range(1, len(values) + 1)
            color = ARCHITECTURE_COLORS.get(arch, 'gray')
            ax.plot(epochs, values, '-', color=color, linewidth=1.5, 
                   label=arch.upper(), alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Training Loss', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Loss Component Comparison Across Architectures', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'all_architectures_loss_components.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    
    if show:
        plt.show()
    plt.close()


# ==============================================================================
# HORIZON ERROR VISUALIZATION
# ==============================================================================

def plot_horizon_errors(histories: Dict[str, Dict], output_dir: Path, show: bool = False):
    """Plot error metrics across prediction horizons for all architectures."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Position error by horizon
    ax = axes[0, 0]
    for arch, history in histories.items():
        stage2 = history.get('stage2', {})
        horizons = stage2.get('horizons', [])
        
        if horizons:
            h_nums = [h.get('horizon', i+1) for i, h in enumerate(horizons)]
            val_losses = [h.get('best_val_loss', 0) for h in horizons]
            color = ARCHITECTURE_COLORS.get(arch, 'gray')
            ax.plot(h_nums, val_losses, 'o-', color=color, linewidth=2, 
                   markersize=8, label=arch.upper())
    
    ax.set_xlabel('Prediction Horizon (frames)')
    ax.set_ylabel('Best Validation Loss')
    ax.set_title('Position Error by Prediction Horizon', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Per-keypoint error by horizon (from stage1 data)
    ax = axes[0, 1]
    for arch, history in histories.items():
        stage1 = history.get('stage1', history)
        values = stage1.get('val_per_keypoint_error', stage1.get('train_per_keypoint_error', []))
        
        if values:
            epochs = range(1, len(values) + 1)
            color = ARCHITECTURE_COLORS.get(arch, 'gray')
            ax.plot(epochs, values, '-', color=color, linewidth=1.5, 
                   label=arch.upper(), alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Per-Keypoint Error (m)')
    ax.set_title('Per-Keypoint Error During Training', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Improvement percentage over time
    ax = axes[1, 0]
    for arch, history in histories.items():
        stage1 = history.get('stage1', history)
        values = stage1.get('improvement_pct', [])
        
        if values:
            epochs = range(1, len(values) + 1)
            color = ARCHITECTURE_COLORS.get(arch, 'gray')
            ax.plot(epochs, values, '-', color=color, linewidth=1.5, 
                   label=arch.upper(), alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Improvement over Baseline (%)')
    ax.set_title('Improvement Percentage Over Training', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final comparison bar chart
    ax = axes[1, 1]
    archs = []
    final_errors = []
    colors = []
    
    for arch, history in histories.items():
        stage1 = history.get('stage1', history)
        val_error = stage1.get('val_error', [])
        
        if val_error:
            archs.append(arch.upper())
            final_errors.append(min(val_error))
            colors.append(ARCHITECTURE_COLORS.get(arch, 'gray'))
    
    if archs:
        bars = ax.bar(archs, final_errors, color=colors, alpha=0.8)
        ax.set_ylabel('Best Validation Error (m)')
        ax.set_title('Best Validation Error by Architecture', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, final_errors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Horizon and Training Error Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'horizon_errors.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_direction_speed_errors(dir_speed_data: Dict, output_dir: Path, show: bool = False):
    """Plot direction and speed error analysis."""
    if not dir_speed_data:
        print("  ⚠ No direction/speed data available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Direction error distribution
    ax = axes[0]
    if 'direction' in dir_speed_data:
        dir_data = dir_speed_data['direction']
        mean_err = dir_data.get('mean_error_deg', 0)
        std_err = dir_data.get('std_error_deg', 0)
        median_err = dir_data.get('median_error_deg', 0)
        count = dir_data.get('count', 0)
        
        # Create distribution plot
        x = np.linspace(0, 180, 100)
        if std_err > 0:
            y = (1 / (std_err * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_err) / std_err) ** 2)
        else:
            y = np.zeros_like(x)
        
        ax.fill_between(x, y, alpha=0.3, color='#3498db')
        ax.plot(x, y, '-', color='#3498db', linewidth=2)
        ax.axvline(x=mean_err, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_err:.1f}°')
        ax.axvline(x=median_err, color='green', linestyle='--', linewidth=2, label=f'Median: {median_err:.1f}°')
        
        ax.set_xlabel('Direction Error (degrees)')
        ax.set_ylabel('Density')
        ax.set_title(f'Direction Error Distribution (n={count})', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 180)
    
    # Speed error distribution
    ax = axes[1]
    if 'speed' in dir_speed_data:
        speed_data = dir_speed_data['speed']
        mean_err = speed_data.get('mean_error_mps', 0)
        std_err = speed_data.get('std_error_mps', 0)
        median_err = speed_data.get('median_error_mps', 0)
        count = speed_data.get('count', 0)
        
        x = np.linspace(0, mean_err + 3*std_err, 100)
        if std_err > 0:
            y = (1 / (std_err * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_err) / std_err) ** 2)
        else:
            y = np.zeros_like(x)
        
        ax.fill_between(x, y, alpha=0.3, color='#2ecc71')
        ax.plot(x, y, '-', color='#2ecc71', linewidth=2)
        ax.axvline(x=mean_err, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_err:.2f} m/s')
        ax.axvline(x=median_err, color='green', linestyle='--', linewidth=2, label=f'Median: {median_err:.2f} m/s')
        
        ax.set_xlabel('Speed Error (m/s)')
        ax.set_ylabel('Density')
        ax.set_title(f'Speed Error Distribution (n={count})', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_dir / 'direction_speed_errors.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    
    if show:
        plt.show()
    plt.close()


# ==============================================================================
# THRESHOLD ANALYSIS VISUALIZATION
# ==============================================================================

def find_pareto_points(results: List[Dict]) -> Tuple[List[int], Dict, Dict]:
    """Find Pareto frontier points and identify practical Low/High Recall pareto points.
    
    Returns:
        pareto_idx: List of indices on the Pareto frontier
        low_recall_point: A practical lower recall point with significantly better error
        high_recall_point: Best recall point with acceptable error
    """
    recalls = np.array([r['recall'] for r in results])
    errors = np.array([r['mean_position_error'] for r in results])
    
    # Handle NaN errors
    finite_errors = errors[np.isfinite(errors)]
    if finite_errors.size > 0:
        errors = np.where(np.isfinite(errors), errors, np.nanmax(finite_errors) * 1.5)
    else:
        errors = np.where(np.isfinite(errors), errors, 1e6)  # Fallback if all are non-finite
    
    # Find Pareto frontier (higher recall, lower error is better)
    pareto_idx = []
    for i in range(len(results)):
        is_pareto = True
        for j in range(len(results)):
            if i != j:
                if recalls[j] >= recalls[i] and errors[j] <= errors[i]:
                    if recalls[j] > recalls[i] or errors[j] < errors[i]:
                        is_pareto = False
                        break
        if is_pareto:
            pareto_idx.append(i)
    
    # Find practical Low Recall and High Recall pareto points
    # High recall: highest recall on pareto frontier (typically 100%)
    # Low recall: find a practical point with recall > 0.5 (50%) but significantly lower error
    
    if pareto_idx:
        # High recall point: highest recall on pareto frontier
        high_recall_point_idx = max(pareto_idx, key=lambda i: recalls[i])
        high_recall_point = results[high_recall_point_idx]
        
        # Low recall point: find a point with decent recall (>50%) but lower error
        # Sort pareto points by error (ascending) and find first with recall > 0.5
        pareto_by_error = sorted(pareto_idx, key=lambda i: errors[i])
        
        low_recall_point = None
        for idx in pareto_by_error:
            if recalls[idx] >= 0.50:  # At least 50% recall for practical use
                low_recall_point = results[idx]
                break
        
        # If no point with >50% recall found, use one with >30% recall
        if low_recall_point is None:
            for idx in pareto_by_error:
                if recalls[idx] >= 0.30:
                    low_recall_point = results[idx]
                    break
        
        # Final fallback: use lowest error point regardless of recall
        if low_recall_point is None:
            low_recall_point_idx = min(pareto_idx, key=lambda i: errors[i])
            low_recall_point = results[low_recall_point_idx]
        
        return pareto_idx, low_recall_point, high_recall_point
    
    return [], None, None


def plot_threshold_analysis(results: List[Dict], output_dir: Path, show: bool = False):
    """Create comprehensive threshold analysis visualization.
    
    Generates:
    1. Recall vs Error colored by confidence
    2. Recall vs Error colored by min_keypoints
    3. Pareto frontier with Low/High recall points marked
    4. Per-loss-type analysis (separate plot)
    """
    if not results:
        print("  ⚠ No threshold analysis results to plot")
        return
    
    # Main threshold analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data
    recalls = np.array([r['recall'] for r in results])
    errors = np.array([r['mean_position_error'] for r in results])
    confs = np.array([r['confidence_threshold'] for r in results])
    kps = np.array([r['min_keypoints'] for r in results])
    precisions = np.array([r['precision'] for r in results])
    f1s = np.array([r['f1'] for r in results])
    
    # Handle NaN errors
    finite_mask = np.isfinite(errors)
    if finite_mask.any():
        max_error = errors[finite_mask].max()
        errors = np.where(finite_mask, errors, max_error * 1.5)
    
    # Find Pareto points
    pareto_idx, low_recall_pt, high_recall_pt = find_pareto_points(results)
    
    # Plot 1: Recall vs Error colored by confidence
    ax = axes[0, 0]
    unique_confs = sorted(set(confs))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_confs)))
    conf_color_map = dict(zip(unique_confs, colors))
    
    for conf in unique_confs:
        mask = confs == conf
        ax.scatter(recalls[mask], errors[mask], c=[conf_color_map[conf]], 
                  s=100, alpha=0.7, label=f'conf={conf:.1f}', edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Mean Position Error (m)', fontsize=12)
    ax.set_title('Recall vs Error\n(colored by confidence threshold)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Recall vs Error colored by min_keypoints
    ax = axes[0, 1]
    unique_kps = sorted(set(kps))
    colors = plt.cm.plasma(np.linspace(0, 1, len(unique_kps)))
    kp_color_map = dict(zip(unique_kps, colors))
    
    for kp in unique_kps:
        mask = kps == kp
        ax.scatter(recalls[mask], errors[mask], c=[kp_color_map[kp]], 
                  s=100, alpha=0.7, label=f'min_kp={kp}', edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Mean Position Error (m)', fontsize=12)
    ax.set_title('Recall vs Error\n(colored by min_keypoints)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Pareto frontier with Low/High recall points
    ax = axes[1, 0]
    
    # All points
    ax.scatter(recalls, errors, c='lightgray', s=60, alpha=0.5, label='All configs')
    
    # Pareto frontier
    if pareto_idx:
        pareto_recalls = recalls[pareto_idx]
        pareto_errors = errors[pareto_idx]
        
        sort_idx = np.argsort(pareto_recalls)
        ax.plot(pareto_recalls[sort_idx], pareto_errors[sort_idx], 
               'r--', linewidth=2, label='Pareto Frontier', zorder=2)
        ax.scatter(pareto_recalls, pareto_errors, c='red', s=150, 
                  marker='*', zorder=3, label='Pareto Optimal')
    
    # Mark Low/High recall points
    if low_recall_pt:
        ax.scatter([low_recall_pt['recall']], [low_recall_pt['mean_position_error']], 
                  c='blue', s=300, marker='s', zorder=4, 
                  label=f"Low Recall (conf={low_recall_pt['confidence_threshold']}, kp={low_recall_pt['min_keypoints']})")
    
    if high_recall_pt:
        ax.scatter([high_recall_pt['recall']], [high_recall_pt['mean_position_error']], 
                  c='green', s=300, marker='^', zorder=4,
                  label=f"High Recall (conf={high_recall_pt['confidence_threshold']}, kp={high_recall_pt['min_keypoints']})")
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Mean Position Error (m)', fontsize=12)
    ax.set_title('Pareto Frontier with Operating Points', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Precision-Recall-F1 comparison
    ax = axes[1, 1]
    
    # Group by confidence threshold
    for conf in unique_confs:
        mask = confs == conf
        color = conf_color_map[conf]
        ax.scatter(recalls[mask], precisions[mask], c=[color], 
                  s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add iso-F1 curves
    for f1_val in [0.7, 0.8, 0.9]:
        r_vals = np.linspace(0.01, 1, 100)
        p_vals = f1_val * r_vals / (2 * r_vals - f1_val)
        valid_mask = (p_vals > 0) & (p_vals <= 1)
        ax.plot(r_vals[valid_mask], p_vals[valid_mask], '--', 
               color='gray', alpha=0.5, linewidth=1)
        # Label the F1 curve
        mid_idx = len(r_vals[valid_mask]) // 2
        if mid_idx > 0:
            ax.annotate(f'F1={f1_val}', 
                       (r_vals[valid_mask][mid_idx], p_vals[valid_mask][mid_idx]),
                       fontsize=8, color='gray')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Threshold Analysis - Recall/Error Tradeoff', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'threshold_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    
    if show:
        plt.show()
    plt.close()
    
    # Return pareto points for baseline comparison
    return low_recall_pt, high_recall_pt


def plot_per_loss_type_threshold(results: List[Dict], output_dir: Path, show: bool = False):
    """Create separate threshold analysis plots for different metrics."""
    if not results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    recalls = np.array([r['recall'] for r in results])
    errors = np.array([r['mean_position_error'] for r in results])
    std_errors = np.array([r['std_position_error'] for r in results])
    median_errors = np.array([r['median_position_error'] for r in results])
    confs = np.array([r['confidence_threshold'] for r in results])
    kps = np.array([r['min_keypoints'] for r in results])
    
    # Handle NaN/Inf - replace non-finite values with fallback
    def handle_nonfinite(arr):
        """Replace non-finite values with max_val * 1.5 or 1e6 as fallback."""
        finite_mask = np.isfinite(arr)
        if finite_mask.any():
            max_val = arr[finite_mask].max()
            arr = np.where(finite_mask, arr, max_val * 1.5)
        else:
            arr = np.where(finite_mask, arr, 1e6)
        return arr
    
    errors = handle_nonfinite(errors)
    std_errors = handle_nonfinite(std_errors)
    median_errors = handle_nonfinite(median_errors)
    
    unique_confs = sorted(set(confs))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_confs)))
    conf_color_map = dict(zip(unique_confs, colors))
    
    # Plot 1: Mean Position Error
    ax = axes[0, 0]
    for conf in unique_confs:
        mask = confs == conf
        ax.scatter(recalls[mask], errors[mask], c=[conf_color_map[conf]], 
                  s=80, alpha=0.7, label=f'conf={conf:.1f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Mean Position Error (m)')
    ax.set_title('Mean Position Error vs Recall', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Std Position Error
    ax = axes[0, 1]
    for conf in unique_confs:
        mask = confs == conf
        ax.scatter(recalls[mask], std_errors[mask], c=[conf_color_map[conf]], 
                  s=80, alpha=0.7, label=f'conf={conf:.1f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Std Position Error (m)')
    ax.set_title('Position Error Std Dev vs Recall', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Median Position Error
    ax = axes[1, 0]
    for conf in unique_confs:
        mask = confs == conf
        ax.scatter(recalls[mask], median_errors[mask], c=[conf_color_map[conf]], 
                  s=80, alpha=0.7, label=f'conf={conf:.1f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Median Position Error (m)')
    ax.set_title('Median Position Error vs Recall', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Number of samples
    ax = axes[1, 1]
    num_samples = np.array([r['num_error_samples'] for r in results])
    for conf in unique_confs:
        mask = confs == conf
        ax.scatter(recalls[mask], num_samples[mask], c=[conf_color_map[conf]], 
                  s=80, alpha=0.7, label=f'conf={conf:.1f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Sample Count vs Recall', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Per-Metric Threshold Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'threshold_analysis_per_metric.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    
    if show:
        plt.show()
    plt.close()


# ==============================================================================
# BASELINE COMPARISON AT PARETO POINTS
# ==============================================================================

def plot_baseline_comparison_pareto(comparison_results: Dict, low_recall_pt: Dict, 
                                    high_recall_pt: Dict, output_dir: Path, show: bool = False):
    """Compare baseline vs corrected at Low and High recall pareto points."""
    if not comparison_results:
        print("  ⚠ No comparison results available for baseline comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    archs = list(comparison_results.keys())
    colors = [ARCHITECTURE_COLORS.get(a, 'gray') for a in archs]
    
    # Plot 1: Best validation error vs baseline
    ax = axes[0, 0]
    x = np.arange(len(archs))
    width = 0.35
    
    baseline_errors = [comparison_results[a]['baseline_error'] for a in archs]
    best_errors = [comparison_results[a]['best_val_error'] for a in archs]
    
    ax.bar(x - width/2, baseline_errors, width, label='Baseline', color='lightgray', edgecolor='black')
    ax.bar(x + width/2, best_errors, width, label='Corrected', color=colors, edgecolor='black', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in archs])
    ax.set_ylabel('Validation Error (m)')
    ax.set_title('Baseline vs Corrected Error', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Improvement percentage
    ax = axes[0, 1]
    improvements = [(b - c) / b * 100 for b, c in zip(baseline_errors, best_errors)]
    bars = ax.bar([a.upper() for a in archs], improvements, color=colors, alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Improvement over Baseline', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Pareto point comparison
    ax = axes[1, 0]
    
    if low_recall_pt and high_recall_pt:
        labels = ['Low Recall\n(Best Accuracy)', 'High Recall\n(Best Coverage)']
        mean_errors = [low_recall_pt['mean_position_error'], high_recall_pt['mean_position_error']]
        recalls = [low_recall_pt['recall'], high_recall_pt['recall']]
        
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, mean_errors, width, label='Mean Error (m)', color='#3498db')
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, recalls, width, label='Recall', color='#2ecc71')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Mean Position Error (m)', color='#3498db')
        ax2.set_ylabel('Recall', color='#2ecc71')
        ax.set_title('Pareto Operating Points Comparison', fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars1, mean_errors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, color='#3498db')
        for bar, val in zip(bars2, recalls):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, color='#2ecc71')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax.text(0.5, 0.5, 'Pareto points not available', ha='center', va='center', fontsize=12)
    
    # Plot 4: Configuration details for pareto points
    ax = axes[1, 1]
    ax.axis('off')
    
    if low_recall_pt and high_recall_pt:
        table_data = [
            ['Metric', 'Low Recall', 'High Recall'],
            ['Confidence Threshold', f"{low_recall_pt['confidence_threshold']:.1f}", f"{high_recall_pt['confidence_threshold']:.1f}"],
            ['Min Keypoints', f"{low_recall_pt['min_keypoints']}", f"{high_recall_pt['min_keypoints']}"],
            ['Recall', f"{low_recall_pt['recall']:.4f}", f"{high_recall_pt['recall']:.4f}"],
            ['Precision', f"{low_recall_pt['precision']:.4f}", f"{high_recall_pt['precision']:.4f}"],
            ['F1 Score', f"{low_recall_pt['f1']:.4f}", f"{high_recall_pt['f1']:.4f}"],
            ['Mean Error (m)', f"{low_recall_pt['mean_position_error']:.4f}", f"{high_recall_pt['mean_position_error']:.4f}"],
            ['Median Error (m)', f"{low_recall_pt['median_position_error']:.4f}", f"{high_recall_pt['median_position_error']:.4f}"],
            ['Sample Count', f"{low_recall_pt['num_error_samples']}", f"{high_recall_pt['num_error_samples']}"],
        ]
        
        table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                        colWidths=[0.35, 0.325, 0.325])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        ax.set_title('Pareto Point Configuration Details', fontweight='bold', pad=20)
    
    plt.suptitle('Baseline Comparison at Pareto Operating Points', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'baseline_comparison_pareto.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    
    if show:
        plt.show()
    plt.close()


# ==============================================================================
# PER-JOINT ERROR VISUALIZATION
# ==============================================================================

def plot_per_joint_errors(per_joint_data: Dict, output_dir: Path, show: bool = False):
    """Plot comprehensive per-joint error analysis."""
    if not per_joint_data:
        print("  ⚠ No per-joint error data available")
        return
    
    # Check for blended_skeleton data (main joint errors)
    if 'blended_skeleton' not in per_joint_data:
        print("  ⚠ No blended_skeleton data in per-joint errors")
        return
    
    joint_data = per_joint_data['blended_skeleton']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data
    joints = list(joint_data.keys())
    means = [joint_data[j]['mean'] for j in joints]
    stds = [joint_data[j].get('std', 0) for j in joints]
    medians = [joint_data[j].get('median', 0) for j in joints]
    counts = [joint_data[j].get('count', 0) for j in joints]
    
    # Plot 1: Mean error by joint
    ax = axes[0, 0]
    x = np.arange(len(joints))
    bars = ax.bar(x, means, yerr=stds, capsize=3, alpha=0.8, color='#3498db')
    ax.set_xticks(x)
    ax.set_xticklabels([j.replace('_', '\n') for j in joints], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Error (m)')
    ax.set_title('Mean Per-Joint Error with Std Dev', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Median error by joint
    ax = axes[0, 1]
    bars = ax.bar(x, medians, alpha=0.8, color='#2ecc71')
    ax.set_xticks(x)
    ax.set_xticklabels([j.replace('_', '\n') for j in joints], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Median Error (m)')
    ax.set_title('Median Per-Joint Error', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Sample count by joint
    ax = axes[1, 0]
    bars = ax.bar(x, counts, alpha=0.8, color='#e67e22')
    ax.set_xticks(x)
    ax.set_xticklabels([j.replace('_', '\n') for j in joints], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Sample Count')
    ax.set_title('Sample Count per Joint', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Sorted error ranking
    ax = axes[1, 1]
    sorted_indices = np.argsort(means)[::-1]
    sorted_joints = [joints[i] for i in sorted_indices]
    sorted_means = [means[i] for i in sorted_indices]
    
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(sorted_joints)))
    bars = ax.barh(range(len(sorted_joints)), sorted_means, color=colors, alpha=0.8)
    ax.set_yticks(range(len(sorted_joints)))
    ax.set_yticklabels([j.replace('_', ' ').title() for j in sorted_joints], fontsize=9)
    ax.set_xlabel('Mean Error (m)')
    ax.set_title('Joints Ranked by Error (Worst to Best)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, sorted_means)):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', ha='left', va='center', fontsize=8)
    
    plt.suptitle('Per-Joint Error Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'per_joint_errors.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    
    if show:
        plt.show()
    plt.close()


# ==============================================================================
# SUMMARY JSON GENERATION
# ==============================================================================

def generate_evaluation_summary(histories: Dict, comparison_results: Dict, 
                                 threshold_results: List[Dict], low_recall_pt: Dict,
                                 high_recall_pt: Dict, output_dir: Path):
    """Generate comprehensive evaluation summary JSON."""
    from datetime import datetime
    summary = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'evaluation_directory': str(Path.cwd()),
        'architectures_evaluated': list(histories.keys()),
        'architecture_comparison': {},
        'pareto_points': {
            'low_recall': low_recall_pt,
            'high_recall': high_recall_pt
        },
        'loss_components_tracked': ['position_loss', 'per_keypoint_loss', 'bone_loss', 
                                    'correction_penalty', 'no_harm_penalty'],
    }
    
    # Add per-architecture summary
    for arch, history in histories.items():
        stage1 = history.get('stage1', history)
        
        arch_summary = {
            'total_epochs': len(stage1.get('train_loss', [])),
            'final_train_loss': stage1.get('train_loss', [None])[-1],
            'best_val_error': min(stage1.get('val_error', [float('inf')])) if stage1.get('val_error') else None,
            'final_improvement_pct': stage1.get('improvement_pct', [None])[-1],
        }
        
        # Add comparison results if available
        if comparison_results and arch in comparison_results:
            arch_summary.update({
                'num_params': comparison_results[arch].get('num_params'),
                'baseline_error': comparison_results[arch].get('baseline_error'),
                'evaluation_metrics': comparison_results[arch].get('evaluation', {}),
            })
        
        summary['architecture_comparison'][arch] = arch_summary
    
    # Add threshold analysis summary
    if threshold_results:
        summary['threshold_analysis'] = {
            'num_configurations_tested': len(threshold_results),
            'confidence_range': [min(r['confidence_threshold'] for r in threshold_results),
                                max(r['confidence_threshold'] for r in threshold_results)],
            'keypoint_range': [min(r['min_keypoints'] for r in threshold_results),
                              max(r['min_keypoints'] for r in threshold_results)],
        }
    
    # Save summary
    summary_path = output_dir / 'evaluation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  ✓ Saved: {summary_path}")
    
    return summary


# ==============================================================================
# DOCUMENTATION GENERATION
# ==============================================================================

def generate_analysis_report(summary: Dict, histories: Dict, comparison_results: Dict,
                             low_recall_pt: Dict, high_recall_pt: Dict, output_dir: Path):
    """Generate updated analysis report markdown."""
    docs_dir = output_dir / 'docs'
    docs_dir.mkdir(exist_ok=True)
    
    report = """# Evaluation Analysis Report

This document provides a detailed analysis of the PBPL model evaluations based on saved checkpoints.

## Executive Summary

| Architecture | Best Val Error | Parameters | Improvement | 
|-------------|----------------|------------|-------------|
"""
    
    # Add architecture rows
    if comparison_results:
        for arch in sorted(comparison_results.keys()):
            data = comparison_results[arch]
            best_val = data.get('best_val_error', 'N/A')
            params = data.get('num_params', 'N/A')
            baseline = data.get('baseline_error', 0)
            
            if isinstance(best_val, (int, float)) and isinstance(baseline, (int, float)) and baseline > 0:
                improvement = (baseline - best_val) / baseline * 100
                imp_str = f"{improvement:.1f}%"
            else:
                imp_str = 'N/A'
            
            best_val_str = f"{best_val:.4f} m" if isinstance(best_val, (int, float)) else str(best_val)
            params_str = f"{params:,}" if isinstance(params, int) else str(params)
            
            report += f"| **{arch.upper()}** | {best_val_str} | {params_str} | {imp_str} |\n"
    
    report += """
---

## Loss Component Analysis

The training tracked five individual loss components:

1. **Position Loss**: Centroid position accuracy
2. **Per-Keypoint Loss**: Individual joint accuracy  
3. **Bone Loss**: Skeletal structure preservation
4. **Correction Penalty**: Regularization on correction magnitude
5. **No-Harm Penalty**: Prevents degradation vs baseline

See `all_architectures_loss_components.png` for the complete breakdown.

---

## Pareto Operating Points

Two key operating points were identified on the Pareto frontier:

"""
    
    if low_recall_pt and high_recall_pt:
        report += f"""### Low Recall Point (Best Accuracy)
- **Confidence Threshold**: {low_recall_pt['confidence_threshold']}
- **Min Keypoints**: {low_recall_pt['min_keypoints']}
- **Recall**: {low_recall_pt['recall']:.4f}
- **Mean Error**: {low_recall_pt['mean_position_error']:.4f} m
- **Use case**: When accuracy is critical and missing some detections is acceptable

### High Recall Point (Best Coverage)
- **Confidence Threshold**: {high_recall_pt['confidence_threshold']}
- **Min Keypoints**: {high_recall_pt['min_keypoints']}
- **Recall**: {high_recall_pt['recall']:.4f}
- **Mean Error**: {high_recall_pt['mean_position_error']:.4f} m
- **Use case**: When detecting all pedestrians is critical

"""
    
    report += """---

## Generated Visualizations

The following visualizations were generated:

1. `all_architectures_loss_components.png` - Loss breakdown by component
2. `{arch}_loss_components.png` - Per-architecture loss details
3. `horizon_errors.png` - Error metrics across prediction horizons
4. `threshold_analysis.png` - Recall/Error tradeoff analysis
5. `threshold_analysis_per_metric.png` - Per-metric threshold analysis
6. `baseline_comparison_pareto.png` - Baseline vs corrected at pareto points
7. `per_joint_errors.png` - Per-joint error analysis
8. `direction_speed_errors.png` - Direction and speed error distributions

---

## Key Findings

"""
    
    # Add findings based on data
    if comparison_results:
        best_arch = min(comparison_results.keys(), 
                       key=lambda k: comparison_results[k].get('best_val_error', float('inf')))
        best_error = comparison_results[best_arch].get('best_val_error', 0)
        baseline = comparison_results[best_arch].get('baseline_error', 0)
        
        if baseline > 0:
            improvement = (baseline - best_error) / baseline * 100
            report += f"""1. **Best Architecture**: {best_arch.upper()} achieved the lowest validation error of {best_error:.4f}m
2. **Improvement over Baseline**: {improvement:.1f}% improvement compared to baseline ({baseline:.4f}m)
"""
    
    if low_recall_pt and high_recall_pt:
        report += f"""3. **Pareto Tradeoff**: 
   - Low recall config achieves {low_recall_pt['mean_position_error']:.4f}m error at {low_recall_pt['recall']:.1%} recall
   - High recall config achieves {high_recall_pt['mean_position_error']:.4f}m error at {high_recall_pt['recall']:.1%} recall
"""
    
    report += """
---

*Report generated by evaluate_and_visualize.py*
"""
    
    report_path = docs_dir / 'analysis_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  ✓ Saved: {report_path}")


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive PBPL evaluations')
    parser.add_argument('--input-dir', '-i', type=str, default='Outputs',
                       help='Input directory containing saved results')
    parser.add_argument('--output-dir', '-o', type=str, default='Evaluations',
                       help='Output directory for evaluation results')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PBPL COMPREHENSIVE EVALUATION")
    print("=" * 70)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load all data
    print("Loading data...")
    histories = {}
    for arch in ARCHITECTURES:
        history = load_history(input_dir, arch)
        if history:
            histories[arch] = history
            stage1 = history.get('stage1', history)
            loss_components = sum(1 for k in ['position_loss', 'per_keypoint_loss', 'bone_loss', 
                                               'correction_penalty', 'no_harm_penalty'] 
                                  if stage1.get(k))
            print(f"  ✓ Loaded {arch} history ({len(stage1.get('train_loss', []))} epochs, {loss_components} loss components)")
        else:
            print(f"  ⚠ No history found for {arch}")
    
    comparison_results = load_comparison_results(input_dir)
    if comparison_results:
        print(f"  ✓ Loaded comparison results ({len(comparison_results)} architectures)")
    
    threshold_results = load_threshold_analysis(input_dir)
    if threshold_results:
        print(f"  ✓ Loaded threshold analysis ({len(threshold_results)} configurations)")
    
    per_joint_data = load_per_joint_errors(input_dir)
    if per_joint_data:
        print("  ✓ Loaded per-joint error data")
    
    dir_speed_data = load_direction_speed_errors(input_dir)
    if dir_speed_data:
        print("  ✓ Loaded direction/speed error data")
    
    # Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    low_recall_pt = None
    high_recall_pt = None
    
    # 1. Loss component plots
    if histories:
        print("\n1. Loss component analysis...")
        for arch, history in histories.items():
            plot_loss_components(history, arch, output_dir, args.show)
        
        plot_all_architectures_loss_components(histories, output_dir, args.show)
    
    # 2. Horizon error plots
    if histories:
        print("\n2. Horizon error analysis...")
        plot_horizon_errors(histories, output_dir, args.show)
    
    # 3. Threshold analysis
    if threshold_results:
        print("\n3. Threshold analysis...")
        low_recall_pt, high_recall_pt = plot_threshold_analysis(threshold_results, output_dir, args.show)
        plot_per_loss_type_threshold(threshold_results, output_dir, args.show)
    
    # 4. Baseline comparison at pareto points
    if comparison_results:
        print("\n4. Baseline comparison at pareto points...")
        plot_baseline_comparison_pareto(comparison_results, low_recall_pt, high_recall_pt, 
                                        output_dir, args.show)
    
    # 5. Per-joint errors
    if per_joint_data:
        print("\n5. Per-joint error analysis...")
        plot_per_joint_errors(per_joint_data, output_dir, args.show)
    
    # 6. Direction/speed errors
    if dir_speed_data:
        print("\n6. Direction/speed error analysis...")
        plot_direction_speed_errors(dir_speed_data, output_dir, args.show)
    
    # Generate summary JSON
    print("\n" + "=" * 70)
    print("GENERATING SUMMARY FILES")
    print("=" * 70)
    
    print("\n7. Evaluation summary...")
    summary = generate_evaluation_summary(histories, comparison_results, threshold_results,
                                          low_recall_pt, high_recall_pt, output_dir)
    
    # Generate documentation
    print("\n8. Analysis report...")
    generate_analysis_report(summary, histories, comparison_results, 
                            low_recall_pt, high_recall_pt, output_dir)
    
    # Copy threshold analysis and other JSONs with updates
    print("\n9. Copying and updating JSON files...")
    
    # Save pareto points separately
    if low_recall_pt or high_recall_pt:
        pareto_path = output_dir / 'pareto_points.json'
        with open(pareto_path, 'w') as f:
            json.dump({
                'low_recall_point': low_recall_pt,
                'high_recall_point': high_recall_pt
            }, f, indent=2)
        print(f"  ✓ Saved: {pareto_path}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"Documentation saved to: {output_dir / 'docs'}")


if __name__ == '__main__':
    main()
