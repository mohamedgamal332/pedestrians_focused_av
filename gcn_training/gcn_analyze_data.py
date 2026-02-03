#!/usr/bin/env python3
"""
Analyze GCN dataset to diagnose training issues.
"""
import numpy as np
import argparse
from pathlib import Path
from collections import Counter

def main():
    parser = argparse.ArgumentParser(description="Analyze GCN dataset")
    parser.add_argument("--data_dir", type=str, default="./gcn_per_pedestrian",
                        help="Directory containing data.npy and labels.npy")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    data_path = data_dir / "data.npy"
    labels_path = data_dir / "labels.npy"
    
    print("=" * 80)
    print("GCN Dataset Analysis")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data from {data_dir}...")
    data = np.load(data_path)
    labels = np.load(labels_path)
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(data):,}")
    print(f"  Data shape: {data.shape} (N, C, T, V)")
    print(f"  Labels shape: {labels.shape}")
    
    # Data statistics
    print(f"\nData Statistics:")
    print(f"  Channels (C): {data.shape[1]}")
    print(f"  Temporal frames (T): {data.shape[2]}")
    print(f"  Joints (V): {data.shape[3]}")
    print(f"  Data dtype: {data.dtype}")
    print(f"  Data range: [{data.min():.4f}, {data.max():.4f}]")
    print(f"  Data mean: {data.mean():.4f}")
    print(f"  Data std: {data.std():.4f}")
    
    # Check for NaN or Inf
    nan_count = np.isnan(data).sum()
    inf_count = np.isinf(data).sum()
    if nan_count > 0:
        print(f"  ⚠️  WARNING: {nan_count} NaN values found!")
    if inf_count > 0:
        print(f"  ⚠️  WARNING: {inf_count} Inf values found!")
    
    # Label statistics
    print(f"\nLabel Statistics:")
    print(f"  Number of classes: {len(np.unique(labels))}")
    print(f"  Label range: [{labels.min()}, {labels.max()}]")
    
    # Class distribution
    class_counts = Counter(labels)
    total = len(labels)
    class_names = ['Walking', 'Running', 'Crossing', 'Waiting_To_Cross', 'Idle']
    
    print(f"\nClass Distribution:")
    print(f"  {'Class':<20} {'Count':<10} {'Percentage':<10} {'Status'}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    
    for i in range(len(np.unique(labels))):
        count = class_counts.get(i, 0)
        pct = 100 * count / total if total > 0 else 0
        name = class_names[i] if i < len(class_names) else f'Class_{i}'
        status = "⚠️  Imbalanced" if pct < 5 or pct > 50 else "✓"
        print(f"  {name:<20} {count:<10} {pct:>9.2f}%  {status}")
    
    # Check for class imbalance
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\nClass Imbalance Analysis:")
    print(f"  Max class count: {max_count}")
    print(f"  Min class count: {min_count}")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")
    if imbalance_ratio > 10:
        print(f"  ⚠️  WARNING: Severe class imbalance detected!")
        print(f"  Consider using weighted loss or data augmentation")
    elif imbalance_ratio > 3:
        print(f"  ⚠️  Moderate class imbalance - consider weighted loss")
    else:
        print(f"  ✓ Classes are reasonably balanced")
    
    # Sample quality check
    print(f"\nSample Quality Check:")
    zero_samples = 0
    constant_samples = 0
    for i in range(min(1000, len(data))):  # Check first 1000 samples
        sample = data[i]
        if np.all(sample == 0):
            zero_samples += 1
        elif np.all(sample == sample.flat[0]):  # All values the same
            constant_samples += 1
    
    if zero_samples > 0:
        print(f"  ⚠️  WARNING: {zero_samples} zero samples found in first 1000")
    if constant_samples > 0:
        print(f"  ⚠️  WARNING: {constant_samples} constant samples found in first 1000")
    if zero_samples == 0 and constant_samples == 0:
        print(f"  ✓ No obvious quality issues in samples")
    
    # Recommendations
    print(f"\n" + "=" * 80)
    print("Recommendations:")
    print("=" * 80)
    
    recommendations = []
    
    if data.std() < 0.01:
        recommendations.append("  ⚠️  Data has very low variance - consider normalization")
    
    if imbalance_ratio > 3:
        recommendations.append("  ⚠️  Use weighted CrossEntropyLoss to handle class imbalance")
        recommendations.append("     Example: weights = torch.tensor([...]) / class_counts")
    
    if nan_count > 0 or inf_count > 0:
        recommendations.append("  ⚠️  Clean data to remove NaN/Inf values")
    
    if len(data) < 1000:
        recommendations.append("  ⚠️  Small dataset - consider data augmentation")
    
    if len(recommendations) == 0:
        print("  ✓ Dataset looks good for training!")
    else:
        for rec in recommendations:
            print(rec)
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
