#!/usr/bin/env python3
"""
Evaluate trained GCN models with detailed metrics.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

sys.path.insert(0, str(Path(__file__).parent))

from models.stgcn import EnhancedSTGCN
from models.ctrgcn import EnhancedCTRGCN
from models.ctrgcn_motion import EnhancedCTRGCN_Motion
from models.tegcn import TE_GCN
try:
    from models.sht import SHT_Hyperformer
    HAS_SHT = True
except ImportError:
    HAS_SHT = False

# -----------------------------
# Dataset class
# -----------------------------
class GCNPedestrianDataset(Dataset):
    def __init__(self, data_path, labels_path):
        self.data = np.load(data_path)
        self.labels = np.load(labels_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), self.labels[idx]

# -----------------------------
# Evaluation function
# -----------------------------
def evaluate_model(model, dataloader, device, class_names=None):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, labels in tqdm(dataloader, desc="Evaluating"):
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(np.unique(all_labels)))]
    
    report = classification_report(all_labels, all_preds, 
                                   target_names=class_names, 
                                   output_dict=True)
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'confusion_matrix': cm,
        'classification_report': report,
        'per_class_accuracy': per_class_acc,
        'class_names': class_names
    }

# -----------------------------
# Print results
# -----------------------------
def print_results(results):
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    print(f"\nOverall Accuracy: {results['accuracy']*100:.2f}%")
    
    print("\nPer-Class Accuracy:")
    for i, (name, acc) in enumerate(zip(results['class_names'], results['per_class_accuracy'])):
        print(f"  {name}: {acc*100:.2f}%")
    
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    print("\nClassification Report:")
    print(classification_report(results['labels'], results['predictions'],
                               target_names=results['class_names']))
    
    # Class distribution
    unique, counts = np.unique(results['labels'], return_counts=True)
    print("\nClass Distribution in Dataset:")
    for cls, count in zip(unique, counts):
        print(f"  {results['class_names'][cls]}: {count} samples ({count/len(results['labels'])*100:.1f}%)")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate GCN models")
    parser.add_argument("--data_dir", type=str, default="./gcn_per_pedestrian",
                        help="Directory containing data.npy and labels.npy")
    model_choices = ["stgcn", "ctrgcn", "ctrgcn_motion", "tegcn"]
    if HAS_SHT:
        model_choices.append("sht")
    parser.add_argument("--model", type=str, required=True,
                        choices=model_choices,
                        help="Model architecture")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of behavior classes (auto-detected from data if not set)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("--class_names", type=str, nargs='+', default=None,
                        help="Class names (e.g., 'Walking' 'Running' 'Waving' 'Texting')")
    args = parser.parse_args()

    # Default class names: expanded set for risk analysis (matches BehaviorState)
    DEFAULT_CLASS_NAMES = [
        'Walking', 'Running', 'Crossing', 'Waiting_To_Cross', 'Idle',
        'Waving', 'Waving_Walking', 'Texting', 'Calling', 'Talking'
    ]
    if args.class_names is None:
        args.class_names = DEFAULT_CLASS_NAMES
    
    # Device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data_dir = Path(args.data_dir)
    data_path = data_dir / "data.npy"
    labels_path = data_dir / "labels.npy"
    
    print(f"Loading data from {data_dir}...")
    dataset = GCNPedestrianDataset(data_path, labels_path)
    print(f"Total samples: {len(dataset)}")

    # Trim class names to num_classes (or auto-detect from data)
    all_labels_eval = [dataset[i][1] for i in range(len(dataset))]
    num_classes_from_data = int(np.max(all_labels_eval)) + 1
    num_classes = args.num_classes if args.num_classes is not None else num_classes_from_data
    args.class_names = args.class_names[:num_classes] if len(args.class_names) > num_classes else args.class_names

    # Data loader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=2, pin_memory=True)
    
    # Get data shape
    sample_data, _ = dataset[0]
    num_joints = sample_data.shape[-1]
    in_channels = sample_data.shape[0]
    
    # Create model
    print(f"\nCreating {args.model.upper()} model...")
    if args.model == "stgcn":
        model = EnhancedSTGCN(
            in_channels=in_channels,
            num_joints=num_joints,
            num_classes=num_classes
        ).to(device)
    elif args.model == "ctrgcn":
        model = EnhancedCTRGCN(
            in_channels=in_channels,
            num_joints=num_joints,
            num_classes=num_classes
        ).to(device)
    elif args.model == "ctrgcn_motion":
        model = EnhancedCTRGCN_Motion(
            in_channels=in_channels,
            num_joints=num_joints,
            num_classes=num_classes
        ).to(device)
    elif args.model == "tegcn":
        model = TE_GCN(
            in_channels=in_channels,
            num_joints=num_joints,
            num_classes=num_classes
        ).to(device)
    elif args.model == "sht":
        if not HAS_SHT:
            raise ImportError("SHT/Hyperformer not available. Please install Hyperformer dependencies.")
        model = SHT_Hyperformer(
            in_channels=in_channels,
            num_joints=num_joints,
            num_classes=num_classes
        ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_acc' in checkpoint:
        print(f"Checkpoint validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Evaluate
    class_names = args.class_names if args.class_names else None
    results = evaluate_model(model, dataloader, device, class_names)
    
    # Print results
    print_results(results)
    
    # Save results
    output_path = Path(args.checkpoint).parent / 'evaluation_results.npz'
    np.savez(output_path,
             predictions=results['predictions'],
             labels=results['labels'],
             probabilities=results['probabilities'],
             confusion_matrix=results['confusion_matrix'],
             accuracy=results['accuracy'],
             per_class_accuracy=results['per_class_accuracy'])
    print(f"\nResults saved to: {output_path}")

if __name__ == '__main__':
    main()
