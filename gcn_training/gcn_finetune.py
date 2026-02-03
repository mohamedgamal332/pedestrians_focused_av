#!/usr/bin/env python3
"""
Fine-tune pre-trained GCN models on augmented dataset.
Uses transfer learning: loads a trained model and fine-tunes on augmented data.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import numpy as np
import argparse
import csv
from pathlib import Path
from tqdm import tqdm
import sys

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
    print("Warning: SHT/Hyperformer not available")

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
# Training function
# -----------------------------
def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for data, labels in pbar:
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100 * correct / total

# -----------------------------
# Validation function
# -----------------------------
def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
        for data, labels in pbar:
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
    
    return total_loss / len(dataloader), 100 * correct / total

# -----------------------------
# Main fine-tuning function
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune GCN models on augmented data")
    parser.add_argument("--pretrained_path", type=str, required=True,
                        help="Path to pretrained model checkpoint")
    parser.add_argument("--original_data_dir", type=str, default="./gcn_per_pedestrian",
                        help="Directory with original data")
    parser.add_argument("--augmented_data_dir", type=str, default="./gcn_per_pedestrian_augmented",
                        help="Directory with augmented data")
    parser.add_argument("--model", type=str, required=True,
                        choices=["stgcn", "ctrgcn", "ctrgcn_motion", "tegcn", "sht"],
                        help="Model architecture")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate (lower for fine-tuning)")
    parser.add_argument("--freeze_backbone", action='store_true',
                        help="Freeze backbone, only train classifier")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("--output_dir", type=str, default="./work_dirs/gcn_finetuning",
                        help="Output directory")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of classes (auto-detected if not specified)")
    args = parser.parse_args()
    
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    print("\nLoading datasets...")
    original_dataset = GCNPedestrianDataset(
        Path(args.original_data_dir) / "data.npy",
        Path(args.original_data_dir) / "labels.npy"
    )
    
    augmented_dataset = GCNPedestrianDataset(
        Path(args.augmented_data_dir) / "data.npy",
        Path(args.augmented_data_dir) / "labels.npy"
    )
    
    print(f"Original samples: {len(original_dataset)}")
    print(f"Augmented samples: {len(augmented_dataset)}")
    
    # Combine datasets
    combined_dataset = ConcatDataset([original_dataset, augmented_dataset])
    print(f"Combined samples: {len(combined_dataset)}")
    
    # Split
    train_size = int(0.8 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=2, pin_memory=True)
    
    # Get data shape and number of classes
    sample_data, _ = original_dataset[0]
    num_joints = sample_data.shape[-1]
    in_channels = sample_data.shape[0]
    
    # Detect number of classes from labels
    all_labels_temp = []
    for dataset in [original_dataset, augmented_dataset]:
        all_labels_temp.extend([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(all_labels_temp))
    print(f"\nDetected {num_classes} classes from data")
    
    # Create model
    print(f"\nCreating {args.model.upper()} model...")
    print(f"  Input channels: {in_channels}, Joints: {num_joints}, Classes: {num_classes}")
    
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
            num_classes=num_classes,
            pretrained_spatial=None,
            pretrained_motion=None
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
            num_classes=num_classes,
            pretrained_path=None  # Will load from checkpoint
        ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load pretrained weights
    print(f"\nLoading pretrained weights from {args.pretrained_path}...")
    if not Path(args.pretrained_path).exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained_path}")
    
    checkpoint = torch.load(args.pretrained_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Try to load with strict=False to handle class number mismatches
    try:
        model.load_state_dict(state_dict, strict=False)
        print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'val_acc' in checkpoint:
            print(f"  Original validation accuracy: {checkpoint['val_acc']:.2f}%")
    except Exception as e:
        print(f"⚠️  Warning: Could not load all weights: {e}")
        print("  Attempting partial load...")
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"  Loaded {len(pretrained_dict)}/{len(state_dict)} layers")
    
    # Freeze backbone if requested
    if args.freeze_backbone:
        print("Freezing backbone layers...")
        for name, param in model.named_parameters():
            if 'fc' not in name:  # Keep classifier trainable
                param.requires_grad = False
        print("Only classifier will be trained")
    
    # Calculate class weights
    all_labels = []
    for dataset in [original_dataset, augmented_dataset]:
        all_labels.extend([dataset[i][1] for i in range(len(dataset))])
    class_counts = np.bincount(all_labels)
    total_samples = len(all_labels)
    class_weights = total_samples / (len(class_counts) * class_counts + 1e-6)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"\nClass distribution: {dict(zip(range(len(class_counts)), class_counts))}")
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Use different learning rates for backbone and classifier
    if args.freeze_backbone:
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=0.0001
        )
    else:
        # Separate learning rates for backbone and classifier
        backbone_params = []
        classifier_params = []
        for name, param in model.named_parameters():
            if 'fc' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': args.lr * 0.1},  # Lower LR for backbone
            {'params': classifier_params, 'lr': args.lr}        # Higher LR for classifier
        ], weight_decay=0.0001)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Output directory
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training history (for visualization: epochs, metrics, lr)
    history = {
        'epochs': [],
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"\nStarting fine-tuning for {args.epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        history['epochs'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history,
                'pretrained_from': args.pretrained_path
            }, output_dir / 'best_finetuned_model.pth')
            print(f"  ✓ Saved best fine-tuned model (val_acc: {val_acc:.2f}%)")
        
        print("-" * 80)
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'history': history,
        'pretrained_from': args.pretrained_path
    }, output_dir / 'final_finetuned_model.pth')
    
    np.save(output_dir / 'finetuning_history.npy', history)
    
    # Save as CSV for easy visualization
    csv_path = output_dir / 'finetuning_history.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])
        for i in range(len(history['epochs'])):
            w.writerow([
                history['epochs'][i],
                history['train_loss'][i],
                history['train_acc'][i],
                history['val_loss'][i],
                history['val_acc'][i],
                history['lr'][i]
            ])
    print(f"Fine-tuning history saved to {csv_path} (for visualization)")
    
    print(f"\nFine-tuning completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    print(f"Models saved to: {output_dir}")

if __name__ == '__main__':
    main()
