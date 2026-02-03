#!/usr/bin/env python3
"""
Train GCN models on pedestrian behavior recognition.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import argparse
import csv
from pathlib import Path
from tqdm import tqdm
import sys
from datetime import datetime

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
        self.data = np.load(data_path)  # [N, C, T, V]
        self.labels = np.load(labels_path)  # [N]
        
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
        
        # Forward
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# -----------------------------
# Validation function
# -----------------------------
def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
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
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy, all_preds, all_labels

# -----------------------------
# Main training function
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train GCN models")
    parser.add_argument("--data_dir", type=str, default="./gcn_per_pedestrian",
                        help="Directory containing data.npy and labels.npy")
    model_choices = ["stgcn", "ctrgcn", "ctrgcn_motion", "tegcn"]
    if HAS_SHT:
        model_choices.append("sht")
    parser.add_argument("--model", type=str, default="stgcn", 
                        choices=model_choices,
                        help="Model architecture")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Path to pretrained weights (optional; for ctrgcn_motion use --pretrained_spatial/--pretrained_motion)")
    parser.add_argument("--pretrained_spatial", type=str, default=None,
                        help="Path to pretrained spatial branch (ctrgcn_motion only)")
    parser.add_argument("--pretrained_motion", type=str, default=None,
                        help="Path to pretrained motion branch (ctrgcn_motion only)")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of behavior classes (auto-detected from data if not set)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                        help="Weight decay")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="Train/val split ratio")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("--output_dir", type=str, default="./work_dirs/gcn_training",
                        help="Output directory for checkpoints")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="Save checkpoint every N epochs")
    args = parser.parse_args()
    
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

    # Auto-detect num_classes from data if not set (supports 5 or 10 class datasets)
    all_labels = [dataset[i][1] for i in range(len(dataset))]
    num_classes_from_data = int(np.max(all_labels)) + 1
    num_classes = args.num_classes if args.num_classes is not None else num_classes_from_data
    if args.num_classes is not None and args.num_classes != num_classes_from_data:
        print(f"Warning: --num_classes {args.num_classes} != classes in data ({num_classes_from_data}); using {args.num_classes}")
    else:
        print(f"Number of classes (from data): {num_classes}")
    
    # Split dataset
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=2, pin_memory=True)
    
    # Get data shape
    sample_data, _ = dataset[0]
    num_joints = sample_data.shape[-1]  # Should be 17 for COCO
    in_channels = sample_data.shape[0]  # Should be 2 or 3
    
    print(f"Input shape: {sample_data.shape} (C={in_channels}, T={sample_data.shape[1]}, V={num_joints})")
    
    # Create model
    print(f"\nCreating {args.model.upper()} model...")
    if args.model == "stgcn":
        model = EnhancedSTGCN(
            in_channels=in_channels,
            num_joints=num_joints,
            num_classes=num_classes,
            pretrained_path=args.pretrained_path
        ).to(device)
    elif args.model == "ctrgcn":
        model = EnhancedCTRGCN(
            in_channels=in_channels,
            num_joints=num_joints,
            num_classes=num_classes,
            pretrained_path=args.pretrained_path
        ).to(device)
    elif args.model == "ctrgcn_motion":
        # Use same path for both branches if only pretrained_path given
        pretrained_spatial = args.pretrained_spatial or args.pretrained_path
        pretrained_motion = args.pretrained_motion or args.pretrained_path
        model = EnhancedCTRGCN_Motion(
            in_channels=in_channels,
            num_joints=num_joints,
            num_classes=num_classes,
            pretrained_spatial=pretrained_spatial,
            pretrained_motion=pretrained_motion
        ).to(device)
    elif args.model == "tegcn":
        model = TE_GCN(
            in_channels=in_channels,
            num_joints=num_joints,
            num_classes=num_classes,
            pretrained_path=args.pretrained_path
        ).to(device)
    elif args.model == "sht":
        if not HAS_SHT:
            raise ImportError("SHT/Hyperformer not available. Please install Hyperformer dependencies.")
        # SHT can use Hyperformer pretrained weights
        hyperformer_pretrained = args.pretrained_path
        model = SHT_Hyperformer(
            in_channels=in_channels,
            num_joints=num_joints,
            num_classes=num_classes,
            pretrained_path=hyperformer_pretrained
        ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Calculate class weights for imbalanced datasets
    all_labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = np.bincount(all_labels)
    total_samples = len(all_labels)
    class_weights = total_samples / (len(class_counts) * class_counts + 1e-6)  # Add small epsilon to avoid division by zero
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"\nClass distribution: {dict(zip(range(len(class_counts)), class_counts))}")
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Create output directory
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
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device, epoch)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history for visualization
        history['epochs'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, output_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
        
        # Periodic checkpoint
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, output_dir / f'checkpoint_epoch_{epoch}.pth')
        
        print("-" * 80)
    
    # Save final model and history
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'history': history
    }, output_dir / 'final_model.pth')
    
    # Save training history (numpy for programmatic use)
    np.save(output_dir / 'history.npy', history)
    
    # Save training history as CSV for easy visualization (Excel, pandas, matplotlib)
    csv_path = output_dir / 'training_history.csv'
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
    print(f"Training history saved to {csv_path} (for visualization)")
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    print(f"Models saved to: {output_dir}")

if __name__ == '__main__':
    main()
