#!/usr/bin/env python3
"""
Plot training / fine-tuning history for visualization.
Loads history from .npy or .csv and plots loss, accuracy, and learning rate.
"""
import argparse
import numpy as np
from pathlib import Path

def load_history(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")
    if path.suffix == '.npy':
        return np.load(path, allow_pickle=True).item()
    if path.suffix == '.csv':
        import csv
        with open(path) as f:
            r = csv.DictReader(f)
            rows = list(r)
        if not rows:
            raise ValueError("CSV is empty")
        return {
            'epochs': [int(row['epoch']) for row in rows],
            'train_loss': [float(row['train_loss']) for row in rows],
            'train_acc': [float(row['train_acc']) for row in rows],
            'val_loss': [float(row['val_loss']) for row in rows],
            'val_acc': [float(row['val_acc']) for row in rows],
            'lr': [float(row['lr']) for row in rows],
        }
    raise ValueError("Use .npy or .csv file")

def main():
    parser = argparse.ArgumentParser(description="Plot GCN training/fine-tuning history")
    parser.add_argument("history_path", type=str,
                        help="Path to history.npy, training_history.csv, or finetuning_history.csv")
    parser.add_argument("--output", type=str, default=None,
                        help="Save figure to this path (default: show only)")
    parser.add_argument("--title", type=str, default=None,
                        help="Plot title")
    args = parser.parse_args()

    history = load_history(args.history_path)
    epochs = history['epochs']

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib to plot: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Loss
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], label='Train loss', color='C0')
    ax.plot(epochs, history['val_loss'], label='Val loss', color='C1')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, history['train_acc'], label='Train acc (%)', color='C0')
    ax.plot(epochs, history['val_acc'], label='Val acc (%)', color='C1')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[2]
    if 'lr' in history and history['lr']:
        ax.plot(epochs, history['lr'], color='C2')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning rate')
    ax.set_title('Learning rate')
    ax.grid(True, alpha=0.3)

    if args.title:
        fig.suptitle(args.title, fontsize=12)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved figure to {args.output}")
    else:
        plt.show()

if __name__ == '__main__':
    main()
