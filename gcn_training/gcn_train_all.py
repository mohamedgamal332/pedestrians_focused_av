#!/usr/bin/env python3
"""
Train all GCN models and compare results.
"""
import subprocess
import sys
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train all GCN models")
    parser.add_argument("--data_dir", type=str, default="./gcn_per_pedestrian")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--pretrained", action='store_true',
                        help="Use pretrained weights for ST-GCN")
    args = parser.parse_args()
    
    models = ['stgcn', 'ctrgcn', 'tegcn']
    pretrained_paths = {
        'stgcn': 'pretrained_weights/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth' if args.pretrained else None
    }
    
    print("=" * 80)
    print("Training All GCN Models")
    print("=" * 80)
    
    for model in models:
        print(f"\n{'='*80}")
        print(f"Training {model.upper()}")
        print(f"{'='*80}\n")
        
        cmd = [
            sys.executable, 'gcn_train.py',
            '--model', model,
            '--data_dir', args.data_dir,
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--lr', str(args.lr)
        ]
        
        if model in pretrained_paths and pretrained_paths[model]:
            cmd.extend(['--pretrained_path', pretrained_paths[model]])
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error training {model}: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print("\nTo evaluate models, run:")
    print("  python gcn_eval.py --model stgcn --checkpoint work_dirs/gcn_training/stgcn/best_model.pth")
    print("  python gcn_eval.py --model ctrgcn --checkpoint work_dirs/gcn_training/ctrgcn/best_model.pth")
    print("  python gcn_eval.py --model tegcn --checkpoint work_dirs/gcn_training/tegcn/best_model.pth")

if __name__ == '__main__':
    main()
