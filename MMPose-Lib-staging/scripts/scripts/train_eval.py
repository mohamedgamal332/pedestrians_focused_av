"""Train, test and evaluate ST-GCN / CTR-GCN models on pose-labeled data.

Usage example:
  python scripts/train_eval.py --data_dir pose_label_sample --out results.txt --models stgcn ctrgcn ctrgcn_motion
"""
import argparse
import random
from pathlib import Path
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np

from data.pose_labeled_dataset import PoseLabeledDataset


def build_model(model_type, in_channels, num_joints, num_classes):
    if model_type == 'stgcn':
        from models.stgcn import STGCN
        return STGCN(in_channels=in_channels, num_joints=num_joints, num_classes=num_classes)
    elif model_type == 'ctrgcn':
        from models.ctrgcn import CTRGCN
        return CTRGCN(in_channels=in_channels, num_joints=num_joints, num_classes=num_classes)
    elif model_type == 'ctrgcn_motion':
        from models.ctrgcn_motion import CTRGCN_Motion
        return CTRGCN_Motion(in_channels=in_channels, num_joints=num_joints, num_classes=num_classes)
    else:
        raise ValueError('Unknown model')


def evaluate(model, dl, device, num_classes):
    model.eval()
    correct = 0
    total = 0
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            p = logits.argmax(dim=1)
            preds.extend(p.cpu().tolist())
            trues.extend(y.cpu().tolist())
            correct += (p == y).sum().item()
            total += y.size(0)

    accuracy = correct / max(1, total)

    # per-class precision/recall
    per_class = {}
    for c in range(num_classes):
        tp = sum(1 for i in range(len(preds)) if preds[i] == c and trues[i] == c)
        fp = sum(1 for i in range(len(preds)) if preds[i] == c and trues[i] != c)
        fn = sum(1 for i in range(len(preds)) if preds[i] != c and trues[i] == c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {'precision': prec, 'recall': rec, 'f1': f1, 'support': tp + fn}

    return {'accuracy': accuracy, 'per_class': per_class, 'total': total}


def train_one_epoch(model, dl, opt, crit, device):
    model.train()
    total_loss = 0.0
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / max(1, len(dl))


def run_experiment(data_dir, out_file, models, epochs=3, batch_size=8, temporal_length=30, stride=15, in_channels=2, device='cpu'):
    ds = PoseLabeledDataset(data_dir, temporal_length=temporal_length, stride=stride, in_channels=in_channels)
    num_classes = len(ds.classes)
    num_joints = get_num_joints_from_ds(ds)

    # split by track to avoid train/test leakage from overlapping sliding windows
    # build mapping from (file, tid) -> sample indices and majority label
    from collections import defaultdict, Counter
    track_to_indices = defaultdict(list)
    track_to_labels = {}
    for idx, (fp, tid, start) in enumerate(ds.samples):
        track_to_indices[(fp, tid)].append(idx)
    for (fp, tid), idxs in track_to_indices.items():
        # majority label for this track
        labs = [ds[i][1] for i in idxs]
        lab = Counter(labs).most_common(1)[0][0]
        track_to_labels[(fp, tid)] = lab

    tracks = list(track_to_indices.keys())
    random.shuffle(tracks)
    split = int(0.8 * len(tracks))
    train_tracks = set(tracks[:split])
    test_tracks = set(tracks[split:])

    train_idx = []
    test_idx = []
    for t in train_tracks:
        train_idx.extend(track_to_indices[t])
    for t in test_tracks:
        test_idx.extend(track_to_indices[t])

    # report label distribution
    all_labels = [ds[i][1] for i in range(len(ds))]
    from collections import Counter
    label_counts = Counter(all_labels)
    print('Overall label counts:', {ds.classes[k]: v for k, v in label_counts.items()})

    # compute class weights inverse to frequency for the loss
    total = float(len(ds))
    class_weights = []
    for i in range(len(ds.classes)):
        freq = label_counts.get(i, 0) / total
        class_weights.append(1.0 / (freq + 1e-6))
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    train_dl = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(Subset(ds, test_idx), batch_size=batch_size, shuffle=False)

    results = {}
    for mtype in models:
        print(f'Running model {mtype}')
        model = build_model(mtype, in_channels, num_joints, num_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss(weight=class_weights.to(device))

        start = time.time()
        per_epoch = []
        for e in range(epochs):
            loss = train_one_epoch(model, train_dl, opt, crit, device)
            val_res = evaluate(model, test_dl, device, num_classes)
            per_epoch.append({'epoch': e, 'train_loss': loss, 'val_accuracy': val_res['accuracy']})
            print(f'Model {mtype} epoch {e} train_loss={loss:.4f} val_acc={val_res["accuracy"]:.4f}')

        eval_res = evaluate(model, test_dl, device, num_classes)
        elapsed = time.time() - start
        # compute simple majority baseline on test set
        test_labels = [ds[i][1] for i in test_idx]
        if len(test_labels) > 0:
            maj = Counter(test_labels).most_common(1)[0]
            maj_acc = sum(1 for l in test_labels if l == maj[0]) / len(test_labels)
        else:
            maj_acc = 0.0

        results[mtype] = {'eval': eval_res, 'time': elapsed, 'per_epoch': per_epoch, 'baseline_majority_acc': maj_acc, 'label_counts': label_counts}

    # write results to out_file
    with open(out_file, 'w') as f:
        f.write(f'Run at {time.ctime()}\n')
        f.write(f'Data dir: {data_dir}\n')
        f.write(f'Params: epochs={epochs}, batch_size={batch_size}, temporal_length={temporal_length}, in_channels={in_channels}\n\n')
        for mtype, r in results.items():
            f.write(f'Model: {mtype}\n')
            f.write(f'Elapsed: {r["time"]:.2f}s\n')
            f.write(f'Final Accuracy: {r["eval"]["accuracy"]:.4f} (N={r["eval"]["total"]})\n')
            f.write('Per-class:\n')
            for c, stats in r['eval']['per_class'].items():
                f.write(f'  class {c}: prec={stats["precision"]:.3f} rec={stats["recall"]:.3f} f1={stats["f1"]:.3f} support={stats["support"]}\n')
            f.write('\n')
            f.write('Per-epoch:\n')
            for pe in r.get('per_epoch', []):
                f.write(f"  epoch {pe['epoch']}: train_loss={pe['train_loss']:.4f} val_acc={pe['val_accuracy']:.4f}\n")
            f.write('\n')

    print('All experiments finished. Results written to', out_file)


def get_num_joints_from_ds(ds: PoseLabeledDataset):
    # attempt to infer V from first sample
    if len(ds) == 0:
        return 12
    x, y = ds[0]
    # x: C, T, V, 1
    return x.shape[2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='pose_label_sample')
    parser.add_argument('--out', default='results.txt')
    parser.add_argument('--models', nargs='+', default=['stgcn', 'ctrgcn', 'ctrgcn_motion'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--temporal_length', type=int, default=30)
    parser.add_argument('--stride', type=int, default=15)
    parser.add_argument('--in_channels', type=int, default=2)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    run_experiment(args.data_dir, args.out, args.models, epochs=args.epochs, batch_size=args.batch_size, temporal_length=args.temporal_length, stride=args.stride, in_channels=args.in_channels, device=args.device)
