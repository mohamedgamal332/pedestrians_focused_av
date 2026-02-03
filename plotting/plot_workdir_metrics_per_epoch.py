#!/usr/bin/env python3
"""
Plot metrics per epoch for each model in work_dirs (academic style).
Discovers training_history.csv, finetuning_history.csv, and coco_validation_metrics.csv
under work_dirs and generates publication-ready figures (serif fonts, high DPI).
Reference: plot_training_log.py
"""

import csv
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless; no display required
import matplotlib.pyplot as plt

# Academic/thesis-style: serif fonts, high DPI, tight layout
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})


# Filenames we look for under work_dirs
HISTORY_CSV_NAMES = frozenset({
    "training_history.csv",
    "finetuning_history.csv",
    "coco_validation_metrics.csv",
})


def find_history_csvs(work_dirs: Path):
    """Yield (csv_path, model_label) for every discovered history CSV."""
    work_dirs = Path(work_dirs)
    if not work_dirs.is_dir():
        return
    for csv_path in work_dirs.rglob("*.csv"):
        if csv_path.name not in HISTORY_CSV_NAMES:
            continue
        # Model label: e.g. gcn_training/ctrgcn or rtmpose_m_carla/20260125_144452
        try:
            rel = csv_path.relative_to(work_dirs)
            model_label = str(rel.parent).replace("\\", "/")
        except ValueError:
            model_label = str(csv_path.parent)
        yield csv_path, model_label


def load_csv(csv_path: Path):
    """Load CSV; return list of dicts (keys = header)."""
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def numeric_series(rows, key):
    """Extract numeric series; normalize epoch/key names (Epoch vs epoch)."""
    epoch_key = "Epoch" if "Epoch" in (rows[0] if rows else {}) else "epoch"
    vals = []
    for r in rows:
        raw = r.get(key) or r.get(key.replace("_", " ")) or r.get(epoch_key if key == "epoch" else key)
        if raw is None:
            continue
        try:
            vals.append(float(raw))
        except ValueError:
            vals.append(np.nan)
    return np.array(vals) if vals else np.array([])


def get_epochs(rows):
    """Epoch column (int)."""
    if not rows:
        return np.array([])
    r0 = rows[0]
    epoch_key = "Epoch" if "Epoch" in r0 else "epoch"
    epochs = []
    for r in rows:
        try:
            epochs.append(int(float(r.get(epoch_key, np.nan))))
        except (TypeError, ValueError):
            epochs.append(len(epochs) + 1)
    return np.array(epochs)


def plot_gcn_style(csv_path: Path, rows: list, out_dir: Path, model_label: str):
    """Plot train/val loss and accuracy vs epoch (GCN training/finetuning history)."""
    epochs = get_epochs(rows)
    if len(epochs) == 0:
        return

    train_loss = numeric_series(rows, "train_loss")
    train_acc = numeric_series(rows, "train_acc")
    val_loss = numeric_series(rows, "val_loss")
    val_acc = numeric_series(rows, "val_acc")

    # Normalize accuracy to [0, 1] if stored as 0–100
    if len(train_acc) and np.nanmax(train_acc) > 1.5:
        train_acc = train_acc / 100.0
    if len(val_acc) and np.nanmax(val_acc) > 1.5:
        val_acc = val_acc / 100.0

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "metrics_per_epoch"

    # —— Loss vs epoch ——
    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    if len(train_loss) == len(epochs):
        ax1.plot(epochs, train_loss, "o-", color="C0", linewidth=1.5, markersize=4, label="Train loss")
    if len(val_loss) == len(epochs):
        ax1.plot(epochs, val_loss, "s-", color="C1", linewidth=1.5, markersize=4, label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss vs epoch — {model_label}")
    ax1.legend(loc="upper right")
    fig1.tight_layout()
    fig1.savefig(out_dir / f"{base.name}_loss.png")
    fig1.savefig(out_dir / f"{base.name}_loss.pdf")
    plt.close(fig1)

    # —— Accuracy vs epoch ——
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    if len(train_acc) == len(epochs):
        ax2.plot(epochs, train_acc, "o-", color="C0", linewidth=1.5, markersize=4, label="Train acc")
    if len(val_acc) == len(epochs):
        ax2.plot(epochs, val_acc, "s-", color="C1", linewidth=1.5, markersize=4, label="Val acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Accuracy vs epoch — {model_label}")
    ax2.legend(loc="lower right")
    ax2.set_ylim(0, 1.02)
    fig2.tight_layout()
    fig2.savefig(out_dir / f"{base.name}_accuracy.png")
    fig2.savefig(out_dir / f"{base.name}_accuracy.pdf")
    plt.close(fig2)

    # —— Combined 2-panel ——
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    if len(train_loss) == len(epochs):
        ax3a.plot(epochs, train_loss, "o-", color="C0", linewidth=1.2, markersize=3, label="Train")
    if len(val_loss) == len(epochs):
        ax3a.plot(epochs, val_loss, "s-", color="C1", linewidth=1.2, markersize=3, label="Val")
    ax3a.set_ylabel("Loss")
    ax3a.set_title(f"Loss and accuracy vs epoch — {model_label}")
    ax3a.legend(loc="upper right")

    if len(train_acc) == len(epochs):
        ax3b.plot(epochs, train_acc, "o-", color="C0", linewidth=1.2, markersize=3, label="Train")
    if len(val_acc) == len(epochs):
        ax3b.plot(epochs, val_acc, "s-", color="C1", linewidth=1.2, markersize=3, label="Val")
    ax3b.set_xlabel("Epoch")
    ax3b.set_ylabel("Accuracy")
    ax3b.legend(loc="lower right")
    ax3b.set_ylim(0, 1.02)
    fig3.tight_layout()
    fig3.savefig(out_dir / f"{base.name}_combined.png")
    fig3.savefig(out_dir / f"{base.name}_combined.pdf")
    plt.close(fig3)

    print(f"  Saved GCN-style plots in {out_dir}")


def plot_coco_style(csv_path: Path, rows: list, out_dir: Path, model_label: str):
    """Plot COCO AP/AR metrics vs epoch (coco_validation_metrics.csv)."""
    epochs = get_epochs(rows)
    if len(epochs) == 0:
        return

    r0 = rows[0]
    # Prefer columns that exist
    metrics = []
    for col in ["AP", "AP@0.5", "AP@0.75", "AR", "AR@0.5", "AR@0.75"]:
        if col in r0:
            metrics.append((col, numeric_series(rows, col)))

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "metrics_per_epoch"

    fig, ax = plt.subplots(figsize=(6, 3.5))
    markers = ["o", "s", "^", "d", "v", "p"]
    for i, (label, series) in enumerate(metrics):
        if len(series) == len(epochs):
            ax.plot(epochs, series, f"{markers[i % len(markers)]}-", linewidth=1.2, markersize=4, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title(f"Validation COCO metrics vs epoch — {model_label}")
    ax.legend(loc="best", ncol=2)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"{base.name}_coco.png")
    fig.savefig(out_dir / f"{base.name}_coco.pdf")
    plt.close(fig)
    print(f"  Saved COCO-style plot in {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot metrics per epoch for each model in work_dirs (academic style)."
    )
    parser.add_argument(
        "work_dirs",
        nargs="?",
        default=Path(__file__).resolve().parent / "work_dirs",
        type=Path,
        help="Root directory to search for history CSVs (default: work_dirs)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Optional: save all plots here instead of next to each CSV",
    )
    args = parser.parse_args()

    work_dirs = Path(args.work_dirs)
    if not work_dirs.is_dir():
        raise SystemExit(f"Directory not found: {work_dirs}")

    collected = list(find_history_csvs(work_dirs))
    if not collected:
        print(f"No history CSVs found under {work_dirs} (looking for {HISTORY_CSV_NAMES}).")
        return

    print(f"Found {len(collected)} history file(s).")
    for csv_path, model_label in collected:
        print(f"  Processing: {model_label} — {csv_path.name}")
        rows = load_csv(csv_path)
        if not rows:
            print(f"    Skipped (empty).")
            continue

        out_dir = Path(args.output_dir) if args.output_dir else csv_path.parent

        if csv_path.name == "coco_validation_metrics.csv":
            plot_coco_style(csv_path, rows, out_dir, model_label)
        else:
            plot_gcn_style(csv_path, rows, out_dir, model_label)

    print("Done.")


if __name__ == "__main__":
    main()
