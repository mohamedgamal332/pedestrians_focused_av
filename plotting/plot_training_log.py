#!/usr/bin/env python3
"""
Plot RTMPose fine-tuning training and accuracy from MMEngine log file.
Generates thesis-defence-ready figures (iterations, loss, accuracy, validation AP).
"""

import csv
import re
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Thesis-style: readable fonts, high DPI
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def parse_log(log_path: str):
    """Parse MMEngine training log. Returns train_records and val_records."""
    train_records = []  # list of dicts: iteration, epoch, step, loss, loss_kpt, acc_pose
    val_records = []    # list of dicts: epoch + all coco metrics

    # Epoch(train)   [8][200/861]  ... loss: 0.215537  loss_kpt: 0.215537  acc_pose: 0.800649
    train_pat = re.compile(
        r"Epoch\(train\)\s+\[(\d+)\]\[\s*(\d+)/(\d+)\].*?"
        r"loss:\s*([\d.]+)\s+loss_kpt:\s*([\d.]+)\s+acc_pose:\s*([\d.]+)"
    )
    # Full COCO line: AP, AP .5, AP .75, AP (M), AP (L), AR, AR .5, AR .75, AR (M), AR (L)
    val_pat_full = re.compile(
        r"Epoch\(val\)\s+\[(\d+)\]\[\d+/\d+\].*?"
        r"coco/AP:\s*([\d.]+)\s+coco/AP \.5:\s*([\d.]+)\s+coco/AP \.75:\s*([\d.]+)"
        r"(?:\s+coco/AP \(M\):\s*([\d.]+))?(?:\s+coco/AP \(L\):\s*([\d.]+))?"
        r"\s+coco/AR:\s*([\d.]+)\s+coco/AR \.5:\s*([\d.]+)\s+coco/AR \.75:\s*([\d.]+)"
        r"(?:\s+coco/AR \(M\):\s*([\d.]+))?(?:\s+coco/AR \(L\):\s*([\d.]+))?"
    )
    val_pat_short = re.compile(
        r"Epoch\(val\)\s+\[(\d+)\]\[\d+/\d+\].*?"
        r"coco/AP:\s*([\d.]+)\s+coco/AP \.5:\s*([\d.]+)\s+coco/AP \.75:\s*([\d.]+).*?"
        r"coco/AR:\s*([\d.]+)"
    )

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = train_pat.search(line)
            if m:
                epoch, step, total = int(m.group(1)), int(m.group(2)), int(m.group(3))
                iteration = (epoch - 1) * total + step
                train_records.append({
                    "iteration": iteration,
                    "epoch": epoch,
                    "step": step,
                    "loss": float(m.group(4)),
                    "loss_kpt": float(m.group(5)),
                    "acc_pose": float(m.group(6)),
                })
                continue
            m = val_pat_full.search(line)
            if m:
                # Groups: 1=epoch, 2=AP, 3=AP.5, 4=AP.75, 5=AP(M), 6=AP(L), 7=AR, 8=AR.5, 9=AR.75, 10=AR(M), 11=AR(L)
                r = {
                    "epoch": int(m.group(1)),
                    "coco_AP": float(m.group(2)),
                    "coco_AP_50": float(m.group(3)),
                    "coco_AP_75": float(m.group(4)),
                    "coco_AP_M": float(m.group(5)) if m.group(5) else None,
                    "coco_AP_L": float(m.group(6)) if m.group(6) else None,
                    "coco_AR": float(m.group(7)),
                    "coco_AR_50": float(m.group(8)),
                    "coco_AR_75": float(m.group(9)),
                    "coco_AR_M": float(m.group(10)) if m.group(10) else None,
                    "coco_AR_L": float(m.group(11)) if m.group(11) else None,
                }
                val_records.append(r)
                continue
            m = val_pat_short.search(line)
            if m:
                val_records.append({
                    "epoch": int(m.group(1)),
                    "coco_AP": float(m.group(2)),
                    "coco_AP_50": float(m.group(3)),
                    "coco_AP_75": float(m.group(4)),
                    "coco_AR": float(m.group(5)),
                })

    return train_records, val_records


def smooth(y, window=51):
    """Moving average for presentation (odd window)."""
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window) / window, mode="valid")


# Columns for COCO tables (display name -> record key)
COCO_TABLE_COLUMNS = [
    ("Epoch", "epoch"),
    ("AP", "coco_AP"),
    ("AP@0.5", "coco_AP_50"),
    ("AP@0.75", "coco_AP_75"),
    ("AP (M)", "coco_AP_M"),
    ("AP (L)", "coco_AP_L"),
    ("AR", "coco_AR"),
    ("AR@0.5", "coco_AR_50"),
    ("AR@0.75", "coco_AR_75"),
    ("AR (M)", "coco_AR_M"),
    ("AR (L)", "coco_AR_L"),
]


def save_coco_tables(val_records: list, out_dir: Path):
    """Write COCO validation metrics to CSV, Markdown, and a matplotlib table figure."""
    if not val_records:
        return
    # Use only columns that appear in the first record (no None for all)
    cols = [(label, key) for label, key in COCO_TABLE_COLUMNS if key in val_records[0] and val_records[0].get(key) is not None]
    if not cols:
        cols = [(label, key) for label, key in COCO_TABLE_COLUMNS if key in val_records[0]]

    # ---- CSV ----
    csv_path = out_dir / "coco_validation_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([c[0] for c in cols])
        for r in val_records:
            row = []
            for _, key in cols:
                v = r.get(key)
                if v is None:
                    row.append("")
                elif key == "epoch":
                    row.append(str(int(v)))
                elif isinstance(v, float):
                    row.append(f"{v:.6f}")
                else:
                    row.append(str(v))
            writer.writerow(row)
    print(f"Saved: {csv_path}")

    # ---- Markdown ----
    md_path = out_dir / "coco_validation_metrics.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("## COCO keypoint validation metrics\n\n")
        headers = [c[0] for c in cols]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in val_records:
            row = []
            for _, key in cols:
                v = r.get(key)
                if isinstance(v, (int, float)):
                    row.append(f"{v:.4f}")
                else:
                    row.append(str(v) if v is not None else "—")
            f.write("| " + " | ".join(row) + " |\n")
    print(f"Saved: {md_path}")

    # ---- Matplotlib table figure (for slides/thesis) ----
    fig, ax = plt.subplots(figsize=(max(10, len(cols) * 1.2), max(3, 0.5 + 0.35 * len(val_records))))
    ax.axis("off")
    cell_text = []
    for r in val_records:
        row = []
        for _, key in cols:
            v = r.get(key)
            if v is None:
                row.append("—")
            elif isinstance(v, int):
                row.append(str(v))
            elif isinstance(v, float):
                row.append(f"{v:.4f}")
            else:
                row.append(str(v))
        cell_text.append(row)
    table = ax.table(
        cellText=cell_text,
        colLabels=[c[0] for c in cols],
        loc="center",
        cellLoc="center",
        colColours=["#e8e8e8"] * len(cols),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.2)
    ax.set_title("COCO keypoint validation metrics", fontsize=14, pad=20)
    plt.tight_layout()
    fig.savefig(out_dir / "coco_validation_metrics_table.png")
    fig.savefig(out_dir / "coco_validation_metrics_table.pdf")
    plt.close(fig)
    print(f"Saved: {out_dir / 'coco_validation_metrics_table.png'}")


def plot_training_curves(train_records, val_records, out_dir: Path, smooth_window: int = 101):
    """Create thesis-style plots: loss and accuracy vs iteration; validation AP vs epoch."""
    if not train_records:
        print("No training records found.")
        return

    iters = np.array([r["iteration"] for r in train_records])
    loss = np.array([r["loss"] for r in train_records])
    acc = np.array([r["acc_pose"] for r in train_records])

    # Smooth for display (trim iters to match smoothed length; convolve valid => n = len - window + 1)
    if len(iters) >= smooth_window:
        n = len(iters) - smooth_window + 1
        half = (smooth_window - 1) // 2
        iters_smooth = iters[half : half + n]
        loss_smooth = smooth(loss, smooth_window)
        acc_smooth = smooth(acc, smooth_window)
    else:
        iters_smooth, loss_smooth, acc_smooth = iters, loss, acc

    # ---- Figure 1: Training loss vs iteration (presentation) ----
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(iters, loss, alpha=0.25, color="C0", linewidth=0.8, label="Raw")
    if len(iters_smooth) == len(loss_smooth):
        ax1.plot(iters_smooth, loss_smooth, color="C0", linewidth=2, label="Smoothed")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Training loss")
    ax1.set_title("RTMPose fine-tuning: training loss")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)
    fig1.tight_layout()
    fig1.savefig(out_dir / "training_loss_vs_iteration.png")
    fig1.savefig(out_dir / "training_loss_vs_iteration.pdf")
    plt.close(fig1)
    print(f"Saved: {out_dir / 'training_loss_vs_iteration.png'}")

    # ---- Figure 2: Training accuracy (acc_pose) vs iteration (presentation) ----
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(iters, acc, alpha=0.25, color="C1", linewidth=0.8, label="Raw")
    if len(iters_smooth) == len(acc_smooth):
        ax2.plot(iters_smooth, acc_smooth, color="C1", linewidth=2, label="Smoothed")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Pose accuracy (acc_pose)")
    ax2.set_title("RTMPose fine-tuning: training pose accuracy")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, 1.02)
    fig2.tight_layout()
    fig2.savefig(out_dir / "training_accuracy_vs_iteration.png")
    fig2.savefig(out_dir / "training_accuracy_vs_iteration.pdf")
    plt.close(fig2)
    print(f"Saved: {out_dir / 'training_accuracy_vs_iteration.png'}")

    # ---- Figure 3: Validation metrics vs epoch (if any) ----
    if val_records:
        epochs_val = [r["epoch"] for r in val_records]
        ap = [r["coco_AP"] for r in val_records]
        ap50 = [r["coco_AP_50"] for r in val_records]
        ap75 = [r["coco_AP_75"] for r in val_records]
        ar = [r["coco_AR"] for r in val_records]

        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.plot(epochs_val, ap, "o-", color="C0", linewidth=2, markersize=8, label="AP")
        ax3.plot(epochs_val, ap50, "s-", color="C1", linewidth=1.5, markersize=6, label="AP@0.5")
        ax3.plot(epochs_val, ap75, "^-", color="C2", linewidth=1.5, markersize=6, label="AP@0.75")
        ax3.plot(epochs_val, ar, "d-", color="C3", linewidth=1.5, markersize=6, label="AR")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Score")
        ax3.set_title("Validation: COCO keypoint metrics")
        ax3.legend(loc="lower right")
        ax3.grid(True, alpha=0.3)
        ax3.set_axisbelow(True)
        ax3.set_ylim(0, 1.02)
        fig3.tight_layout()
        fig3.savefig(out_dir / "validation_AP_vs_epoch.png")
        fig3.savefig(out_dir / "validation_AP_vs_epoch.pdf")
        plt.close(fig3)
        print(f"Saved: {out_dir / 'validation_AP_vs_epoch.png'}")

    # ---- Figure 4: Combined 2-panel (loss + accuracy) for slides ----
    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax4a.plot(iters, loss, alpha=0.2, color="C0", linewidth=0.6)
    if len(iters_smooth) == len(loss_smooth):
        ax4a.plot(iters_smooth, loss_smooth, color="C0", linewidth=2, label="Loss (smoothed)")
    ax4a.set_ylabel("Training loss")
    ax4a.set_title("RTMPose fine-tuning: loss and accuracy vs iteration")
    ax4a.legend(loc="upper right")
    ax4a.grid(True, alpha=0.3)
    ax4a.set_axisbelow(True)

    ax4b.plot(iters, acc, alpha=0.2, color="C1", linewidth=0.6)
    if len(iters_smooth) == len(acc_smooth):
        ax4b.plot(iters_smooth, acc_smooth, color="C1", linewidth=2, label="Accuracy (smoothed)")
    ax4b.set_xlabel("Iteration")
    ax4b.set_ylabel("Pose accuracy")
    ax4b.legend(loc="lower right")
    ax4b.grid(True, alpha=0.3)
    ax4b.set_axisbelow(True)
    ax4b.set_ylim(0, 1.02)
    fig4.tight_layout()
    fig4.savefig(out_dir / "training_loss_and_accuracy_combined.png")
    fig4.savefig(out_dir / "training_loss_and_accuracy_combined.pdf")
    plt.close(fig4)
    print(f"Saved: {out_dir / 'training_loss_and_accuracy_combined.png'}")


def main():
    parser = argparse.ArgumentParser(description="Plot RTMPose training log for thesis/slides.")
    parser.add_argument(
        "log_file",
        nargs="?",
        default=r"C:\Users\samso\Downloads\RTMPoseFinetuning\work_dirs\rtmpose_m_carla\20260125_144452\20260125_144452.log",
        help="Path to MMEngine log file",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Directory for plots (default: same as log file)",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=101,
        help="Moving average window for smoothing (default: 101)",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.is_file():
        raise SystemExit(f"Log file not found: {log_path}")

    out_dir = Path(args.output_dir) if args.output_dir else log_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    train_records, val_records = parse_log(str(log_path))
    print(f"Parsed {len(train_records)} training records, {len(val_records)} validation records.")

    plot_training_curves(train_records, val_records, out_dir, smooth_window=args.smooth)
    save_coco_tables(val_records, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
