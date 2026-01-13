import argparse
import importlib.util
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

# Dynamically import TrajectoryDataset from train_mamba_trajectory.py
def load_dataset_class(script_path: Path):
    spec = importlib.util.spec_from_file_location("train_mamba_mod", str(script_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod.TrajectoryDataset, mod.INPUT_LEN, mod.PRED_LEN, mod.STATE_DIM


class LSTMModel(nn.Module):
    def __init__(self, d_model: int = 128, hidden_size: int = 128, num_layers: int = 1,
                 dropout: float = 0.0, pred_len: int = 8):
        super().__init__()
        self.input_proj = nn.Linear(8, d_model)
        self.lstm = nn.LSTM(d_model, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_size, pred_len * 3)
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 8)
        z = self.input_proj(x)
        y, _ = self.lstm(z)
        last = y[:, -1, :]
        out = self.head(last)
        out = out.view(-1, self.pred_len, 3)
        return out


def ADE_FDE(pred_pos: torch.Tensor, gt_pos: torch.Tensor) -> Tuple[float, float]:
    diff = pred_pos - gt_pos
    dists = torch.norm(diff, dim=-1)
    return dists.mean().item(), dists[:, -1].mean().item()


def collate_batch(batch):
    inps = torch.stack([b[0] for b in batch], dim=0)
    tgts = torch.stack([b[1] for b in batch], dim=0)
    return inps, tgts


def train_epoch(model, loader, opt, device, loss_fn, clip_grad=None):
    model.train()
    total = 0.0
    for inp, tgt in loader:
        inp = inp.to(device)
        tgt = tgt.to(device)
        opt.zero_grad()
        pred = model(inp)
        loss = loss_fn(pred, tgt)
        loss.backward()
        if clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        opt.step()
        total += loss.item() * inp.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, dataset):
    model.eval()
    mse = nn.MSELoss()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    count = 0
    for inp, tgt in loader:
        inp = inp.to(device)
        tgt = tgt.to(device)
        pred = model(inp)
        loss = mse(pred, tgt)
        total_loss += loss.item() * inp.size(0)

        # unnormalize similar to Mamba script
        delta_mean = dataset.delta_mean.to(device)
        delta_std = dataset.delta_std.to(device)
        pred_unnorm = pred * delta_std.view(1, 1, 3) + delta_mean.view(1, 1, 3)
        tgt_unnorm = tgt * delta_std.view(1, 1, 3) + delta_mean.view(1, 1, 3)

        state_mean = dataset.state_mean.to(device)
        state_std = dataset.state_std.to(device)
        inp_last = inp[:, -1, :] * state_std.view(1, -1) + state_mean.view(1, -1)
        # convert scaled x/y back to pixels (dataset uses IMG_W/IMG_H scaling)
        IMG_W = 1928.88
        IMG_H = 881.76
        C_last = inp_last[:, 0:3].clone()
        C_last[:, 0] = C_last[:, 0] * IMG_W
        C_last[:, 1] = C_last[:, 1] * IMG_H

        pred_px = pred_unnorm.clone()
        tgt_px = tgt_unnorm.clone()
        pred_px[:, :, 0] = pred_px[:, :, 0] * IMG_W
        pred_px[:, :, 1] = pred_px[:, :, 1] * IMG_H
        tgt_px[:, :, 0] = tgt_px[:, :, 0] * IMG_W
        tgt_px[:, :, 1] = tgt_px[:, :, 1] * IMG_H

        B = pred.size(0)
        pred_pos = torch.zeros((B, pred.size(1), 3), device=device)
        gt_pos = torch.zeros_like(pred_pos)
        for k in range(pred.size(1)):
            if k == 0:
                pred_pos[:, 0, :] = C_last + pred_px[:, 0, :]
                gt_pos[:, 0, :] = C_last + tgt_px[:, 0, :]
            else:
                pred_pos[:, k, :] = pred_pos[:, k - 1, :] + pred_px[:, k, :]
                gt_pos[:, k, :] = gt_pos[:, k - 1, :] + tgt_px[:, k, :]

        ade, fde = ADE_FDE(pred_pos, gt_pos)
        total_ade += ade * inp.size(0)
        total_fde += fde * inp.size(0)
        count += inp.size(0)

    return total_loss / len(loader.dataset), total_ade / count, total_fde / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs="?", default=str(Path(__file__).resolve().parent.parent / "hrnet_depth"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", type=str, default="lstm_traj.pt")
    args = parser.parse_args()

    script_path = Path(__file__).resolve().parent / "train_mamba_trajectory.py"
    TrajectoryDataset, INPUT_LEN, PRED_LEN, STATE_DIM = load_dataset_class(script_path)

    device = torch.device(args.device)

    dataset = TrajectoryDataset(Path(args.data_dir))
    n = len(dataset)
    if n == 0:
        raise SystemExit(f"No samples found in {args.data_dir}")
    n_train = int(0.9 * n)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n - n_train])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = LSTMModel(d_model=args.d_model, hidden_size=args.hidden_size, num_layers=args.num_layers,
                      dropout=args.dropout, pred_len=PRED_LEN).to(device)

    if args.loss == "mse":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.SmoothL1Loss()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, opt, device, loss_fn, clip_grad=args.grad_clip or None)
        val_loss, ade, fde = evaluate(model, val_loader, device, dataset)
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, ADE={ade:.6f}, FDE={fde:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, args.save)


if __name__ == "__main__":
    main()
