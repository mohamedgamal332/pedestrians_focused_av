import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

try:
    from mamba_ssm import Mamba
except Exception as e:
    raise ImportError("mamba_ssm not available. Ensure it's installed and importable.")


INPUT_LEN = 16
PRED_LEN = 8
STATE_DIM = 8  # [Cx, Cy, Cz, Vx, Vy, Vz, w, h]

# image normalization (from dataset stats)
IMG_W = 1928.88
IMG_H = 881.76


class TrajectoryDataset(Dataset):
    def __init__(self, root: Path, input_len: int = INPUT_LEN, pred_len: int = PRED_LEN):
        self.root = Path(root)
        self.input_len = input_len
        self.pred_len = pred_len
        self.samples = []  # list of (input_states, target_deltas)
        self._build()
        # compute normalization stats
        self._compute_stats()

    def _compute_stats(self):
        # collect all input state values and target deltas
        if not self.samples:
            # default tensors
            self.state_mean = torch.zeros(STATE_DIM)
            self.state_std = torch.ones(STATE_DIM)
            self.delta_mean = torch.zeros(3)
            self.delta_std = torch.ones(3)
            return

        all_states = []
        all_deltas = []
        for inp_states, tgt_deltas in self.samples:
            for s in inp_states:
                # scale x/y, Vx/Vy, w/h by image dims
                sx = float(s[0]) / IMG_W if s[0] is not None else 0.0
                sy = float(s[1]) / IMG_H if s[1] is not None else 0.0
                sz = float(s[2]) if s[2] is not None else 0.0
                svx = float(s[3]) / IMG_W if s[3] is not None else 0.0
                svy = float(s[4]) / IMG_H if s[4] is not None else 0.0
                svz = float(s[5]) if s[5] is not None else 0.0
                sw = float(s[6]) / IMG_W if s[6] is not None else 0.0
                sh = float(s[7]) / IMG_H if s[7] is not None else 0.0
                all_states.append([sx, sy, sz, svx, svy, svz, sw, sh])
            for d in tgt_deltas:
                # scale delta x/y by image dims
                dx = float(d[0]) / IMG_W
                dy = float(d[1]) / IMG_H
                dz = float(d[2])
                all_deltas.append([dx, dy, dz])

        S = torch.tensor(all_states, dtype=torch.float32)
        D = torch.tensor(all_deltas, dtype=torch.float32)

        self.state_mean = S.mean(dim=0)
        self.state_std = S.std(dim=0)
        self.delta_mean = D.mean(dim=0)
        self.delta_std = D.std(dim=0)

        # avoid zero std
        self.state_std[self.state_std == 0] = 1.0
        self.delta_std[self.delta_std == 0] = 1.0

    def _load_file(self, path: Path) -> List[Dict]:
        return json.loads(path.read_text())

    def _build(self):
        files = list(self.root.rglob("*_states.json")) + list(self.root.rglob("*.json"))
        files = [f for f in files if f.is_file()]
        for f in files:
            try:
                data = self._load_file(f)
            except Exception:
                continue

            # group by person_id within this file
            persons: Dict[str, List[Tuple[int, List[float]]]] = {}
            for frame in data:
                idx = frame.get("frame_index")
                for p in frame.get("people", []):
                    pid = p.get("person_id")
                    state = p.get("state")
                    if pid is None or state is None:
                        continue
                    persons.setdefault(pid, []).append((idx, state))

            for pid, seq in persons.items():
                # sort by frame index
                seq.sort(key=lambda x: x[0])
                # break into contiguous segments
                frames = [fi for fi, _ in seq]
                states = [s for _, s in seq]
                segments = []
                if not frames:
                    continue
                seg_start = 0
                for i in range(1, len(frames)):
                    if frames[i] != frames[i - 1] + 1:
                        segments.append((frames[seg_start:i], states[seg_start:i]))
                        seg_start = i
                segments.append((frames[seg_start:], states[seg_start:]))

                for frs, sts in segments:
                    L = len(sts)
                    if L < self.input_len + self.pred_len:
                        continue
                    # extract centers
                    centers = []
                    valid_mask = []
                    for s in sts:
                        Cx, Cy, Cz = s[0], s[1], s[2]
                        if Cx is None or Cy is None or Cz is None:
                            centers.append((0.0, 0.0, 0.0))
                            valid_mask.append(False)
                        else:
                            centers.append((float(Cx), float(Cy), float(Cz)))
                            valid_mask.append(True)

                    # skip sequences that contain invalid centers in the required span
                    for start in range(0, L - (self.input_len + self.pred_len) + 1):
                        window_valid = all(valid_mask[start : start + self.input_len + self.pred_len])
                        if not window_valid:
                            continue
                        inp_states = sts[start : start + self.input_len]
                        # compute target deltas: for k=0..pred_len-1 -> C[start+input_len + k] - C[start+input_len-1 + k]
                        t0 = start + self.input_len - 1
                        target_deltas = []
                        for k in range(self.pred_len):
                            Ca = centers[t0 + k]
                            Cb = centers[t0 + k + 1]
                            dx = Cb[0] - Ca[0]
                            dy = Cb[1] - Ca[1]
                            dz = Cb[2] - Ca[2]
                            target_deltas.append([dx, dy, dz])

                        self.samples.append((inp_states, target_deltas))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        # return tensors
        inp_t = torch.tensor(inp, dtype=torch.float32)  # (input_len, STATE_DIM)
        tgt_t = torch.tensor(tgt, dtype=torch.float32)  # (pred_len, 3)
        # apply same scaling as used for stats: scale x/y, Vx/Vy, w/h
        inp_t = inp_t.clone()
        inp_t[:, 0] = inp_t[:, 0] / IMG_W
        inp_t[:, 1] = inp_t[:, 1] / IMG_H
        inp_t[:, 3] = inp_t[:, 3] / IMG_W
        inp_t[:, 4] = inp_t[:, 4] / IMG_H
        inp_t[:, 6] = inp_t[:, 6] / IMG_W
        inp_t[:, 7] = inp_t[:, 7] / IMG_H
        # scale target deltas x/y
        tgt_t = tgt_t.clone()
        tgt_t[:, 0] = tgt_t[:, 0] / IMG_W
        tgt_t[:, 1] = tgt_t[:, 1] / IMG_H

        # normalize using dataset stats
        inp_t = (inp_t - self.state_mean) / self.state_std
        tgt_t = (tgt_t - self.delta_mean) / self.delta_std
        return inp_t, tgt_t


class TrajectoryModel(nn.Module):
    def __init__(self, d_model: int = 128, d_state: int = 64, d_conv: int = 4, expand: int = 2, pred_len: int = PRED_LEN):
        super().__init__()
        self.d_model = d_model
        # project raw state vector (STATE_DIM) to model dimension
        self.input_proj = nn.Linear(STATE_DIM, d_model)
        # use Mamba SSM with configurable hyperparameters
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.pred_len = pred_len
        # map last hidden to pred_len * 3
        self.output_head = nn.Linear(d_model, pred_len * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, STATE_DIM)
        z = self.input_proj(x)  # (B, T, d_model)
        y = self.mamba(z)  # expect (B, T, d_model)
        last = y[:, -1, :]  # (B, d_model)
        out = self.output_head(last)  # (B, pred_len*3)
        out = out.view(-1, self.pred_len, 3)  # (B, pred_len, 3)
        return out


def ADE_FDE(pred_pos: torch.Tensor, gt_pos: torch.Tensor) -> Tuple[float, float]:
    # pred_pos, gt_pos: (B, pred_len, 3)
    diff = pred_pos - gt_pos
    dists = torch.norm(diff, dim=-1)  # (B, pred_len)
    ade = dists.mean().item()
    fde = dists[:, -1].mean().item()
    return ade, fde


def collate_batch(batch):
    inps = torch.stack([b[0] for b in batch], dim=0)
    tgts = torch.stack([b[1] for b in batch], dim=0)
    return inps, tgts


def train_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    mse = nn.MSELoss()
    for inp, tgt in loader:
        inp = inp.to(device)  # (B, T, 8)
        tgt = tgt.to(device)  # (B, pred_len, 3)
        opt.zero_grad()
        pred = model(inp)  # (B, pred_len, 3)
        loss = mse(pred, tgt)
        loss.backward()
        opt.step()
        total_loss += loss.item() * inp.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, dataset: TrajectoryDataset):
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
        # unnormalize predicted and target deltas back to scaled units (x,y scaled by IMG_W/IMG_H)
        delta_mean = dataset.delta_mean.to(device)
        delta_std = dataset.delta_std.to(device)
        pred_unnorm = pred * delta_std.view(1, 1, 3) + delta_mean.view(1, 1, 3)
        tgt_unnorm = tgt * delta_std.view(1, 1, 3) + delta_mean.view(1, 1, 3)

        # reconstruct positions
        # we need C_last in original pixel units: unnormalize input centers (which are in scaled units)
        state_mean = dataset.state_mean.to(device)
        state_std = dataset.state_std.to(device)
        inp_last = inp[:, -1, :] * state_std.view(1, -1) + state_mean.view(1, -1)
        # inp_last is in scaled units for x/y; convert back to pixels for position reconstruction
        C_last = inp_last[:, 0:3].clone()
        C_last[:, 0] = C_last[:, 0] * IMG_W
        C_last[:, 1] = C_last[:, 1] * IMG_H
        # build predicted positions
        B = pred.size(0)
        # pred_unnorm/tgt_unnorm are in scaled units for x/y; convert to pixel deltas for reconstruction
        pred_px = pred_unnorm.clone()
        tgt_px = tgt_unnorm.clone()
        pred_px[:, :, 0] = pred_px[:, :, 0] * IMG_W
        pred_px[:, :, 1] = pred_px[:, :, 1] * IMG_H
        tgt_px[:, :, 0] = tgt_px[:, :, 0] * IMG_W
        tgt_px[:, :, 1] = tgt_px[:, :, 1] * IMG_H

        pred_pos = torch.zeros((B, pred.size(1), 3), device=device)
        for k in range(pred.size(1)):
            if k == 0:
                pred_pos[:, 0, :] = C_last + pred_px[:, 0, :]
            else:
                pred_pos[:, k, :] = pred_pos[:, k - 1, :] + pred_px[:, k, :]

        # ground truth positions: reconstruct from last input center and cumulative deltas (pixels)
        gt_pos = torch.zeros_like(pred_pos)
        for k in range(pred.size(1)):
            if k == 0:
                gt_pos[:, 0, :] = C_last + tgt_px[:, 0, :]
            else:
                gt_pos[:, k, :] = gt_pos[:, k - 1, :] + tgt_px[:, k, :]

        ade, fde = ADE_FDE(pred_pos, gt_pos)
        total_ade += ade * inp.size(0)
        total_fde += fde * inp.size(0)
        count += inp.size(0)

    return total_loss / len(loader.dataset), total_ade / count, total_fde / count


def main():
    parser = argparse.ArgumentParser(description="Train Mamba SSM for pedestrian trajectory prediction")
    parser.add_argument("data_dir", nargs="?", default=str(Path(__file__).resolve().parent.parent / "hrnet_depth"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-state", type=int, default=64, help="Internal Mamba state dimension (d_state).")
    parser.add_argument("--d-conv", type=int, default=4, help="Mamba conv parameter (d_conv).")
    parser.add_argument("--expand", type=int, default=2, help="Mamba expand factor.")
    parser.add_argument("--save", type=str, default="mamba_traj.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    dataset = TrajectoryDataset(Path(args.data_dir))
    # simple split
    n = len(dataset)
    if n == 0:
        raise SystemExit(f"No samples found in {args.data_dir}")
    n_train = int(0.9 * n)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n - n_train])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = TrajectoryModel(d_model=args.d_model, d_state=args.d_state, d_conv=args.d_conv, expand=args.expand).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, opt, device)
        val_loss, ade, fde = evaluate(model, val_loader, device, dataset)
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, ADE={ade:.6f}, FDE={fde:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, args.save)


if __name__ == "__main__":
    main()
