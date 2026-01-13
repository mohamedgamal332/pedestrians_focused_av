import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from graphs.custom_coco_graph import adjacency_matrix, get_num_node

# --------------------------
# CTR Graph Convolution Layer
# --------------------------
class CTRGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A_init: np.ndarray):
        super().__init__()
        V = A_init.shape[0]
        self.register_buffer('A_init', torch.from_numpy(A_init).float())
        # learnable adjacency residual
        self.A_res = nn.Parameter(torch.zeros_like(self.A_init))
        self.fc = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: N, C, T, V
        A = (self.A_init + self.A_res).to(x.device)
        N, C, T, V = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(N * T, C, V)
        x = torch.matmul(x, A.T)
        x = x.view(N, T, C, V).permute(0, 2, 1, 3).contiguous()
        x = self.fc(x)
        return x

# --------------------------
# CTR Block with TCN + Residual
# --------------------------
class CTRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, kernel_size=9, stride=1):
        super().__init__()
        self.gcn = CTRGraphConv(in_channels, out_channels, A)
        pad = (kernel_size - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels)
        )
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1))
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return F.relu(x + res)

# --------------------------
# Temporal Smoothing / Hysteresis
# --------------------------
class TemporalSmoother(nn.Module):
    def __init__(self, alpha=0.85):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        # x: N, C, T, V
        x_smooth = x.clone()
        for t in range(1, x.size(2)):
            x_smooth[:, :, t, :] = self.alpha * x_smooth[:, :, t-1, :] + (1 - self.alpha) * x[:, :, t, :]
        return x_smooth

# --------------------------
# Enhanced CTR-GCN Model
# --------------------------
class EnhancedCTRGCN(nn.Module):
    def __init__(self, in_channels=2, num_joints=None, num_classes=10, A_norm=None, pretrained_path=None):
        super().__init__()
        if num_joints is None:
            num_joints = get_num_node()
        A = A_norm if A_norm is not None else adjacency_matrix()

        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        # CTR-GCN layers
        self.layer1 = CTRBlock(in_channels, 64, A)
        self.layer2 = CTRBlock(64, 128, A, stride=2)
        self.layer3 = CTRBlock(128, 256, A, stride=2)

        # Temporal smoothing
        self.smoother = TemporalSmoother(alpha=0.85)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        # Load pretrained weights if provided
        if pretrained_path is not None:
            self._load_pretrained_weights(pretrained_path)

    def _load_pretrained_weights(self, path):
        print(f"[INFO] Loading pretrained CTR-GCN weights from {path} ...")
        pretrained_dict = torch.load(path, map_location='cpu')
        model_dict = self.state_dict()
        # Option A: project weights to custom graph (ignore adjacency buffers)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'A_init' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print("[INFO] Pretrained weights loaded and projected to custom graph.")

    def forward(self, x):
        # x: N, C, T, V, 1 -> squeeze last dim
        if x.dim() == 5:
            x = x.squeeze(-1)

        N, C, T, V = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()  # N, C, V, T
        x = x.view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()  # N, C, T, V

        # Forward through CTR blocks + smoothing
        x = self.layer1(x)
        x = self.smoother(x)
        x = self.layer2(x)
        x = self.smoother(x)
        x = self.layer3(x)
        x = self.smoother(x)

        # Global pooling + classification
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --------------------------
# Testing
# --------------------------
if __name__ == '__main__':
    model = EnhancedCTRGCN(in_channels=2, num_joints=12, num_classes=5, pretrained_path='ctrgcn_pretrained.pth')
    x = torch.randn(2, 2, 30, 12, 1)
    print("Output shape:", model(x).shape)
