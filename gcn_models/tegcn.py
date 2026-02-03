import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from graphs.custom_coco_graph import adjacency_matrix, get_num_node

# --------------------------
# Taylor Graph Convolution
# --------------------------
class TaylorGraphConv(nn.Module):
    """
    Graph Convolution using 2nd-order Taylor expansion approximation:
        X' = X + alpha * A @ X + beta * (A @ (A @ X))
    """
    def __init__(self, in_channels, out_channels, A: np.ndarray, alpha=0.5, beta=0.25):
        super().__init__()
        self.register_buffer('A', torch.from_numpy(A).float())
        self.alpha = alpha
        self.beta = beta
        self.fc = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.shape
        A = self.A.to(x.device)
        x_t = x
        x_t = x_t + self.alpha * torch.matmul(x_t.permute(0, 2, 1, 3).reshape(N*T, C, V), A.T).reshape(N, T, C, V).permute(0, 2, 1, 3)
        x_t = x_t + self.beta * torch.matmul(x_t.permute(0, 2, 1, 3).reshape(N*T, C, V), torch.matmul(A, A).T).reshape(N, T, C, V).permute(0, 2, 1, 3)
        x_t = self.fc(x_t)
        return x_t

# --------------------------
# TE-GCN Block with TCN + Residual
# --------------------------
class TEGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, kernel_size=9, stride=1):
        super().__init__()
        self.gcn = TaylorGraphConv(in_channels, out_channels, A)
        pad = (kernel_size - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size,1), padding=(pad,0), stride=(stride,1)),
            nn.BatchNorm2d(out_channels)
        )
        if in_channels != out_channels or stride !=1:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride,1))
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return F.relu(x + res)

# --------------------------
# Temporal Smoothing
# --------------------------
class TemporalSmoother(nn.Module):
    def __init__(self, alpha=0.85):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        x_smooth = x.clone()
        for t in range(1, x.size(2)):
            x_smooth[:, :, t, :] = self.alpha * x_smooth[:, :, t-1, :] + (1 - self.alpha) * x[:, :, t, :]
        return x_smooth

# --------------------------
# Taylor Expansion GCN Model
# --------------------------
class TE_GCN(nn.Module):
    def __init__(self, in_channels=2, num_joints=None, num_classes=10, A_norm=None, pretrained_path=None):
        super().__init__()
        if num_joints is None:
            num_joints = get_num_node()
        A = A_norm if A_norm is not None else adjacency_matrix()

        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        self.layer1 = TEGCNBlock(in_channels, 64, A)
        self.layer2 = TEGCNBlock(64, 128, A, stride=2)
        self.layer3 = TEGCNBlock(128, 256, A, stride=2)

        self.smoother = TemporalSmoother(alpha=0.85)

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, num_classes)

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path):
        print(f"[INFO] Loading pretrained TE-GCN weights from {path} ...")
        checkpoint = torch.load(path, map_location='cpu')
        
        # Handle different checkpoint formats (MMACTION2 uses 'state_dict' key)
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                pretrained_dict = checkpoint['model_state_dict']
            else:
                # Assume the dict itself is the state_dict
                pretrained_dict = checkpoint
        else:
            pretrained_dict = checkpoint
        
        model_dict = self.state_dict()
        filtered_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered_dict[k] = v
            elif k in model_dict:
                print(f"[WARNING] Shape mismatch for {k}: pretrained {v.shape} vs model {model_dict[k].shape}, skipping")
        
        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict, strict=False)
        print(f"[INFO] Pretrained weights loaded: {len(filtered_dict)}/{len(pretrained_dict)} layers matched")

    def forward(self, x):
        if x.dim()==5:
            x = x.squeeze(-1)
        N,C,T,V = x.shape
        x = x.permute(0,1,3,2).contiguous()
        x = x.view(N,C*V,T)
        x = self.data_bn(x)
        x = x.view(N,C,V,T).permute(0,1,3,2).contiguous()

        x = self.layer1(x)
        x = self.smoother(x)
        x = self.layer2(x)
        x = self.smoother(x)
        x = self.layer3(x)
        x = self.smoother(x)

        x = self.pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

# --------------------------
# Testing
# --------------------------
if __name__=='__main__':
    model = TE_GCN(in_channels=2, num_joints=12, num_classes=5, pretrained_path='te_gcn_pretrained.pth')
    x = torch.randn(2,2,30,12,1)
    print("Output shape:", model(x).shape)
