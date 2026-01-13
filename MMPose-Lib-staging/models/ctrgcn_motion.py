import torch
import torch.nn as nn
import torch.nn.functional as F
from graphs.custom_coco_graph import adjacency_matrix, get_num_node
from models.ctrgcn import CTRGCN, EnhancedCTRGCN  # use our enhanced version

# --------------------------
# CTR-GCN with Motion Stream (Enhanced)
# --------------------------
class EnhancedCTRGCN_Motion(nn.Module):
    def __init__(self, in_channels=2, num_joints=None, num_classes=10,
                 A_norm=None, pretrained_spatial=None, pretrained_motion=None):
        super().__init__()
        if num_joints is None:
            num_joints = get_num_node()
        A = A_norm if A_norm is not None else adjacency_matrix()

        # Spatial branch
        self.spatial = EnhancedCTRGCN(in_channels=in_channels, num_joints=num_joints,
                                      num_classes=num_classes, A_norm=A, pretrained_path=pretrained_spatial)
        # Motion branch (Δx, Δy)
        self.motion_branch = EnhancedCTRGCN(in_channels=in_channels, num_joints=num_joints,
                                            num_classes=num_classes, A_norm=A, pretrained_path=pretrained_motion)

        # Final classifier (fusion)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: N,C,T,V,1 -> squeeze last dim
        if x.dim() == 5:
            x = x.squeeze(-1)

        # Compute motion (temporal deltas)
        motion = x[:, :, 1:, :] - x[:, :, :-1, :]
        motion = torch.cat([motion, torch.zeros_like(motion[:, :, :1, :])], dim=2)  # pad

        # Run both branches
        s_feat = self.spatial(x)  # N x num_classes (or features if we modify EnhancedCTRGCN)
        m_feat = self.motion_branch(motion)

        # Fusion: element-wise sum in logit space
        out = s_feat + m_feat
        return out

# --------------------------
# Testing
# --------------------------
if __name__ == '__main__':
    model = EnhancedCTRGCN_Motion(
        in_channels=2,
        num_joints=12,
        num_classes=5,
        pretrained_spatial='ctrgcn_pretrained.pth',
        pretrained_motion='ctrgcn_pretrained.pth'
    )
    x = torch.randn(2, 2, 30, 12, 1)
    print("Output shape:", model(x).shape)
