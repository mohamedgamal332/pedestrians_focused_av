"""
ST-GCN Model Configuration based on MMACTION2 pretrained weights

Pretrained model: stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth
Config file: mmaction2/configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py

Model Structure:
- Backbone: STGCN with 10 stages
- Graph: NTU RGB+D layout (25 keypoints) - but config shows 'coco' mode
- Input: 2D keypoints (x, y coordinates)
- Stages: 
  - Stage 0: in_channels (3) -> 64
  - Stages 1-4: 64 -> 64
  - Stage 5: 64 -> 128 (downsampling)
  - Stages 6-7: 128 -> 128
  - Stage 8: 128 -> 256 (downsampling)
  - Stage 9: 256 -> 256
- Output: 256 channels -> 60 classes (NTU60)

MMACTION2 Structure per stage:
- gcn.{stage}.gcn.conv (Conv2d) - maps to our layer{N}.gcn.fc
- gcn.{stage}.gcn.bn (BatchNorm2d) - we don't have this in GCN
- gcn.{stage}.gcn.A (adjacency matrix buffer) - different graph structure
- gcn.{stage}.gcn.PA (learnable adjacency, if adaptive) - we don't have this
- gcn.{stage}.tcn.conv (Conv2d) - maps to our layer{N}.tcn.2
- gcn.{stage}.tcn.bn (BatchNorm2d) - maps to our layer{N}.tcn.3
- gcn.{stage}.tcn.drop (Dropout) - we don't have this
- gcn.{stage}.residual (unit_tcn or identity) - maps to our layer{N}.residual

Our Custom Structure per layer:
- layer{N}.gcn.A (adjacency matrix buffer) - different
- layer{N}.gcn.fc (Conv2d) - corresponds to gcn.conv
- layer{N}.tcn.0 (BatchNorm2d) - we add this, MMACTION2 doesn't have it before conv
- layer{N}.tcn.1 (ReLU) - no parameters
- layer{N}.tcn.2 (Conv2d) - corresponds to tcn.conv
- layer{N}.tcn.3 (BatchNorm2d) - corresponds to tcn.bn
- layer{N}.residual (Conv2d or Identity) - corresponds to residual

Key Differences:
1. MMACTION2 TCN: Conv -> BN -> Dropout
2. Our TCN: BN -> ReLU -> Conv -> BN
3. MMACTION2 GCN has BN after conv, we don't
4. MMACTION2 uses 25 keypoints (NTU RGB+D), we use 17 (COCO)
5. MMACTION2 has 10 stages, we have 3 layers
"""

STGCN_CONFIG = {
    'pretrained_path': 'pretrained_weights/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth',
    'config_file': 'mmaction2/configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py',
    'model_type': 'STGCN',
    'num_stages': 10,
    'num_joints_pretrained': 25,  # NTU RGB+D
    'num_joints_custom': 17,  # COCO
    'in_channels': 3,  # MMACTION2 default (x, y, confidence)
    'in_channels_custom': 2,  # Our model (x, y only)
    'base_channels': 64,
    'ch_ratio': 2,
    'inflate_stages': [5, 8],
    'down_stages': [5, 8],
    'num_classes': 60,  # NTU60
    'stage_to_layer_mapping': {
        0: 'layer1',  # 3->64
        5: 'layer2',  # 64->128, stride=2
        8: 'layer3',  # 128->256, stride=2
    }
}
