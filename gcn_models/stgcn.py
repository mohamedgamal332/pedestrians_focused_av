import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from graphs.custom_coco_graph import adjacency_matrix, get_num_node

# --------------------------
# Graph Convolution Layer
# --------------------------
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A: np.ndarray):
        super().__init__()
        self.register_buffer('A', torch.from_numpy(A).float())  # V x V
        self.fc = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: N, C, T, V
        N, C, T, V = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # N, T, C, V
        x = x.view(N * T, C, V)
        x = torch.matmul(x, self.A.T)  # Graph convolution
        x = x.view(N, T, C, V).permute(0, 2, 1, 3).contiguous()
        x = self.fc(x)
        return x

# --------------------------
# Temporal Convolution + Residual Block
# --------------------------
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, kernel_size=9, stride=1):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels, A)
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
# Enhanced ST-GCN Model
# --------------------------
class EnhancedSTGCN(nn.Module):
    def __init__(self, in_channels=2, num_joints=None, num_classes=10, A_norm=None, pretrained_path=None):
        super().__init__()
        if num_joints is None:
            num_joints = get_num_node()
        A = A_norm if A_norm is not None else adjacency_matrix()

        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        # ST-GCN layers
        self.layer1 = STGCNBlock(in_channels, 64, A)
        self.layer2 = STGCNBlock(64, 128, A, stride=2)
        self.layer3 = STGCNBlock(128, 256, A, stride=2)

        # Temporal smoothing
        self.smoother = TemporalSmoother(alpha=0.85)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        # Load pretrained weights if provided
        if pretrained_path is not None:
            self._load_pretrained_weights(pretrained_path)

    def _load_pretrained_weights(self, path):
        print(f"[INFO] Loading pretrained ST-GCN weights from {path} ...")
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
        
        # Create mapping from MMACTION2 names to our custom model names
        def map_mmaction2_to_custom(pretrained_key):
            """Map MMACTION2 layer names to custom model names"""
            # Remove 'backbone.' prefix if present
            if pretrained_key.startswith('backbone.'):
                key = pretrained_key[9:]  # Remove 'backbone.'
            elif pretrained_key.startswith('cls_head.'):
                # Skip classifier head, we have our own
                return None
            else:
                key = pretrained_key
            
            # Map data_bn
            if key.startswith('data_bn.'):
                return key  # Same name
            
            # Map GCN stages to our layers
            # MMACTION2 structure: gcn.{stage}.gcn.{component} and gcn.{stage}.tcn.{component}
            # Our structure: layer{N}.gcn.{component} and layer{N}.tcn.{component}
            if key.startswith('gcn.'):
                parts = key.split('.')
                if len(parts) >= 3:
                    try:
                        stage_num = int(parts[1])
                        component = parts[2]  # 'gcn' or 'tcn'
                        rest = '.'.join(parts[3:]) if len(parts) > 3 else ''
                        
                        # MMACTION2 STGCN has 10 stages: gcn.0 (3->64), gcn.1..3 (64->64),
                        # gcn.4 (64->128, stride 2), gcn.5..6 (128->128), gcn.7 (128->256, stride 2), gcn.8..9 (256->256).
                        # Our model has 3 blocks: layer1 (64), layer2 (128), layer3 (256).
                        # Load only the first block of each channel stage to avoid overwriting.
                        layer_map = {0: 'layer1', 4: 'layer2', 7: 'layer3'}
                        
                        if stage_num in layer_map:
                            target_layer = layer_map[stage_num]
                            
                            if component == 'gcn':
                                # MMACTION2: gcn.{stage}.gcn.conv -> layer{N}.gcn.fc
                                # MMACTION2: gcn.{stage}.gcn.bn -> skip (we don't have BN in GCN)
                                # MMACTION2: gcn.{stage}.gcn.PA -> skip (we don't have learnable adjacency)
                                # MMACTION2: gcn.{stage}.gcn.A -> skip (graph structure, different)
                                if rest.startswith('conv.'):
                                    # Map conv to fc
                                    return f"{target_layer}.gcn.fc.{rest[5:]}"  # Remove 'conv.' prefix
                                elif rest.startswith('bn.'):
                                    # Skip BatchNorm in GCN (we don't have it)
                                    return None
                                elif 'PA' in rest or rest == 'A':
                                    # Skip learnable adjacency and graph structure
                                    return None
                                    
                            elif component == 'tcn':
                                # MMACTION2 unit_tcn structure: conv (Conv2d) -> bn (BatchNorm2d) -> drop (Dropout)
                                # Our TCN structure: 0 (BatchNorm2d) -> 1 (ReLU) -> 2 (Conv2d) -> 3 (BatchNorm2d)
                                # Mapping:
                                #   tcn.conv.* -> tcn.2.* (Conv2d)
                                #   tcn.bn.* -> tcn.3.* (BatchNorm2d)
                                #   tcn.drop.* -> skip (we don't have dropout)
                                if rest.startswith('conv.'):
                                    # Map conv to our tcn.2 (Conv2d)
                                    return f"{target_layer}.tcn.2.{rest[5:]}"  # Remove 'conv.' prefix
                                elif rest.startswith('bn.'):
                                    # Map bn to our tcn.3 (BatchNorm2d)
                                    return f"{target_layer}.tcn.3.{rest[3:]}"  # Remove 'bn.' prefix
                                elif rest.startswith('drop.'):
                                    # Skip dropout (we don't have it)
                                    return None
                                else:
                                    # Try direct mapping for other cases
                                    return f"{target_layer}.tcn.{rest}"
                            
                            elif component == 'residual':
                                # MMACTION2 residual: unit_tcn (Conv2d -> BN) or identity
                                # Our residual: Conv2d or Identity
                                # Map: residual.conv.* -> residual.*, residual.bn.* -> skip (we don't have BN in residual)
                                if rest.startswith('conv.'):
                                    return f"{target_layer}.residual.{rest[5:]}"
                                elif rest.startswith('bn.'):
                                    # Skip BN in residual (we don't have it)
                                    return None
                                else:
                                    return f"{target_layer}.residual.{rest}"
                            
                    except (ValueError, IndexError):
                        pass
            
            return None
        
        # Build mapped dictionary
        filtered_dict = {}
        unmapped_keys = []
        
        # Print sample keys for debugging
        sample_keys = list(pretrained_dict.keys())[:10]
        print(f"[DEBUG] Sample pretrained keys: {sample_keys}")
        print(f"[DEBUG] Sample model keys: {list(model_dict.keys())[:10]}")
        
        for pretrained_key, pretrained_value in pretrained_dict.items():
            # Skip graph adjacency matrices
            if 'A' in pretrained_key and ('gcn' in pretrained_key.lower() or 'graph' in pretrained_key.lower()):
                continue
            
            # Try direct match first
            if pretrained_key in model_dict:
                if model_dict[pretrained_key].shape == pretrained_value.shape:
                    filtered_dict[pretrained_key] = pretrained_value
                    continue
            
            # Try mapping from MMACTION2 format
            mapped_key = map_mmaction2_to_custom(pretrained_key)
            if mapped_key and mapped_key in model_dict:
                if model_dict[mapped_key].shape == pretrained_value.shape:
                    filtered_dict[mapped_key] = pretrained_value
                else:
                    unmapped_keys.append(f"{pretrained_key} -> {mapped_key} (shape mismatch: {pretrained_value.shape} vs {model_dict[mapped_key].shape})")
            else:
                unmapped_keys.append(pretrained_key)
        
        # Update model dict
        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict, strict=False)
        
        print(f"[INFO] Pretrained weights loaded: {len(filtered_dict)}/{len(pretrained_dict)} layers matched")
        
        # Show what was successfully mapped
        if filtered_dict:
            mapped_examples = list(filtered_dict.keys())[:5]
            print(f"[INFO] Successfully mapped examples: {mapped_examples}")
        
        if unmapped_keys and len(unmapped_keys) <= 20:
            print(f"[INFO] Unmapped keys (first 20): {unmapped_keys[:20]}")
        elif unmapped_keys:
            print(f"[INFO] {len(unmapped_keys)} keys could not be mapped")
            
        # Show mapping statistics
        mapped_by_type = {}
        for key in filtered_dict.keys():
            if 'gcn' in key:
                mapped_by_type['GCN'] = mapped_by_type.get('GCN', 0) + 1
            elif 'tcn' in key:
                mapped_by_type['TCN'] = mapped_by_type.get('TCN', 0) + 1
            elif 'residual' in key:
                mapped_by_type['Residual'] = mapped_by_type.get('Residual', 0) + 1
            elif 'data_bn' in key:
                mapped_by_type['DataBN'] = mapped_by_type.get('DataBN', 0) + 1
        
        if mapped_by_type:
            print(f"[INFO] Mapped by component: {mapped_by_type}")

    def forward(self, x):
        # x: N, C, T, V, 1 -> remove last dim
        if x.dim() == 5:
            x = x.squeeze(-1)

        N, C, T, V = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()  # N, C, V, T
        x = x.view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()  # N, C, T, V

        # Forward through ST-GCN blocks
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
    model = EnhancedSTGCN(in_channels=2, num_joints=12, num_classes=5, pretrained_path='stgcn_pretrained.pth')
    x = torch.randn(2, 2, 30, 12, 1)
    print("Output shape:", model(x).shape)
