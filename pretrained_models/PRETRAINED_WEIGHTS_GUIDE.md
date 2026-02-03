# Pretrained Weights Guide for GCN Models

This document provides information on where to obtain pretrained weights for each model in the `models/` directory, with consideration for 17-keypoint skeleton graphs (COCO format).

---

## Models Overview

| Model | File | Keypoint Compatibility | Pretrained Source |
|-------|------|------------------------|-------------------|
| **ST-GCN** | `stgcn.py` | NTU (25 joints) → Adaptable to 17 | GitHub: yysijie/st-gcn |
| **CTR-GCN** | `ctrgcn.py` | NTU (25 joints) → Adaptable to 17 | GitHub: Uason-Chen/CTR-GCN |
| **CTR-GCN Motion** | `ctrgcn_motion.py` | NTU (25 joints) → Adaptable to 17 | Same as CTR-GCN |
| **TE-GCN** | `tegcn.py` | Custom implementation | Limited availability |
| **Hyperformer** | `sht.py` (wrapper) | NTU60 (25 joints) → Adaptable to 17 | GitHub: ZhouYuxuanYX/Hyperformer |

---

## 1. ST-GCN (Spatial Temporal Graph Convolutional Network)

### Paper
- **Title**: Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition
- **Conference**: AAAI 2018
- **Authors**: Sijie Yan, Yuanjun Xiong, Dahua Lin

### Pretrained Weights Location
**Official Repository**: https://github.com/yysijie/st-gcn

**Download Instructions**:
1. Visit the official repository
2. Check the `models/` directory or releases section
3. Look for NTU RGB+D 60 or NTU RGB+D 120 pretrained models
4. Common checkpoint names:
   - `st_gcn.kinetics-6fa43f73.pt`
   - Models trained on NTU RGB+D datasets

**Keypoint Configuration**:
- **Original**: NTU RGB+D (25 joints)
- **Target**: COCO (17 keypoints)
- **Adaptation Strategy**: 
  - Load pretrained weights excluding graph adjacency matrices (`'gcn.A'`)
  - The feature extraction layers (TCN, pooling, FC) can transfer
  - Graph structure will be retrained on your 17-keypoint topology

### Usage in Your Code
```python
from models.stgcn import EnhancedSTGCN

model = EnhancedSTGCN(
    in_channels=2,
    num_joints=17,  # COCO keypoints
    num_classes=10,
    pretrained_path='path/to/st_gcn_ntu60.pth'
)
```

---

## 2. CTR-GCN (Channel-wise Topology Refinement GCN)

### Paper
- **Title**: Channel-Wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition
- **Conference**: ICCV 2021
- **Authors**: Yuxin Chen, Ziqi Zhang, Chunfeng Yuan, et al.

### Pretrained Weights Location
**Official Repository**: https://github.com/Uason-Chen/CTR-GCN

**Download Instructions**:
1. Clone or visit: https://github.com/Uason-Chen/CTR-GCN
2. Navigate to the repository and check:
   - README.md for download links
   - `work_dir/` or `checkpoints/` directory
   - GitHub Releases section
3. Look for pretrained models on:
   - **NTU RGB+D 60** (Cross-Subject and Cross-View)
   - **NTU RGB+D 120** (Cross-Subject and Cross-Setup)
   - **NW-UCLA**

**Keypoint Configuration**:
- **Original**: NTU RGB+D (25 joints)
- **Target**: COCO (17 keypoints)
- **Adaptation Strategy**:
  - Your implementation already handles this via `_load_pretrained_weights()`
  - Loads all weights except adjacency matrices (`'A_init'`)
  - Channel-wise topology refinement parameters (`A_res`) will be randomly initialized

### Usage in Your Code
```python
from models.ctrgcn import EnhancedCTRGCN

model = EnhancedCTRGCN(
    in_channels=2,
    num_joints=17,  # COCO keypoints
    num_classes=10,
    pretrained_path='path/to/ctrgcn_ntu60_xsub.pth'
)
```

### Additional Resources
- **MMAction2 Integration**: CTR-GCN is also available in MMAction2 model zoo
- Check: https://github.com/open-mmlab/mmaction2/tree/main/configs/skeleton/stgcn

---

## 3. CTR-GCN with Motion Stream

### Paper
Same as CTR-GCN above

### Pretrained Weights Location
**Source**: Same repository as CTR-GCN (Uason-Chen/CTR-GCN)

**Download Instructions**:
- Download two sets of weights (one for spatial branch, one for motion branch)
- Typically the same base CTR-GCN weights are used for both branches
- Fine-tune motion branch separately if needed

### Usage in Your Code
```python
from models.ctrgcn_motion import EnhancedCTRGCN_Motion

model = EnhancedCTRGCN_Motion(
    in_channels=2,
    num_joints=17,  # COCO keypoints
    num_classes=10,
    pretrained_spatial='path/to/ctrgcn_ntu60_xsub.pth',
    pretrained_motion='path/to/ctrgcn_ntu60_xsub.pth'  # Can use same weights initially
)
```

---

## 4. TE-GCN (Taylor Expansion GCN)

### Paper
- **Title**: Graph Convolutional Networks Based on Taylor Approximation
- **Reference**: Similar approaches in literature, but this is a custom implementation

### Pretrained Weights Location
**Status**: ⚠️ **Limited Public Availability**

**Alternative Sources**:
1. **General Taylor-based GCN research**:
   - Search arXiv: "Taylor expansion graph convolutional networks"
   - Check related repositories on GitHub
   
2. **Option 1**: Train from scratch on your dataset
   - Your implementation in `tegcn.py` is ready for training
   - Use pretrained weights from ST-GCN or CTR-GCN as initialization if available

3. **Option 2**: Use ST-GCN or CTR-GCN weights as warm start
   - Initialize TE-GCN with ST-GCN weights where architecturally compatible
   - The TCN and FC layers should be transferable

### Usage in Your Code
```python
from models.tegcn import TE_GCN

# Train from scratch or use partial initialization
model = TE_GCN(
    in_channels=2,
    num_joints=17,  # COCO keypoints
    num_classes=10,
    pretrained_path=None  # Train from scratch initially
)
```

---

## 5. Hyperformer (Hypergraph Transformer)

### Paper
- **Title**: Hypergraph Transformer for Skeleton-based Action Recognition
- **Authors**: Yuxuan Zhou, Zhi-Qi Cheng, et al.
- **Publication**: arXiv 2022 (arXiv:2211.09590)

### Pretrained Weights Location
**Official Repository**: https://github.com/ZhouYuxuanYX/Hyperformer

**Download Instructions**:
1. Visit: https://github.com/ZhouYuxuanYX/Hyperformer/releases
2. Download: `hyperformer_pretrained_weights.zip`
3. Extract the weights for different modalities:
   - **Joint modality**: `Hyperformer_ntu60_xsub_joint.pth`
   - **Bone modality**: `Hyperformer_ntu60_xsub_bone.pth`
   - **Joint-Vel modality**: `Hyperformer_ntu60_xsub_vel.pth`
   - **Bone-Vel modality**: `Hyperformer_ntu60_xsub_bone_vel.pth`
   - Similar files for NTU 120 and Cross-View benchmarks

**Direct Download Link**:
```
https://github.com/ZhouYuxuanYX/Hyperformer/releases/download/pretrained_weights/hyperformer_pretrained_weights.zip
```

**Keypoint Configuration**:
- **Original**: NTU RGB+D (25 joints)
- **Target**: COCO (17 keypoints)
- **Adaptation Strategy**:
  - Modify the Hyperformer instantiation to use `num_point=17`
  - Update graph configuration to use COCO skeleton topology
  - Load pretrained weights with partial loading (ignore shape mismatches)

### Usage in Your Code
```python
from model.hyperformer import Hyperformer

# Load the official Hyperformer model
model = Hyperformer(
    num_class=10,       # Your action classes
    num_point=17,       # COCO keypoints (was 25 for NTU)
    num_person=1,       # Number of people in scene
    graph='coco',       # Use COCO graph instead of NTU
    in_channels=2       # (x, y) coordinates
)

# Load pretrained with custom adapter
from models.sht import load_pretrained
model = load_pretrained(model, 'path/to/Hyperformer_ntu60_xsub_joint.pth')
```

**Important Notes**:
- Hyperformer has been verified in your codebase at `Hyperformer/` directory
- The wrapper in `models/sht.py` adapts Hyperformer to 12 joints (modify for 17)
- You may need to adjust positional encodings and graph distance embeddings

---

## Keypoint Adaptation Strategy

Since all pretrained models use **NTU RGB+D (25 joints)** and you're targeting **COCO (17 keypoints)**, here's the recommended approach:

### 1. Partial Weight Loading (Recommended)
```python
def _load_pretrained_weights(self, path):
    print(f"[INFO] Loading pretrained weights from {path} ...")
    pretrained_dict = torch.load(path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['model_state_dict']
    
    model_dict = self.state_dict()
    
    # Filter out:
    # 1. Mismatched shapes
    # 2. Graph adjacency matrices
    # 3. Joint-specific embeddings
    filtered_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                filtered_dict[k] = v
            else:
                print(f"[SKIP] Shape mismatch for {k}: "
                      f"pretrained {v.shape} vs model {model_dict[k].shape}")
        else:
            print(f"[SKIP] Key not in model: {k}")
    
    model_dict.update(filtered_dict)
    self.load_state_dict(model_dict)
    print(f"[INFO] Loaded {len(filtered_dict)}/{len(pretrained_dict)} layers")
```

### 2. What Transfers Well
✅ **Transferable Components**:
- Temporal convolution layers (TCN)
- Batch normalization layers
- Feature extraction conv layers
- Pooling mechanisms
- Early fusion layers

❌ **Non-Transferable Components**:
- Graph adjacency matrices (different skeleton topology)
- Joint-specific positional embeddings
- Any layers with hardcoded joint counts (25 → 17)

### 3. Joint Mapping (Optional Advanced Strategy)
If you want to map specific NTU joints to COCO joints:

| NTU RGB+D (25) | COCO (17) | Mapping Strategy |
|----------------|-----------|------------------|
| Spine base → Hip center | Average left/right hip |
| Shoulder center → Shoulders | Average left/right shoulder |
| Wrists, Elbows, Shoulders | Direct mapping | 1:1 correspondence |
| Hands → Wrists | Use wrist as proxy | |
| Spine mid/shoulder → Discard | N/A | No direct equivalent |

---

## Quick Start Command Summary

### Download All Pretrained Weights

```bash
# Create pretrained weights directory
mkdir -p pretrained_weights
cd pretrained_weights

# ST-GCN
git clone https://github.com/yysijie/st-gcn
# Follow their README to download specific checkpoints

# CTR-GCN
git clone https://github.com/Uason-Chen/CTR-GCN
# Check releases or work_dir/ for .pth files

# Hyperformer
wget https://github.com/ZhouYuxuanYX/Hyperformer/releases/download/pretrained_weights/hyperformer_pretrained_weights.zip
unzip hyperformer_pretrained_weights.zip
```

---

## Testing Pretrained Models

### Verification Script
```python
import torch
from models.stgcn import EnhancedSTGCN
from models.ctrgcn import EnhancedCTRGCN

# Test ST-GCN
print("Testing ST-GCN...")
stgcn = EnhancedSTGCN(
    in_channels=2,
    num_joints=17,
    num_classes=10,
    pretrained_path='pretrained_weights/stgcn_ntu60.pth'
)
x = torch.randn(2, 2, 64, 17, 1)  # Batch=2, C=2, T=64, V=17
out = stgcn(x)
print(f"ST-GCN output shape: {out.shape}")

# Test CTR-GCN
print("\nTesting CTR-GCN...")
ctrgcn = EnhancedCTRGCN(
    in_channels=2,
    num_joints=17,
    num_classes=10,
    pretrained_path='pretrained_weights/ctrgcn_ntu60_xsub.pth'
)
out = ctrgcn(x)
print(f"CTR-GCN output shape: {out.shape}")

print("\n✅ All models loaded successfully!")
```

---

## Recommended Priority

Based on **availability**, **performance**, and **17-keypoint compatibility**:

1. **CTR-GCN** (Highest Priority)
   - ✅ State-of-the-art performance (ICCV 2021)
   - ✅ Well-documented pretrained weights
   - ✅ Excellent transfer learning capability
   - ✅ Your code already handles adaptation

2. **Hyperformer** (High Priority)
   - ✅ Latest architecture with attention mechanisms
   - ✅ Pretrained weights readily available
   - ✅ Superior performance on NTU benchmarks
   - ⚠️ Requires more adaptation for 17 keypoints

3. **ST-GCN** (Medium Priority)
   - ✅ Classic baseline model
   - ✅ Stable and well-tested
   - ⚠️ Older architecture (AAAI 2018)
   - ✅ Good for comparison experiments

4. **TE-GCN** (Low Priority)
   - ⚠️ Limited pretrained weight availability
   - ⚠️ Custom implementation
   - ✅ Can use ST-GCN weights as initialization
   - 💡 Consider training from scratch

---

## Additional Resources

### Official Model Zoos
- **MMAction2**: https://github.com/open-mmlab/mmaction2
  - Contains implementations and pretrained weights for ST-GCN, CTR-GCN, and more
  - Easier integration if you're already using MMPose

### Papers with Code
- **CTR-GCN**: https://paperswithcode.com/paper/channel-wise-topology-refinement-graph
- **Hyperformer**: https://paperswithcode.com/paper/hypergraph-transformer-for-skeleton-based

### Community Implementations
- Search GitHub for: `skeleton action recognition pretrained`
- Check Issues sections of official repos for download links

---

## Troubleshooting

### Issue: Shape Mismatch When Loading Weights
**Solution**: Use partial loading (implemented in your models)

### Issue: Graph Adjacency Not Matching
**Solution**: Skip loading adjacency matrices, use your custom 17-keypoint graph

### Issue: Missing Keys in State Dict
**Solution**: Load only the keys that exist in both dicts (implemented)

### Issue: NTU 25 joints → COCO 17 joints
**Solution**: 
- Don't map individual joints
- Transfer only topology-agnostic layers (TCN, FC, BN)
- Train graph layers from scratch on your 17-keypoint data

---

## Contact & Support

For model-specific issues:
- **ST-GCN**: https://github.com/yysijie/st-gcn/issues
- **CTR-GCN**: https://github.com/Uason-Chen/CTR-GCN/issues
- **Hyperformer**: zhouyuxuanyx@gmail.com

---

## Last Updated
January 26, 2026
