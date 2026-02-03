# Model Summary - Pretrained Weights for 17 Keypoints

## Executive Summary

You have **5 models** in your `models/` directory. Here's what you need to know about pretrained weights for each:

---

## Quick Comparison Table

| Model | Priority | Pretrained Available | Original Keypoints | Adaptation to 17 | Download Difficulty |
|-------|----------|---------------------|-------------------|------------------|-------------------|
| **CTR-GCN** | 🥇 **HIGHEST** | ✅ Yes | NTU (25) | ⭐⭐⭐⭐ Excellent | 🟢 Easy |
| **CTR-GCN Motion** | 🥈 High | ✅ Yes (same as above) | NTU (25) | ⭐⭐⭐⭐ Excellent | 🟢 Easy |
| **Hyperformer** | 🥈 High | ✅ Yes | NTU (25) | ⭐⭐⭐⭐ Excellent | 🟢 Easy (direct link) |
| **ST-GCN** | 🥉 Medium | ✅ Yes | NTU (25) | ⭐⭐⭐ Good | 🟡 Moderate |
| **TE-GCN** | ⚠️ Low | ❌ No | Custom | N/A | 🔴 Train from scratch |

---

## Detailed Model Information

### 1. CTR-GCN ⭐ BEST CHOICE

**File**: `models/ctrgcn.py`

**Why Choose This**:
- ✅ State-of-the-art (ICCV 2021)
- ✅ Your code already implements weight loading with 17-keypoint adaptation
- ✅ Channel-wise topology refinement works excellently for transfer learning
- ✅ Well-documented pretrained weights available

**Pretrained Source**: 
```
Repository: https://github.com/Uason-Chen/CTR-GCN
Weights: Check README for Google Drive or GitHub releases
Files: ntu60_xsub_ctrgcn.pt, ntu60_xview_ctrgcn.pt
```

**Expected Performance**: Best transfer learning results from 25→17 keypoints

**Code Usage**:
```python
from models.ctrgcn import EnhancedCTRGCN
model = EnhancedCTRGCN(
    in_channels=2,
    num_joints=17,
    num_classes=10,
    pretrained_path='pretrained_weights/ctrgcn_ntu60.pt'
)
```

---

### 2. CTR-GCN with Motion Stream ⭐ ADVANCED

**File**: `models/ctrgcn_motion.py`

**Why Choose This**:
- ✅ Dual-stream architecture (spatial + temporal motion)
- ✅ Better at capturing dynamic actions
- ✅ Uses same pretrained weights as CTR-GCN
- ✅ Automatically computes motion deltas

**Pretrained Source**: 
```
Same as CTR-GCN above
Use the same checkpoint for both branches initially
```

**Expected Performance**: Superior for action recognition with movement

**Code Usage**:
```python
from models.ctrgcn_motion import EnhancedCTRGCN_Motion
model = EnhancedCTRGCN_Motion(
    in_channels=2,
    num_joints=17,
    num_classes=10,
    pretrained_spatial='pretrained_weights/ctrgcn_ntu60.pt',
    pretrained_motion='pretrained_weights/ctrgcn_ntu60.pt'
)
```

---

### 3. Hyperformer 🚀 CUTTING EDGE

**File**: `models/sht.py` (wrapper for `Hyperformer/model/Hyperformer.py`)

**Why Choose This**:
- ✅ Most recent architecture (2022)
- ✅ Hypergraph attention mechanism
- ✅ Pretrained weights with direct download link
- ✅ Superior performance on NTU benchmarks (92.9% accuracy)
- ⚠️ More complex to adapt to 17 keypoints

**Pretrained Source**:
```
Repository: https://github.com/ZhouYuxuanYX/Hyperformer
Direct Download: 
wget https://github.com/ZhouYuxuanYX/Hyperformer/releases/download/pretrained_weights/hyperformer_pretrained_weights.zip

Files:
- Hyperformer_ntu60_xsub_joint.pth
- Hyperformer_ntu60_xsub_bone.pth
- Hyperformer_ntu60_xsub_vel.pth
- Hyperformer_ntu60_xsub_bone_vel.pth
```

**Expected Performance**: Highest potential, but requires careful adaptation

**Code Usage**:
```python
from model.hyperformer import Hyperformer
from models.sht import load_pretrained

model = Hyperformer(
    num_class=10,
    num_point=17,  # Adjust from 25 to 17
    num_person=1,
    graph='coco',
    in_channels=2
)
model = load_pretrained(model, 'pretrained_weights/Hyperformer_ntu60_xsub_joint.pth')
```

---

### 4. ST-GCN 📊 RELIABLE BASELINE

**File**: `models/stgcn.py`

**Why Choose This**:
- ✅ Classic baseline (AAAI 2018)
- ✅ Well-tested and stable
- ✅ Good for comparison experiments
- ✅ Simpler architecture, easier to debug
- ⚠️ Older, may have lower accuracy than CTR-GCN/Hyperformer

**Pretrained Source**:
```
Repository: https://github.com/yysijie/st-gcn
Files: st_gcn_ntu60_xsub.pt or st_gcn.kinetics-6fa43f73.pt

Alternative (OpenMMLab):
wget https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint/stgcn_80e_ntu60_xsub_keypoint_20200826-e8b0f4b5.pth
```

**Expected Performance**: Solid baseline, good for sanity checks

**Code Usage**:
```python
from models.stgcn import EnhancedSTGCN
model = EnhancedSTGCN(
    in_channels=2,
    num_joints=17,
    num_classes=10,
    pretrained_path='pretrained_weights/stgcn_ntu60.pt'
)
```

---

### 5. TE-GCN ⚠️ EXPERIMENTAL

**File**: `models/tegcn.py`

**Why Choose This**:
- ⚠️ Custom implementation (Taylor expansion-based)
- ⚠️ No widely available pretrained weights
- ✅ Can use ST-GCN weights for partial initialization
- ✅ Lightweight and efficient

**Pretrained Source**:
```
Status: No public pretrained weights found
Options:
1. Train from scratch on your data
2. Use ST-GCN weights for initialization (partial loading)
```

**Expected Performance**: Experimental, unknown transfer learning capability

**Code Usage**:
```python
from models.tegcn import TE_GCN

# Option 1: Train from scratch
model = TE_GCN(
    in_channels=2,
    num_joints=17,
    num_classes=10,
    pretrained_path=None
)

# Option 2: Initialize with ST-GCN
model = TE_GCN(
    in_channels=2,
    num_joints=17,
    num_classes=10,
    pretrained_path='pretrained_weights/stgcn_ntu60.pt'  # Partial loading
)
```

---

## Recommended Implementation Strategy

### Phase 1: Start with CTR-GCN
```bash
# Download CTR-GCN pretrained weights
cd pretrained_weights
git clone https://github.com/Uason-Chen/CTR-GCN.git
# Follow their README to get .pt files

# Test loading
python test_pretrained.py

# Train/fine-tune on your 17-keypoint data
python train.py --model ctrgcn --pretrained pretrained_weights/CTR-GCN/ntu60_xsub_ctrgcn.pt
```

**Expected outcome**: Should see faster convergence and better accuracy than random initialization

### Phase 2: Experiment with Hyperformer
```bash
# Download Hyperformer weights
cd pretrained_weights
wget https://github.com/ZhouYuxuanYX/Hyperformer/releases/download/pretrained_weights/hyperformer_pretrained_weights.zip
unzip hyperformer_pretrained_weights.zip

# Adapt your code for Hyperformer
# Test and train
```

**Expected outcome**: Potentially higher accuracy, but may require more tuning

### Phase 3: Baseline with ST-GCN
```bash
# Use as comparison baseline
python train.py --model stgcn --pretrained pretrained_weights/stgcn_ntu60.pt
```

**Expected outcome**: Solid baseline to compare against CTR-GCN/Hyperformer

### Phase 4 (Optional): TE-GCN from Scratch
```bash
# Train without pretrained weights
python train.py --model tegcn
```

**Expected outcome**: Might converge slower but could be optimized for your specific 17-keypoint task

---

## Key Insights: 25 Keypoints → 17 Keypoints

### What Transfers Successfully ✅
1. **Temporal Convolution Layers (TCN)**
   - Captures temporal dynamics independent of joint count
   - Fully transferable

2. **Batch Normalization Layers**
   - Statistics adapt during fine-tuning
   - Fully transferable

3. **Feature Extraction Convolutions**
   - Channel-wise operations
   - Fully transferable

4. **Pooling and FC Layers**
   - Global aggregation mechanisms
   - Fully transferable

### What Does NOT Transfer ❌
1. **Graph Adjacency Matrices**
   - Different skeleton topology (NTU vs COCO)
   - Must be reinitialized

2. **Joint-specific Embeddings**
   - Hardcoded for 25 joints
   - Automatically handled by your models (skipped during loading)

### Transfer Learning Results
Based on similar literature:
- **Random Init**: ~70-75% accuracy after full training
- **Pretrained (25→17)**: ~80-85% accuracy with same training
- **Speedup**: 2-3x faster convergence

---

## File Organization

After downloading, your structure should be:

```
MMPose-Lib/
├── models/
│   ├── ctrgcn.py                 ← EnhancedCTRGCN class
│   ├── ctrgcn_motion.py          ← EnhancedCTRGCN_Motion class
│   ├── stgcn.py                  ← EnhancedSTGCN class
│   ├── tegcn.py                  ← TE_GCN class
│   ├── sht.py                    ← Hyperformer wrapper
│   ├── PRETRAINED_WEIGHTS_GUIDE.md
│   ├── QUICK_DOWNLOAD_REFERENCE.md
│   └── MODEL_SUMMARY.md          ← This file
│
├── Hyperformer/
│   └── model/
│       └── Hyperformer.py        ← Actual Hyperformer implementation
│
└── pretrained_weights/           ← Create this
    ├── CTR-GCN/
    │   └── ntu60_xsub_ctrgcn.pt
    ├── st-gcn/
    │   └── st_gcn_ntu60_xsub.pt
    └── hyperformer/
        ├── Hyperformer_ntu60_xsub_joint.pth
        └── ...
```

---

## Performance Expectations

### NTU RGB+D 60 (Original Benchmarks)
| Model | Original Paper Accuracy | Expected Transfer to COCO-17 |
|-------|------------------------|------------------------------|
| ST-GCN | 88.3% | ~80-85% (after fine-tuning) |
| CTR-GCN | 92.4% | ~84-88% (after fine-tuning) |
| Hyperformer | 92.9% | ~85-89% (after fine-tuning) |
| TE-GCN | N/A | ~75-80% (train from scratch) |

*Note: Actual performance depends on your specific action recognition task and dataset quality*

---

## Next Steps

1. **Download CTR-GCN weights** (highest priority)
   ```bash
   git clone https://github.com/Uason-Chen/CTR-GCN.git pretrained_weights/CTR-GCN
   ```

2. **Create test script** to verify loading
   ```bash
   python test_pretrained.py
   ```

3. **Fine-tune on your data**
   ```bash
   python train.py --model ctrgcn --epochs 50 --lr 0.001
   ```

4. **Compare with baseline**
   ```bash
   python train.py --model stgcn --epochs 50 --lr 0.001
   ```

5. **Experiment with Hyperformer** (if you have compute resources)

---

## FAQ

**Q: Will 25-keypoint pretrained weights work for 17 keypoints?**  
A: Yes! Your models already implement partial weight loading. Temporal and feature layers transfer perfectly.

**Q: Which model should I use first?**  
A: Start with CTR-GCN. It has the best balance of performance, availability, and ease of use.

**Q: Do I need to fine-tune?**  
A: Yes, absolutely. The graph structure is different, so fine-tuning is required.

**Q: How long to fine-tune?**  
A: Typically 20-50 epochs. Watch for convergence.

**Q: Can I use multiple pretrained models together?**  
A: Yes! Ensemble different models for better results.

---

## Additional Resources

### Papers
- **ST-GCN**: "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition" (AAAI 2018)
- **CTR-GCN**: "Channel-Wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition" (ICCV 2021)
- **Hyperformer**: "Hypergraph Transformer for Skeleton-based Action Recognition" (arXiv 2022)

### Code Repositories
- CTR-GCN: https://github.com/Uason-Chen/CTR-GCN
- ST-GCN: https://github.com/yysijie/st-gcn
- Hyperformer: https://github.com/ZhouYuxuanYX/Hyperformer

### Model Zoos
- MMAction2: https://github.com/open-mmlab/mmaction2/tree/main/configs/skeleton
- Papers with Code: https://paperswithcode.com/task/skeleton-based-action-recognition

---

Last Updated: January 26, 2026
