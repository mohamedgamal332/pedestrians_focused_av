# Models Directory - Pretrained Weights Documentation

This directory contains implementations of Graph Convolutional Network (GCN) models for skeleton-based action recognition, along with comprehensive documentation for obtaining and using pretrained weights.

## 📚 Documentation Index

| Document | Purpose | Best For |
|----------|---------|----------|
| **[MODEL_SUMMARY.md](MODEL_SUMMARY.md)** | Executive summary and comparison | Quick overview and model selection |
| **[PRETRAINED_WEIGHTS_GUIDE.md](PRETRAINED_WEIGHTS_GUIDE.md)** | Complete guide with detailed instructions | Deep dive into each model |
| **[QUICK_DOWNLOAD_REFERENCE.md](QUICK_DOWNLOAD_REFERENCE.md)** | Direct commands and quick tips | Getting started fast |
| **[README.md](README.md)** | This file | Navigation and quick start |

## 🚀 Quick Start

### 1. Download Pretrained Weights
```bash
# Automatic download (recommended)
python models/download_pretrained.py --model all

# Or download specific models
python models/download_pretrained.py --model hyperformer
python models/download_pretrained.py --model ctrgcn
```

### 2. Test Model Loading
```bash
# Test all models
python models/test_pretrained.py

# Test specific model
python models/test_pretrained.py --model ctrgcn --weights pretrained_weights/CTR-GCN/ntu60_xsub.pt
```

### 3. Use in Your Code
```python
from models.ctrgcn import EnhancedCTRGCN

# Load with pretrained weights
model = EnhancedCTRGCN(
    in_channels=2,
    num_joints=17,  # COCO keypoints
    num_classes=10,
    pretrained_path='pretrained_weights/CTR-GCN/ntu60_xsub_ctrgcn.pt'
)

# Or without pretrained (random initialization)
model = EnhancedCTRGCN(
    in_channels=2,
    num_joints=17,
    num_classes=10
)
```

## 📦 Available Models

| Model | File | Priority | Pretrained Available |
|-------|------|----------|---------------------|
| **CTR-GCN** | `ctrgcn.py` | 🥇 Highest | ✅ Yes |
| **CTR-GCN Motion** | `ctrgcn_motion.py` | 🥈 High | ✅ Yes |
| **Hyperformer** | `sht.py` | 🥈 High | ✅ Yes (direct download) |
| **ST-GCN** | `stgcn.py` | 🥉 Medium | ✅ Yes |
| **TE-GCN** | `tegcn.py` | ⚠️ Low | ❌ No (train from scratch) |

### Model Details

#### CTR-GCN (Recommended ⭐)
- **Paper**: Channel-Wise Topology Refinement Graph Convolution (ICCV 2021)
- **Why**: Best transfer learning from 25→17 keypoints
- **Pretrained**: NTU RGB+D 60/120
- **Usage**: See [QUICK_DOWNLOAD_REFERENCE.md](QUICK_DOWNLOAD_REFERENCE.md#1-ctr-gcn--recommended)

#### Hyperformer (Latest 🚀)
- **Paper**: Hypergraph Transformer for Skeleton-based Action Recognition (2022)
- **Why**: Most advanced architecture with attention mechanisms
- **Pretrained**: Direct download available
- **Usage**: See [QUICK_DOWNLOAD_REFERENCE.md](QUICK_DOWNLOAD_REFERENCE.md#2-hyperformer--newest)

#### ST-GCN (Baseline 📊)
- **Paper**: Spatial Temporal Graph Convolutional Networks (AAAI 2018)
- **Why**: Solid baseline for comparison
- **Pretrained**: NTU RGB+D, Kinetics
- **Usage**: See [QUICK_DOWNLOAD_REFERENCE.md](QUICK_DOWNLOAD_REFERENCE.md#3-st-gcn--baseline)

## 🎯 Recommended Workflow

### For New Users
1. **Read** [MODEL_SUMMARY.md](MODEL_SUMMARY.md) for overview
2. **Download** pretrained weights: `python models/download_pretrained.py --model all`
3. **Test** loading: `python models/test_pretrained.py`
4. **Start** with CTR-GCN for best results

### For Experienced Users
1. **Check** [QUICK_DOWNLOAD_REFERENCE.md](QUICK_DOWNLOAD_REFERENCE.md) for direct commands
2. **Download** specific models you need
3. **Fine-tune** on your dataset
4. **Compare** results across different models

### For Researchers
1. **Read** [PRETRAINED_WEIGHTS_GUIDE.md](PRETRAINED_WEIGHTS_GUIDE.md) for detailed architecture info
2. **Experiment** with different pretrained sources
3. **Adapt** models to your specific keypoint configuration
4. **Ensemble** multiple models for better performance

## 🔑 Key Features

### 17-Keypoint Adaptation
All models support loading pretrained weights trained on NTU RGB+D (25 joints) and adapting to COCO (17 keypoints):

```python
# The adaptation is handled automatically
model = EnhancedCTRGCN(
    num_joints=17,  # Your target keypoints
    pretrained_path='path/to/ntu60_25joints.pt'  # Trained on 25 joints
)
# Only compatible layers are loaded, graph structure is reinitialized
```

### What Transfers
✅ Temporal convolution layers  
✅ Feature extraction layers  
✅ Batch normalization  
✅ Pooling and FC layers  

### What Doesn't Transfer
❌ Graph adjacency matrices (different topology)  
❌ Joint-specific embeddings (different count)  

## 📖 Detailed Documentation

### MODEL_SUMMARY.md
- Executive summary of all models
- Performance expectations
- Comparison table
- Recommended implementation strategy
- **Best for**: Quick decision-making

### PRETRAINED_WEIGHTS_GUIDE.md
- Complete guide for each model
- Download instructions with URLs
- Adaptation strategies
- Usage examples
- Troubleshooting
- **Best for**: Comprehensive understanding

### QUICK_DOWNLOAD_REFERENCE.md
- Direct download commands
- Quick setup instructions
- One-line usage examples
- Verification scripts
- **Best for**: Getting started immediately

## 🛠️ Utility Scripts

### download_pretrained.py
Automated downloader for pretrained weights.

```bash
# Download all models
python models/download_pretrained.py --model all

# Download specific model
python models/download_pretrained.py --model hyperformer
python models/download_pretrained.py --model ctrgcn
python models/download_pretrained.py --model stgcn
```

### test_pretrained.py
Test that models load correctly and run forward/backward passes.

```bash
# Test all models
python models/test_pretrained.py

# Test specific model
python models/test_pretrained.py --model ctrgcn

# Test with specific weights
python models/test_pretrained.py --model ctrgcn --weights path/to/weights.pt

# Test with GPU
python models/test_pretrained.py --device cuda
```

## 📁 Directory Structure

After downloading pretrained weights:

```
models/
├── README.md                          ← This file
├── MODEL_SUMMARY.md                   ← Executive summary
├── PRETRAINED_WEIGHTS_GUIDE.md        ← Complete guide
├── QUICK_DOWNLOAD_REFERENCE.md        ← Quick commands
│
├── download_pretrained.py             ← Auto-downloader
├── test_pretrained.py                 ← Testing script
│
├── ctrgcn.py                          ← CTR-GCN model
├── ctrgcn_motion.py                   ← CTR-GCN with motion stream
├── stgcn.py                           ← ST-GCN model
├── tegcn.py                           ← TE-GCN model
└── sht.py                             ← Hyperformer wrapper

../pretrained_weights/                 ← Downloaded weights (created by you)
├── CTR-GCN/
│   └── ntu60_xsub_ctrgcn.pt
├── st-gcn/
│   └── stgcn_80e_ntu60_xsub_keypoint.pth
└── hyperformer/
    ├── Hyperformer_ntu60_xsub_joint.pth
    └── ...
```

## 🔍 Finding the Right Model

### Use CTR-GCN if:
- ✅ You want best transfer learning results
- ✅ You're working with 17 keypoints (COCO)
- ✅ You need state-of-the-art performance
- ✅ You want well-documented pretrained weights

### Use Hyperformer if:
- ✅ You want the latest architecture
- ✅ You have sufficient compute resources
- ✅ You're willing to experiment with adaptation
- ✅ You want attention mechanisms

### Use ST-GCN if:
- ✅ You need a reliable baseline
- ✅ You want to compare against classic models
- ✅ You prefer simpler architectures
- ✅ You're doing ablation studies

### Use TE-GCN if:
- ✅ You're doing research on Taylor expansions
- ✅ You can train from scratch
- ✅ You want a lightweight model
- ✅ You're comparing different graph convolutions

## ⚙️ Configuration Examples

### Basic Usage
```python
from models.ctrgcn import EnhancedCTRGCN

model = EnhancedCTRGCN(
    in_channels=2,        # (x, y) coordinates
    num_joints=17,        # COCO keypoints
    num_classes=10,       # Your action classes
    pretrained_path='pretrained_weights/CTR-GCN/ntu60_xsub_ctrgcn.pt'
)
```

### With Motion Stream
```python
from models.ctrgcn_motion import EnhancedCTRGCN_Motion

model = EnhancedCTRGCN_Motion(
    in_channels=2,
    num_joints=17,
    num_classes=10,
    pretrained_spatial='pretrained_weights/CTR-GCN/ntu60_xsub_ctrgcn.pt',
    pretrained_motion='pretrained_weights/CTR-GCN/ntu60_xsub_ctrgcn.pt'
)
```

### Custom Graph
```python
from graphs.custom_coco_graph import adjacency_matrix
from models.ctrgcn import EnhancedCTRGCN

A = adjacency_matrix()  # Your custom 17-keypoint graph
model = EnhancedCTRGCN(
    in_channels=2,
    num_joints=17,
    num_classes=10,
    A_norm=A,
    pretrained_path='pretrained_weights/CTR-GCN/ntu60_xsub_ctrgcn.pt'
)
```

## 🐛 Troubleshooting

### Issue: "FileNotFoundError: pretrained weights not found"
**Solution**: Run `python models/download_pretrained.py` first

### Issue: "Shape mismatch when loading weights"
**Solution**: This is normal! Your models automatically skip incompatible layers

### Issue: "CUDA out of memory"
**Solution**: Load to CPU first, then move to GPU:
```python
model = EnhancedCTRGCN(..., pretrained_path='...')
model = model.to('cuda')
```

### Issue: "Git not found"
**Solution**: 
- Install git: https://git-scm.com/downloads
- Or manually download repositories from GitHub

### Issue: "Models not improving with pretrained weights"
**Solution**: 
- Fine-tune for 20-50 epochs
- Use lower learning rate for pretrained layers
- Graph structure is different (25 vs 17), needs adaptation time

## 📚 Additional Resources

### Official Repositories
- **CTR-GCN**: https://github.com/Uason-Chen/CTR-GCN
- **ST-GCN**: https://github.com/yysijie/st-gcn
- **Hyperformer**: https://github.com/ZhouYuxuanYX/Hyperformer

### Model Zoos
- **MMAction2**: https://github.com/open-mmlab/mmaction2
- **Papers with Code**: https://paperswithcode.com/task/skeleton-based-action-recognition

### Related Projects
- **MMPose**: https://github.com/open-mmlab/mmpose
- **MMDetection**: https://github.com/open-mmlab/mmdetection

## 🤝 Contributing

If you find issues or have improvements:
1. Check existing documentation first
2. Test your changes with `test_pretrained.py`
3. Update relevant documentation files
4. Submit detailed bug reports with model configs

## 📞 Support

For model-specific issues, contact:
- **CTR-GCN**: https://github.com/Uason-Chen/CTR-GCN/issues
- **ST-GCN**: https://github.com/yysijie/st-gcn/issues
- **Hyperformer**: zhouyuxuanyx@gmail.com

For general GCN questions:
- Check documentation in this directory
- Search existing GitHub issues
- Review papers for architecture details

## 📝 Citation

If you use these models, please cite the original papers:

**CTR-GCN**:
```bibtex
@inproceedings{chen2021channel,
  title={Channel-wise topology refinement graph convolution for skeleton-based action recognition},
  author={Chen, Yuxin and Zhang, Ziqi and Yuan, Chunfeng and Li, Bing and Deng, Ying and Hu, Weiming},
  booktitle={ICCV},
  year={2021}
}
```

**ST-GCN**:
```bibtex
@inproceedings{yan2018spatial,
  title={Spatial temporal graph convolutional networks for skeleton-based action recognition},
  author={Yan, Sijie and Xiong, Yuanjun and Lin, Dahua},
  booktitle={AAAI},
  year={2018}
}
```

**Hyperformer**:
```bibtex
@article{zhou2022hypergraph,
  title={Hypergraph Transformer for Skeleton-based Action Recognition},
  author={Zhou, Yuxuan and Cheng, Zhi-Qi and Li, Chao and Geng, Yifeng and Xie, Xuansong and Keuper, Margret},
  journal={arXiv preprint arXiv:2211.09590},
  year={2022}
}
```

---

## 🎉 Getting Started Checklist

- [ ] Read [MODEL_SUMMARY.md](MODEL_SUMMARY.md) for overview
- [ ] Choose your model (CTR-GCN recommended)
- [ ] Run `python models/download_pretrained.py --model all`
- [ ] Test with `python models/test_pretrained.py`
- [ ] Integrate into your training pipeline
- [ ] Fine-tune on your 17-keypoint data
- [ ] Compare results across models
- [ ] Enjoy state-of-the-art skeleton action recognition! 🎊

---

**Last Updated**: January 26, 2026  
**Documentation Version**: 1.0  
**Maintained by**: MMPose-Lib Contributors
