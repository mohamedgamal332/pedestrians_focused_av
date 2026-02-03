import os
import torch
import torch.nn as nn
import numpy as np

# add repo to path
import sys
sys.path.append(os.path.abspath("."))

try:
    from model.hyperformer import Hyperformer  # official model definition
    HAS_HYPERFORMER = True
except ImportError:
    HAS_HYPERFORMER = False
    print("Warning: Hyperformer model not found. SHT model will not work.")

from torchlight.utils import import_class

# ----------------------------
# Custom Hyperformer Loader
# ----------------------------

class Hyperformer12Joints(nn.Module):
    def __init__(self, base_model: Hyperformer, num_joints: int = 12):
        super().__init__()
        self.model = base_model

        # override joint embedding if needed
        # some official configs use spatial positional encoding based on skeleton topology;
        # you can adjust here if original Hyperformer uses distance graph embedding

        self.model.num_node = num_joints  # override internal nodes

    def forward(self, x):
        # x: N, C, T, V (no bone modality here)
        return self.model(x)

# ----------------------------
# Standard Interface for SHT/Hyperformer
# ----------------------------

class SHT_Hyperformer(nn.Module):
    """
    Standard interface wrapper for Hyperformer (SHT) model.
    Matches the interface of EnhancedSTGCN, EnhancedCTRGCN, etc.
    """
    def __init__(self, in_channels=2, num_joints=17, num_classes=5, 
                 pretrained_path=None, graph='coco'):
        super().__init__()
        if not HAS_HYPERFORMER:
            raise ImportError("Hyperformer model not available. Please install Hyperformer dependencies.")
        
        # Create base Hyperformer model
        self.base_model = Hyperformer(
            num_class=num_classes,
            num_point=num_joints,
            num_person=1,
            graph=graph,
            in_channels=in_channels
        )
        
        # Wrap for joint adaptation
        self.model = Hyperformer12Joints(self.base_model, num_joints=num_joints)
        
        # Load pretrained weights if provided
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)
    
    def _load_pretrained(self, path):
        """Load pretrained weights using the load_pretrained function"""
        if os.path.exists(path):
            self.model = load_pretrained(self.model, path)
        else:
            print(f"Warning: Pretrained weights not found at {path}, using random initialization")
    
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: [N, C, T, V] tensor
        Returns:
            [N, num_classes] logits
        """
        # Handle 5D input if needed
        if x.dim() == 5:
            x = x.squeeze(-1)
        return self.model(x)

# ----------------------------
# Load Pretrained Hyperformer
# ----------------------------

def load_pretrained(model, path):
    print(f"Loading pretrained: {path}")
    ckpt = torch.load(path, map_location='cpu')
    # official Hyperformer checkpoints might store under key 'model_state_dict'
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    model_dict = model.state_dict()

    # filter keys
    new_dict = {}
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            new_dict[k] = v
        else:
            # missing OR mismatched shape
            print(f"skip {k} -> {v.shape if hasattr(v,'shape') else None}")

    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    return model

# ----------------------------
# Entry point
# ----------------------------

if __name__ == "__main__":
    # Choose input modality
    # Usually joint only: Hyperformer_ntu60_xsub_joint.pth
    # Or bone modality: Hyperformer_ntu60_xsub_bone.pth
    pretrained_joint = "Hyperformer_ntu60_xsub_joint.pth"
    pretrained_bone  = "Hyperformer_ntu60_xsub_bone.pth"

    # Temporarily use joint
    model = Hyperformer(
        num_class=60,   # NTU60 num classes
        num_point=12,   # override for 12 joints
        num_person=1,   # number of skeletons
        graph='coco'    # if you use COCO graph – adjust if needed
    )

    model = Hyperformer12Joints(model, num_joints=12)

    # Load weights
    model = load_pretrained(model, pretrained_joint)
    model.eval()

