import os
import torch
import numpy as np

# add repo to path
import sys
sys.path.append(os.path.abspath("."))

from model.hyperformer import Hyperformer  # official model definition
from torchlight.utils import import_class

# ----------------------------
# Custom Hyperformer Loader
# ----------------------------

class Hyperformer12Joints(torch.nn.Module):
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

