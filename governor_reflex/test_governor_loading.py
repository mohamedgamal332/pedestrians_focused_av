#!/usr/bin/env python3
"""
Test Alpamayo model loading.
Run from alpo environment: conda activate alpo && python test_governor_loading.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_alpamayo_import():
    """Test alpamayo_r1 package import."""
    print("Testing alpamayo_r1 import...")
    
    try:
        import alpamayo_r1
        print(f"  ✓ alpamayo_r1 imported from: {alpamayo_r1.__file__}")
    except ImportError as e:
        print(f"  ✗ Failed to import alpamayo_r1: {e}")
        return False
    
    return True


def test_torch_cuda():
    """Test PyTorch and CUDA."""
    print("Testing PyTorch and CUDA...")
    
    import torch
    
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("  ✓ CUDA working")
    else:
        print("  ⚠ CUDA not available")
    
    return torch.cuda.is_available()


def test_model_loading():
    """Test loading the Alpamayo model."""
    print("Testing Alpamayo model loading...")
    
    model_path = "/home/jemmi/trajectory-system/models/alpamayo"
    
    if not Path(model_path).exists():
        print(f"  ✗ Model path not found: {model_path}")
        return False
    
    print(f"  Model path: {model_path}")
    
    # List model files
    files = list(Path(model_path).glob("*"))
    print(f"  Found {len(files)} files")
    
    try:
        from governor.alpamayo_wrapper import AlpamayoWrapper
        
        print("  Creating wrapper (INT8 quantization)...")
        wrapper = AlpamayoWrapper(
            model_path=model_path,
            quantization="int8",
            device="cuda"
        )
        
        print("  Loading model (this may take a few minutes)...")
        wrapper.load()
        
        if wrapper.is_loaded():
            print("  ✓ Model loaded successfully!")
            
            mem = wrapper.get_memory_usage()
            if mem:
                print(f"  GPU Memory used: {mem['allocated_gb']:.2f} GB")
            
            return True
        else:
            print("  ✗ Model failed to load")
            return False
            
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference():
    """Test model inference (requires successful loading)."""
    print("Testing inference...")
    
    try:
        import torch
        import numpy as np
        from governor.alpamayo_wrapper import AlpamayoWrapper
        from governor.trajectory_decoder import TrajectoryDecoder
        
        model_path = "/home/jemmi/trajectory-system/models/alpamayo"
        
        wrapper = AlpamayoWrapper(
            model_path=model_path,
            quantization="int8",
            device="cuda"
        )
        wrapper.load()
        
        # Create dummy inputs
        images = torch.zeros(4, 4, 3, 320, 576)  # 4 cameras, 4 frames
        egomotion = torch.zeros(16, 12)
        prompt = "You are driving. Plan a safe trajectory."
        
        print("  Running inference...")
        import time
        start = time.time()
        
        output = wrapper.generate(
            images=images,
            egomotion_history=egomotion,
            prompt=prompt,
            current_speed=10.0
        )
        
        elapsed = time.time() - start
        print(f"  Inference time: {elapsed:.2f}s")
        
        # Decode trajectory
        decoder = TrajectoryDecoder()
        waypoints, causation = decoder.decode_from_model_output(output, initial_speed=10.0)
        
        print(f"  Generated {len(waypoints)} waypoints")
        print(f"  Causation: {causation[:100]}...")
        
        print("  ✓ Inference working")
        return True
        
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Alpamayo Model Loading Tests")
    print("=" * 60)
    print()
    
    if not test_alpamayo_import():
        print("\nCannot proceed without alpamayo_r1 package")
        return
    print()
    
    if not test_torch_cuda():
        print("\nWarning: CUDA not available, model loading will be slow")
    print()
    
    if not test_model_loading():
        print("\nModel loading failed, cannot test inference")
        return
    print()
    
    # Optional: test inference (takes time)
    response = input("Run inference test? This will take a minute or two. (y/n): ")
    if response.lower() == 'y':
        test_inference()
    
    print()
    print("=" * 60)
    print("Tests completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
