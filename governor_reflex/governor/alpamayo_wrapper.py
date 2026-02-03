"""Alpamayo model wrapper following official inference pattern."""

import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlpamayoWrapper:
    """
    Wrapper for Alpamayo model following official NVIDIA inference pattern.
    
    Based on: alpamayo_r1/test_inference.py
    """
    
    def __init__(
        self,
        model_path: str,
        quantization: str = "int8",
        device: str = "cuda",
        max_generation_length: int = 256,
        num_traj_samples: int = 1,
        temperature: float = 0.6,
        top_p: float = 0.98
    ):
        self.model_path = Path(model_path)
        self.quantization = quantization
        self.device = device
        self.max_generation_length = max_generation_length
        self.num_traj_samples = num_traj_samples
        self.temperature = temperature
        self.top_p = top_p
        
        self.model = None
        self.processor = None
        self.helper = None
        self._loaded = False
    
    def load(self):
        """Load the Alpamayo model following official pattern."""
        if self._loaded:
            logger.info("Model already loaded")
            return
        
        logger.info(f"Loading Alpamayo model from {self.model_path}")
        logger.info(f"Quantization: {self.quantization}")
        
        try:
            # Import required modules
            from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
            from alpamayo_r1 import helper
            self.helper = helper
            
            # Determine dtype based on quantization
            if self.quantization == "int8":
                # For INT8, we use bitsandbytes
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )
                
                logger.info("Loading model with INT8 quantization...")
                self.model = AlpamayoR1.from_pretrained(
                    str(self.model_path),
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            elif self.quantization == "int4":
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                logger.info("Loading model with INT4 quantization...")
                self.model = AlpamayoR1.from_pretrained(
                    str(self.model_path),
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                # Full precision (bfloat16)
                logger.info("Loading model with bfloat16 precision...")
                self.model = AlpamayoR1.from_pretrained(
                    str(self.model_path),
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                ).to(self.device)
            
            # Get processor using helper
            logger.info("Getting processor...")
            self.processor = helper.get_processor(self.model.tokenizer)
            
            self.model.eval()
            self._loaded = True
            
            logger.info("Model loaded successfully!")
            
            # Log memory usage
            mem = self.get_memory_usage()
            if mem:
                logger.info(f"GPU Memory: {mem['allocated_gb']:.2f} GB allocated")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def prepare_inputs(
        self,
        images: torch.Tensor,
        ego_history_xyz: torch.Tensor,
        ego_history_rot: torch.Tensor,
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prepare inputs in the format expected by Alpamayo.
        
        Args:
            images: Camera images tensor - flattened (N, C, H, W)
            ego_history_xyz: Ego position history (history_len, 3)
            ego_history_rot: Ego rotation history (history_len, 3, 3)
            prompt: Optional text prompt
            
        Returns:
            Dict ready for model inference
        """
        # Model expects ego_history_xyz shape: (batch, n_traj_group, history_len, 3)
        # Model expects ego_history_rot shape: (batch, n_traj_group, history_len, 3, 3)
        
        # Add batch and n_traj_group dimensions
        if ego_history_xyz.dim() == 2:
            # (history_len, 3) -> (1, 1, history_len, 3)
            ego_history_xyz = ego_history_xyz.unsqueeze(0).unsqueeze(0)
        elif ego_history_xyz.dim() == 3:
            # (batch, history_len, 3) -> (batch, 1, history_len, 3)
            ego_history_xyz = ego_history_xyz.unsqueeze(1)
        
        if ego_history_rot.dim() == 3:
            # (history_len, 3, 3) -> (1, 1, history_len, 3, 3)
            ego_history_rot = ego_history_rot.unsqueeze(0).unsqueeze(0)
        elif ego_history_rot.dim() == 4:
            # (batch, history_len, 3, 3) -> (batch, 1, history_len, 3, 3)
            ego_history_rot = ego_history_rot.unsqueeze(1)
        
        # Create messages from images using helper
        # Images should be (N, C, H, W) for message creation
        if images.dim() == 5:
            # (batch, N, C, H, W) -> (N, C, H, W)
            images = images[0]
        
        messages = self.helper.create_message(images)
        
        # Process with chat template
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # Build model inputs dict
        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        
        # Move to device
        model_inputs = self.helper.to_device(model_inputs, self.device)
        
        return model_inputs
    
    def generate(
        self,
        images: torch.Tensor,
        egomotion_history: torch.Tensor,
        prompt: str,
        timestamps: Optional[torch.Tensor] = None,
        current_speed: float = 0.0
    ) -> Dict:
        """
        Generate trajectory and reasoning from inputs.
        
        Args:
            images: Camera images tensor (num_cameras * frames, 3, H, W)
            egomotion_history: Egomotion tensor (history_size, 12) - xyz (3) + rot_matrix (9)
            prompt: Text prompt (for context)
            timestamps: Frame timestamps (optional)
            current_speed: Current vehicle speed (m/s)
            
        Returns:
            Dictionary with 'trajectory' and 'reasoning' keys
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            # Parse egomotion history into xyz and rotation
            # egomotion_history shape: (history_size, 12) where 12 = 3 (xyz) + 9 (rotation matrix)
            if egomotion_history.dim() == 2:
                ego_history_xyz = egomotion_history[:, :3]  # (history_size, 3)
                ego_history_rot = egomotion_history[:, 3:].reshape(-1, 3, 3)  # (history_size, 3, 3)
            else:
                ego_history_xyz = egomotion_history[..., :3]
                ego_history_rot = egomotion_history[..., 3:].reshape(*egomotion_history.shape[:-1], 3, 3)
            
            # Prepare inputs (this will add the required dimensions)
            model_inputs = self.prepare_inputs(
                images=images,
                ego_history_xyz=ego_history_xyz,
                ego_history_rot=ego_history_rot,
                prompt=prompt
            )
            
            # Log shapes for debugging
            logger.info(f"Input shapes - xyz: {model_inputs['ego_history_xyz'].shape}, rot: {model_inputs['ego_history_rot'].shape}")
            
            # Run inference
            logger.info("Running Alpamayo inference...")
            
            torch.cuda.manual_seed_all(42)  # For reproducibility
            
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_xyz, pred_rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                        data=model_inputs,
                        top_p=self.top_p,
                        temperature=self.temperature,
                        num_traj_samples=self.num_traj_samples,
                        max_generation_length=self.max_generation_length,
                        return_extra=True,
                    )
            
            logger.info(f"Output shapes - xyz: {pred_xyz.shape}, rot: {pred_rot.shape}")
            
            # Parse outputs
            result = self._parse_outputs(pred_xyz, pred_rot, extra, current_speed)
            
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
            # Return fallback trajectory
            return self._generate_fallback(current_speed)
    
    def _parse_outputs(
        self,
        pred_xyz: torch.Tensor,
        pred_rot: torch.Tensor,
        extra: Dict,
        current_speed: float
    ) -> Dict:
        """Parse model outputs to our standard format."""
        result = {
            'trajectory': None,
            'reasoning': '',
            'pred_xyz': None,
            'pred_rot': None
        }
        
        try:
            # pred_xyz shape: [batch_size, num_traj_sets, num_traj_samples, num_waypoints, 3]
            # We take the first trajectory
            pred_xyz_np = pred_xyz.cpu().numpy()
            pred_rot_np = pred_rot.cpu().numpy()
            
            # Extract first batch, first set, first sample
            trajectory_xyz = pred_xyz_np[0, 0, 0]  # (num_waypoints, 3)
            trajectory_rot = pred_rot_np[0, 0, 0]  # (num_waypoints, 3, 3)
            
            # Store raw predictions
            result['pred_xyz'] = trajectory_xyz
            result['pred_rot'] = trajectory_rot
            
            # Convert to waypoints format
            waypoints = []
            for i in range(len(trajectory_xyz)):
                x, y, z = trajectory_xyz[i]
                
                # Extract yaw from rotation matrix
                rot = trajectory_rot[i]
                yaw = np.arctan2(rot[1, 0], rot[0, 0])
                
                waypoints.append({
                    'x': float(x),
                    'y': float(y),
                    'z': float(z),
                    'yaw': float(np.degrees(yaw))
                })
            
            result['trajectory'] = {
                'waypoints': waypoints,
                'xyz': trajectory_xyz,
                'rot': trajectory_rot
            }
            
            # Extract Chain-of-Causation reasoning
            if extra and 'cot' in extra:
                cot = extra['cot']
                if isinstance(cot, list) and len(cot) > 0:
                    result['reasoning'] = cot[0] if isinstance(cot[0], str) else str(cot[0])
                else:
                    result['reasoning'] = str(cot)
            
            logger.info(f"Generated {len(waypoints)} waypoints")
            if result['reasoning']:
                logger.info(f"CoC: {result['reasoning'][:100]}...")
            
        except Exception as e:
            logger.warning(f"Error parsing outputs: {e}")
            import traceback
            traceback.print_exc()
            result = self._generate_fallback(current_speed)
        
        return result
    
    def _generate_fallback(self, current_speed: float) -> Dict:
        """Generate fallback trajectory (continue straight, maintain speed)."""
        logger.warning("Using fallback trajectory")
        
        # Generate simple straight trajectory
        num_waypoints = 64
        dt = 0.1
        
        # Gentle deceleration to be safe
        if current_speed > 5.0:
            accel = -0.5
        else:
            accel = 0.0
        
        waypoints = []
        x, y, z = 0.0, 0.0, 0.0
        v = current_speed
        
        for i in range(num_waypoints):
            waypoints.append({
                'x': x,
                'y': y,
                'z': z,
                'yaw': 0.0
            })
            v = max(0.0, v + accel * dt)
            x += v * dt
        
        return {
            'trajectory': {
                'waypoints': waypoints,
                'accelerations': np.full(num_waypoints, accel),
                'curvatures': np.zeros(num_waypoints)
            },
            'reasoning': 'Fallback trajectory: maintaining course with gentle deceleration for safety.'
        }
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    def get_memory_usage(self) -> Dict:
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'reserved_gb': torch.cuda.memory_reserved() / 1e9,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9
            }
        return {}
