"""
Mock Alpamayo wrapper for testing when real model loading fails.
Generates simple trajectories based on current state.
"""

import numpy as np
import logging
from typing import Dict, Optional
import torch
import time

logger = logging.getLogger(__name__)


class MockAlpamayoWrapper:
    """
    Mock Alpamayo model that generates reasonable trajectories
    without the actual model. Useful for testing the pipeline.
    """
    
    def __init__(
        self,
        model_path: str = None,
        quantization: str = "int8",
        device: str = "cuda",
        max_new_tokens: int = 512
    ):
        self.model_path = model_path
        self.quantization = quantization
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._loaded = False
        
        # Trajectory parameters
        self.n_waypoints = 64
        self.dt = 0.1  # 10Hz
    
    def load(self):
        """Simulate model loading."""
        logger.info("=" * 50)
        logger.info("MOCK ALPAMAYO - Using mock model for testing")
        logger.info("=" * 50)
        logger.warning("Real model loading failed - using mock trajectories")
        
        # Simulate loading delay
        time.sleep(2.0)
        
        self._loaded = True
        logger.info("Mock model 'loaded' successfully")
    
    def is_loaded(self) -> bool:
        return self._loaded
    
    def get_memory_usage(self) -> Dict:
        return {
            'allocated_gb': 0.0,
            'reserved_gb': 0.0,
            'max_allocated_gb': 0.0
        }
    
    def generate(
        self,
        images: torch.Tensor,
        egomotion_history: torch.Tensor,
        prompt: str,
        timestamps: Optional[torch.Tensor] = None,
        current_speed: float = 0.0
    ) -> Dict:
        """
        Generate a mock trajectory based on simple heuristics.
        
        This creates a trajectory that:
        1. Maintains roughly constant speed
        2. Goes mostly straight with slight curves
        3. Generates reasonable causation text
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        
        # Simulate inference delay (real model takes 4-6 seconds)
        time.sleep(0.5)
        
        # Parse prompt for context
        has_pedestrians = "pedestrian" in prompt.lower()
        
        # Generate trajectory
        if has_pedestrians:
            # Slow down if pedestrians mentioned
            target_speed = max(5.0, current_speed * 0.7)
            causation = self._generate_pedestrian_causation(prompt)
        else:
            # Maintain or slightly increase speed
            target_speed = min(current_speed + 1.0, 14.0)  # ~50 km/h max
            causation = self._generate_normal_causation(current_speed)
        
        # Generate accelerations to reach target speed
        speed_diff = target_speed - current_speed
        if abs(speed_diff) < 0.5:
            accel = 0.0
        elif speed_diff > 0:
            accel = min(speed_diff / (self.n_waypoints * self.dt), 2.0)  # Gentle acceleration
        else:
            accel = max(speed_diff / (self.n_waypoints * self.dt), -3.0)  # Gentle braking
        
        accelerations = np.full(self.n_waypoints, accel)
        
        # Add slight random curvature (simulating lane following)
        curvatures = np.random.normal(0, 0.01, self.n_waypoints)
        curvatures = np.clip(curvatures, -0.1, 0.1)
        
        return {
            'trajectory': {
                'accelerations': accelerations,
                'curvatures': curvatures
            },
            'reasoning': causation
        }
    
    def _generate_pedestrian_causation(self, prompt: str) -> str:
        """Generate causation text when pedestrians are present."""
        templates = [
            "Pedestrian activity detected ahead. Reducing speed for safety. "
            "Maintaining awareness of pedestrian movements and preparing to yield if necessary.",
            
            "Observing pedestrians in the vicinity. Applying cautious driving behavior. "
            "Speed reduced to ensure adequate reaction time.",
            
            "Multiple pedestrians detected. Implementing defensive driving strategy. "
            "Slowing down and increasing following distance for safety margins."
        ]
        return np.random.choice(templates)
    
    def _generate_normal_causation(self, speed: float) -> str:
        """Generate causation text for normal driving."""
        speed_kmh = speed * 3.6
        
        if speed_kmh < 20:
            return (
                f"Currently traveling at {speed_kmh:.1f} km/h. Road ahead appears clear. "
                "Gradually accelerating to match traffic flow while maintaining safety margins."
            )
        elif speed_kmh < 40:
            return (
                f"Maintaining speed of {speed_kmh:.1f} km/h. Traffic conditions normal. "
                "Continuing along planned route with standard lane positioning."
            )
        else:
            return (
                f"Traveling at {speed_kmh:.1f} km/h. Monitoring road conditions. "
                "Maintaining current trajectory with minor adjustments for road geometry."
            )


def create_alpamayo_wrapper(
    model_path: str,
    quantization: str = "int8",
    device: str = "cuda",
    max_new_tokens: int = 512,
    use_mock: bool = False
):
    """
    Factory function to create Alpamayo wrapper.
    Falls back to mock if real model fails to load.
    """
    if use_mock:
        logger.info("Using mock Alpamayo (explicitly requested)")
        return MockAlpamayoWrapper(model_path, quantization, device, max_new_tokens)
    
    try:
        from governor.alpamayo_wrapper import AlpamayoWrapper
        wrapper = AlpamayoWrapper(model_path, quantization, device, max_new_tokens)
        wrapper.load()
        return wrapper
    except Exception as e:
        logger.error(f"Failed to load real Alpamayo model: {e}")
        logger.info("Falling back to mock Alpamayo")
        mock = MockAlpamayoWrapper(model_path, quantization, device, max_new_tokens)
        mock.load()
        return mock

