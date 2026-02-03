"""Prompt templates for Alpamayo model."""

from typing import List, Dict, Optional


class PromptBuilder:
    """Build text prompts for Alpamayo model."""
    
    SYSTEM_PROMPT = """You are an autonomous vehicle AI. Your task is to plan a safe trajectory for the next 6.4 seconds while explaining your reasoning."""
    
    WITH_PEDESTRIANS_TEMPLATE = """You are driving an autonomous vehicle. Plan a safe trajectory.

Current situation:
- Current speed: {speed_kmh:.1f} km/h
- Speed limit: {speed_limit:.1f} km/h
- Route: {route_context}

Detected pedestrians:
{pedestrian_descriptions}

Based on the camera views and the situation above, plan the next 6.4 seconds of trajectory. Explain your reasoning step by step, considering:
1. The current traffic situation
2. Pedestrian movements and intentions
3. Required speed adjustments
4. Any necessary lane changes or turns

Provide your Chain-of-Causation reasoning, then output the planned trajectory."""

    WITHOUT_PEDESTRIANS_TEMPLATE = """You are driving an autonomous vehicle. Plan a safe trajectory.

Current situation:
- Current speed: {speed_kmh:.1f} km/h
- Speed limit: {speed_limit:.1f} km/h
- Route: {route_context}

Based on the camera views and the situation above, plan the next 6.4 seconds of trajectory. Explain your reasoning step by step, considering:
1. The current traffic situation
2. Road conditions and obstacles
3. Required speed adjustments
4. Any necessary lane changes or turns

Provide your Chain-of-Causation reasoning, then output the planned trajectory."""

    def __init__(self, include_pedestrian_info: bool = True):
        self.include_pedestrian_info = include_pedestrian_info
    
    def format_pedestrian_description(self, pedestrian: Dict) -> str:
        """Format a single pedestrian description."""
        behavior = pedestrian.get('behavior', 'unknown')
        distance = pedestrian.get('distance_to_ego', 0)
        
        # Determine direction relative to ego
        pos = pedestrian.get('position', {})
        vel = pedestrian.get('velocity', {})
        
        # Position description
        if pos.get('y', 0) < -2:
            lateral = "on the left"
        elif pos.get('y', 0) > 2:
            lateral = "on the right"
        else:
            lateral = "ahead"
        
        # Movement description
        if behavior == 'standing':
            movement = "stationary"
        else:
            vx = vel.get('x', 0)
            vy = vel.get('y', 0)
            
            if abs(vy) > abs(vx):
                if vy > 0:
                    movement = f"{behavior}, moving right-to-left"
                else:
                    movement = f"{behavior}, moving left-to-right"
            else:
                if vx > 0:
                    movement = f"{behavior}, moving away"
                else:
                    movement = f"{behavior}, approaching"
        
        return f"- Pedestrian {lateral} at {distance:.1f}m: {movement}"
    
    def build_prompt(
        self,
        speed_kmh: float,
        speed_limit: float,
        route_context: str,
        pedestrians: Optional[List[Dict]] = None
    ) -> str:
        """
        Build the prompt for Alpamayo.
        
        Args:
            speed_kmh: Current speed in km/h
            speed_limit: Speed limit in km/h
            route_context: Description of upcoming route
            pedestrians: List of pedestrian info dicts
            
        Returns:
            Formatted prompt string
        """
        if self.include_pedestrian_info and pedestrians:
            # Format pedestrian descriptions
            ped_descriptions = []
            for ped in pedestrians[:10]:  # Limit to 10 nearest
                ped_descriptions.append(self.format_pedestrian_description(ped))
            
            if not ped_descriptions:
                ped_text = "- No pedestrians detected in immediate vicinity"
            else:
                ped_text = '\n'.join(ped_descriptions)
            
            return self.WITH_PEDESTRIANS_TEMPLATE.format(
                speed_kmh=speed_kmh,
                speed_limit=speed_limit,
                route_context=route_context,
                pedestrian_descriptions=ped_text
            )
        else:
            return self.WITHOUT_PEDESTRIANS_TEMPLATE.format(
                speed_kmh=speed_kmh,
                speed_limit=speed_limit,
                route_context=route_context
            )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt."""
        return self.SYSTEM_PROMPT
