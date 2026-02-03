"""XML route writer for CaRL compatibility."""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import List
from .coordinate_transform import Pose


class XMLRouteWriter:
    """Write route files in CaRL-compatible XML format."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_route(
        self,
        waypoints: List[Pose],
        route_id: str,
        town: str = "Town10HD_Opt",
        filename: str = None
    ) -> Path:
        """
        Write waypoints to XML route file.
        
        Args:
            waypoints: List of Pose objects
            route_id: Unique route identifier
            town: CARLA town name
            filename: Output filename (default: trajectory_{route_id}.xml)
            
        Returns:
            Path to written file
        """
        # Create root element
        root = ET.Element("route")
        root.set("id", route_id)
        root.set("town", town)
        
        # Add waypoints
        for wp in waypoints:
            waypoint_elem = ET.SubElement(root, "waypoint")
            waypoint_elem.set("x", f"{wp.x:.6f}")
            waypoint_elem.set("y", f"{wp.y:.6f}")
            waypoint_elem.set("z", f"{wp.z:.6f}")
            waypoint_elem.set("pitch", f"{wp.pitch:.6f}")
            waypoint_elem.set("yaw", f"{wp.yaw:.6f}")
            waypoint_elem.set("roll", f"{wp.roll:.6f}")
        
        # Pretty print
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        
        # Remove extra blank lines
        lines = [line for line in xml_str.split('\n') if line.strip()]
        xml_str = '\n'.join(lines)
        
        # Write file
        if filename is None:
            filename = f"trajectory_{route_id}.xml"
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            f.write(xml_str)
        
        return output_path
    
    def read_route(self, filepath: str) -> List[Pose]:
        """
        Read waypoints from XML route file.
        
        Args:
            filepath: Path to XML file
            
        Returns:
            List of Pose objects
        """
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        waypoints = []
        for wp_elem in root.findall('waypoint'):
            pose = Pose(
                x=float(wp_elem.get('x', 0)),
                y=float(wp_elem.get('y', 0)),
                z=float(wp_elem.get('z', 0)),
                pitch=float(wp_elem.get('pitch', 0)),
                yaw=float(wp_elem.get('yaw', 0)),
                roll=float(wp_elem.get('roll', 0))
            )
            waypoints.append(pose)
        
        return waypoints
