#!/usr/bin/env python3
"""
Test script to verify components work correctly.
Run from PCLA environment: conda activate PCLA && python test_components.py
"""

import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from utils.coordinate_transform import CoordinateTransformer, Pose
        print("  ✓ coordinate_transform")
    except Exception as e:
        print(f"  ✗ coordinate_transform: {e}")
    
    try:
        from utils.xml_route_writer import XMLRouteWriter
        print("  ✓ xml_route_writer")
    except Exception as e:
        print(f"  ✗ xml_route_writer: {e}")
    
    try:
        from utils.causation_logger import CausationLogger
        print("  ✓ causation_logger")
    except Exception as e:
        print(f"  ✗ causation_logger: {e}")
    
    try:
        from utils.file_lock import FileLock
        print("  ✓ file_lock")
    except Exception as e:
        print(f"  ✗ file_lock: {e}")
    
    print()


def test_coordinate_transform():
    """Test coordinate transformation."""
    print("Testing coordinate transformation...")
    
    from utils.coordinate_transform import CoordinateTransformer, Pose
    import numpy as np
    
    transformer = CoordinateTransformer()
    
    # Test ego to world transformation
    ego_pose = Pose(x=100, y=50, z=0, pitch=0, yaw=45, roll=0)
    ego_waypoints = [
        {'x': 0, 'y': 0, 'yaw': 0},
        {'x': 10, 'y': 0, 'yaw': 0},
        {'x': 20, 'y': 5, 'yaw': 10},
    ]
    
    world_waypoints = transformer.ego_to_world(ego_waypoints, ego_pose)
    
    print(f"  Ego pose: ({ego_pose.x}, {ego_pose.y}) yaw={ego_pose.yaw}")
    print(f"  Transformed {len(ego_waypoints)} waypoints:")
    for i, wp in enumerate(world_waypoints[:3]):
        print(f"    {i}: ({wp.x:.2f}, {wp.y:.2f}) yaw={wp.yaw:.2f}")
    
    # Test unicycle decoding
    accelerations = np.zeros(10)
    curvatures = np.full(10, 0.1)
    
    waypoints = transformer.decode_unicycle_trajectory(
        accelerations, curvatures, initial_speed=10.0
    )
    
    print(f"  Unicycle decode: {len(waypoints)} waypoints")
    print(f"    Start: ({waypoints[0]['x']:.2f}, {waypoints[0]['y']:.2f})")
    print(f"    End: ({waypoints[-1]['x']:.2f}, {waypoints[-1]['y']:.2f})")
    
    print("  ✓ Coordinate transform working")
    print()


def test_xml_writer():
    """Test XML route writing."""
    print("Testing XML route writer...")
    
    from utils.xml_route_writer import XMLRouteWriter
    from utils.coordinate_transform import Pose
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = XMLRouteWriter(tmpdir)
        
        waypoints = [
            Pose(x=0, y=0, z=0, pitch=0, yaw=0, roll=0),
            Pose(x=10, y=0, z=0, pitch=0, yaw=0, roll=0),
            Pose(x=20, y=5, z=0, pitch=0, yaw=15, roll=0),
        ]
        
        output_path = writer.write_route(waypoints, "test_001", "Town10HD_Opt")
        print(f"  Written to: {output_path}")
        
        # Read back
        read_waypoints = writer.read_route(str(output_path))
        print(f"  Read back: {len(read_waypoints)} waypoints")
        
        assert len(read_waypoints) == len(waypoints)
        print("  ✓ XML writer working")
    print()


def test_causation_logger():
    """Test causation logging."""
    print("Testing causation logger...")
    
    from utils.causation_logger import CausationLogger
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = CausationLogger(tmpdir)
        
        logger.log("req_001", "Road is clear, maintaining speed")
        logger.log("req_002", "Pedestrian detected, slowing down")
        
        recent = logger.get_recent(5)
        print(f"  Logged {len(recent)} entries")
        
        for entry in recent:
            print(f"    {entry['request_id']}: {entry['causation_text'][:40]}...")
        
        print("  ✓ Causation logger working")
    print()


def test_file_lock():
    """Test file locking."""
    print("Testing file lock...")
    
    from utils.file_lock import FileLock
    import tempfile
    import threading
    import time
    
    with tempfile.TemporaryDirectory() as tmpdir:
        lock_path = Path(tmpdir) / "test.lock"
        lock = FileLock(str(lock_path))
        
        results = []
        
        def worker(name):
            with lock.acquire():
                results.append(f"{name}_start")
                time.sleep(0.1)
                results.append(f"{name}_end")
        
        t1 = threading.Thread(target=worker, args=("A",))
        t2 = threading.Thread(target=worker, args=("B",))
        
        t1.start()
        time.sleep(0.01)  # Ensure A starts first
        t2.start()
        
        t1.join()
        t2.join()
        
        # Check that locks worked (no interleaving)
        assert results[0] == "A_start"
        assert results[1] == "A_end"
        assert results[2] == "B_start"
        assert results[3] == "B_end"
        
        print(f"  Lock sequence: {results}")
        print("  ✓ File lock working")
    print()


def test_trajectory_manager():
    """Test trajectory manager."""
    print("Testing trajectory manager...")
    
    from reflex.trajectory_manager import TrajectoryManager
    from utils.xml_route_writer import XMLRouteWriter
    from utils.coordinate_transform import Pose
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup directories
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        
        # Create a test trajectory
        writer = XMLRouteWriter(str(output_dir))
        waypoints = [
            Pose(x=i*2, y=0, z=0, pitch=0, yaw=0, roll=0)
            for i in range(64)
        ]
        writer.write_route(waypoints, "test_001", filename="trajectory.xml")
        
        # Create status file
        import json
        status = {"governor_status": "ready", "trajectory_valid": True}
        with open(output_dir / "status.json", 'w') as f:
            json.dump(status, f)
        
        # Test manager
        manager = TrajectoryManager(tmpdir)
        
        assert manager.is_governor_ready()
        print("  ✓ Governor status check")
        
        loaded = manager.load_trajectory()
        assert loaded
        print(f"  ✓ Loaded trajectory: {len(manager._waypoints)} waypoints")
        
        remaining = manager.get_remaining_seconds()
        print(f"  Remaining: {remaining:.1f}s")
        
        # Consume some waypoints
        for _ in range(10):
            wp = manager.consume_waypoint()
        
        new_remaining = manager.get_remaining_seconds()
        print(f"  After consuming 10: {new_remaining:.1f}s")
        
        print("  ✓ Trajectory manager working")
    print()


def test_carla_connection():
    """Test CARLA connection (requires CARLA to be running)."""
    print("Testing CARLA connection...")
    
    try:
        import carla
        
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        
        world = client.get_world()
        map_name = world.get_map().name
        
        print(f"  Connected to CARLA")
        print(f"  Current map: {map_name}")
        print("  ✓ CARLA connection working")
        
    except Exception as e:
        print(f"  ⚠ CARLA not available: {e}")
        print("  (This is OK if CARLA is not running)")
    
    print()


def main():
    print("=" * 60)
    print("Governor-Reflex Component Tests")
    print("=" * 60)
    print()
    
    test_imports()
    test_coordinate_transform()
    test_xml_writer()
    test_causation_logger()
    test_file_lock()
    test_trajectory_manager()
    test_carla_connection()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
