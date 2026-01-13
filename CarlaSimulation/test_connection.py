#!/usr/bin/env python3
"""
CARLA Connection Test Script
Tests connection to CARLA server and reports status.
"""

import sys
import time
import argparse
import socket

def check_port(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if port is open."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except socket.error:
        return False

def test_carla_connection(host: str, port: int, timeout: float = 10.0) -> dict:
    """Test CARLA connection and return status."""
    result = {
        'port_open': False,
        'connected': False,
        'world_accessible': False,
        'map_name': None,
        'num_actors': 0,
        'error': None
    }
    
    # Check port
    print(f"Checking if port {port} is open...")
    result['port_open'] = check_port(host, port)
    if not result['port_open']:
        result['error'] = f"Port {port} is not open"
        return result
    print(f"  Port {port} is open!")
    
    # Try to import carla
    try:
        import carla
    except ImportError as e:
        result['error'] = f"Cannot import carla module: {e}"
        return result
    
    # Try to connect
    print(f"Connecting to CARLA at {host}:{port}...")
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        result['connected'] = True
        print("  Connected to CARLA client!")
        
        # Try to get world
        print("Getting world...")
        world = client.get_world()
        result['world_accessible'] = True
        print("  World accessible!")
        
        # Get map name
        map_name = world.get_map().name
        result['map_name'] = map_name
        print(f"  Map: {map_name}")
        
        # Count actors
        actors = world.get_actors()
        result['num_actors'] = len(actors)
        print(f"  Actors in world: {len(actors)}")
        
        # Get weather
        weather = world.get_weather()
        print(f"  Weather - Cloudiness: {weather.cloudiness:.0f}%, "
              f"Precipitation: {weather.precipitation:.0f}%")
        
        # Get settings
        settings = world.get_settings()
        print(f"  Sync mode: {settings.synchronous_mode}, "
              f"Fixed delta: {settings.fixed_delta_seconds}")
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Test CARLA connection')
    parser.add_argument('--host', type=str, default='localhost',
                        help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000,
                        help='CARLA server port')
    parser.add_argument('--timeout', type=float, default=10.0,
                        help='Connection timeout in seconds')
    parser.add_argument('--wait', action='store_true',
                        help='Wait for CARLA to become available')
    parser.add_argument('--max-wait', type=int, default=120,
                        help='Maximum wait time in seconds')
    args = parser.parse_args()
    
    print("=" * 50)
    print("CARLA Connection Test")
    print("=" * 50)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Timeout: {args.timeout}s")
    print("")
    
    if args.wait:
        print(f"Waiting for CARLA (max {args.max_wait}s)...")
        start_time = time.time()
        while time.time() - start_time < args.max_wait:
            result = test_carla_connection(args.host, args.port, args.timeout)
            if result['world_accessible']:
                break
            elapsed = time.time() - start_time
            print(f"\nRetrying... ({elapsed:.0f}s / {args.max_wait}s)\n")
            time.sleep(5)
    else:
        result = test_carla_connection(args.host, args.port, args.timeout)
    
    print("")
    print("=" * 50)
    print("Result Summary")
    print("=" * 50)
    print(f"  Port open: {result['port_open']}")
    print(f"  Connected: {result['connected']}")
    print(f"  World accessible: {result['world_accessible']}")
    print(f"  Map: {result['map_name']}")
    print(f"  Actors: {result['num_actors']}")
    
    if result['error']:
        print(f"  Error: {result['error']}")
        sys.exit(1)
    else:
        print("")
        print("SUCCESS: CARLA is ready!")
        sys.exit(0)

if __name__ == '__main__':
    main()
