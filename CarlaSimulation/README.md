# CARLA Stereo Pedestrian Data Collection System

Collects synchronized stereo camera data (RGB + Depth) with pedestrian skeleton annotations.

## Features

- **Robust Connection Handling**: Automatic retry logic when connecting to CARLA
- **Stereo RGB and Depth Images**: Synchronized capture from left/right cameras
- **Occlusion-Aware Visibility**: Per-bone visibility detection using depth buffer
- **Jaywalking Behavior**: Pedestrians check traffic before crossing
- **Behavior Annotation**: Walking, running, crossing, waiting, idle states
- **Skeleton Synchronization**: Bone positions captured in sync with images

## Quick Start

### Option 1: Manual CARLA Start

Terminal 1 - Start CARLA:

    cd /path/to/carla
    ./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=low

Wait 60+ seconds for CARLA to initialize.

Terminal 2 - Run collection:

    cd CarlaSimulation
    pip install -r requirements.txt
    ./run_collection.sh 300

### Option 2: Auto-Start Script

    export CARLA_ROOT=/path/to/carla
    cd CarlaSimulation
    ./start_and_collect.sh 300

### Option 3: Test Connection First

    # Test if CARLA is ready
    python test_connection.py --host localhost --port 2000

    # Wait for CARLA to be ready (with retries)
    python test_connection.py --wait --max-wait 120

## Troubleshooting Connection Issues

### Error: "time-out of 30000ms while waiting for the simulator"

**Causes:**
1. CARLA hasn't finished initializing (most common)
2. CARLA crashed during startup
3. Wrong host/port

**Solutions:**

1. Wait longer after starting CARLA:

       ./CarlaUE4.sh -RenderOffScreen -nosound &
       sleep 60
       ./run_collection.sh 300

2. Use the auto-start script:

       export CARLA_ROOT=/path/to/carla
       ./start_and_collect.sh 300

3. Test connection first:

       python test_connection.py --wait

4. Check CARLA process:

       ps aux | grep CarlaUE4
       netstat -tlnp | grep 2000

### Error: "Cannot import carla module"

Make sure you have the CARLA Python API installed:

    # Option 1: Install from CARLA egg
    pip install /path/to/carla/PythonAPI/carla/dist/carla-0.9.15-py3.x-linux-x86_64.egg

    # Option 2: Add to PYTHONPATH
    export PYTHONPATH=$PYTHONPATH:/path/to/carla/PythonAPI/carla/dist/carla-0.9.15-py3.x-linux-x86_64.egg

## Configuration

Edit config.yaml to customize:

### Connection Settings

    connection:
      host: "localhost"
      port: 2000
      timeout: 30.0
      max_retries: 12
      retry_delay: 5.0
      post_map_load_wait: 5.0

### Simulation Settings

    simulation:
      map: "Town10HD_Opt"
      duration_seconds: 300
      tick_rate: 15

### Traffic Settings

    traffic:
      num_vehicles: 25
      num_pedestrians: 100
      runner_ratio: 0.25
      walk_speed_min: 0.8
      walk_speed_max: 1.8
      run_speed_min: 2.5
      run_speed_max: 4.5

### Jaywalking Settings

    jaywalking:
      enabled: true
      jaywalker_ratio: 0.3
      safety_time_threshold: 4.0
      safety_distance_threshold: 15.0
      max_wait_time: 10.0
      crossing_speed_multiplier: 1.3

### Sensor Settings

    sensors:
      stereo_baseline: 0.54
      camera_height: 1.5
      camera_forward: 2.2
      image_width: 1920
      image_height: 1080
      fov: 90

## Pedestrian Behaviors

| Behavior | Description |
|----------|-------------|
| WALKING | Normal walking pace (0.8-1.8 m/s) |
| RUNNING | Fast movement (2.5-4.5 m/s) |
| WAITING_TO_CROSS | Stopped at road edge, checking traffic |
| CROSSING | Actively crossing the road |
| IDLE | Stationary (speed < 0.1 m/s) |

## Jaywalking State Machine

    NORMAL -> APPROACHING -> WAITING -> CROSSING -> COMPLETED -> NORMAL
                               |
                         (checks traffic)
                               |
                         safe? -> cross
                         not safe? -> wait more
                         timeout? -> cross anyway

## Output Structure

    session_YYYYMMDD_HHMMSS/
    ├── rgb_left/                    # Left camera RGB (PNG)
    │   ├── frame_000001.png
    │   └── ...
    ├── rgb_right/                   # Right camera RGB (PNG)
    ├── depth_left/                  # Left depth maps (16-bit PNG, mm)
    ├── depth_right/                 # Right depth maps (16-bit PNG, mm)
    ├── annotations/                 # Per-frame JSON annotations
    │   ├── frame_000001.json
    │   └── ...
    ├── demo/                        # Annotated visualization frames
    ├── camera_intrinsics.json       # Camera calibration
    ├── road_layout.json             # Map waypoints
    └── session_metadata.json        # Session info

## Annotation Format

Each frame JSON contains:

    {
      "frame_id": 12345,
      "timestamp": 123.456,
      "weather": {
        "preset": "ClearNoon",
        "cloudiness": 0,
        "precipitation": 0
      },
      "ego_vehicle": {
        "id": 100,
        "transform": {...},
        "velocity": {...}
      },
      "pedestrians": [
        {
          "id": 200,
          "behavior": "waiting_to_cross",
          "assigned_behavior": "walking",
          "is_runner": false,
          "speed": 0.0,
          "jaywalking": {
            "is_jaywalker": true,
            "jaywalking_state": "waiting",
            "is_safe_to_cross": false,
            "blocking_vehicles": [101, 102],
            "crossing_count": 1
          },
          "visible_in_frame": true,
          "visibility": {
            "left": {
              "visible": true,
              "visible_bones": 25,
              "occluded_bones": 3
            },
            "right": {...}
          },
          "skeleton": {
            "crl_Head__C": {
              "world": {"location": {...}, "rotation": {...}},
              "cameras": {
                "left": {
                  "pixel": [960, 540],
                  "bone_depth": 15.5,
                  "visibility_state": "visible"
                }
              }
            }
          }
        }
      ]
    }

## Visibility States

| State | Description |
|-------|-------------|
| VISIBLE | Bone in frame and not occluded |
| OCCLUDED | Bone in frame but blocked by object |
| OUT_OF_FRAME | Bone projects outside image |
| BEHIND_CAMERA | Bone is behind camera |

## Demo Visualization

Demo images show:
- Skeleton overlays with color-coded bones
- Visibility indicators (green=visible, red=occluded)
- Behavior labels and tags
- Speed and depth information
- Jaywalking status

Tags shown:
- [R] = Runner
- [JW] = Jaywalker
- Wait: X cars = Waiting for vehicles
- CROSSING! = Actively crossing

## Command Line Options

    python carla_data_collector.py [OPTIONS]

    Options:
      --config PATH      Config file (default: config.yaml)
      --skeleton PATH    Skeleton file (default: skeleton.txt)
      --host HOST        CARLA host (default: localhost)
      --port PORT        CARLA port (default: 2000)
      --duration SECS    Override duration

## Tips for Best Results

1. Use a lighter map for testing:
   Change map: "Town10HD_Opt" to map: "Town03" for faster loading.

2. Reduce traffic for debugging:

       traffic:
         num_vehicles: 5
         num_pedestrians: 20

3. More frequent demo images:

       output:
         demo_interval: 100

4. Check GPU memory:
   Town10HD_Opt requires significant VRAM. Use -quality-level=low if needed.

## License

MIT License - See LICENSE file for details.
