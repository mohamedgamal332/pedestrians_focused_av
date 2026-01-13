#!/bin/bash
# =============================================================================
# CARLA Data Collection - Auto-Start Script
# =============================================================================
# This script will:
# 1. Check if CARLA is running, start it if not
# 2. Wait for CARLA to be fully ready
# 3. Run the data collection
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DURATION=${1:-300}

# IMPORTANT: Set this to your CARLA installation directory
CARLA_ROOT="${CARLA_ROOT:-/media/mr-theta-iii/Dev/Playgrounds/Unreal/Carla9.15}"

CARLA_PORT=2000
CARLA_HOST="localhost"

echo "=============================================="
echo "CARLA Auto-Start Data Collection"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  CARLA_ROOT: ${CARLA_ROOT}"
echo "  Duration: ${DURATION} seconds"
echo "  Port: ${CARLA_PORT}"
echo ""

# Function to check if CARLA accepts connections
check_carla_ready() {
    python3 -c "
import carla
import sys
try:
    client = carla.Client('${CARLA_HOST}', ${CARLA_PORT})
    client.set_timeout(3.0)
    world = client.get_world()
    map_name = world.get_map().name
    print(f'CARLA ready: {map_name}')
    sys.exit(0)
except Exception as e:
    sys.exit(1)
" 2>/dev/null
    return $?
}

# Function to wait for CARLA to be ready
wait_for_carla() {
    local max_wait=${1:-120}
    local waited=0
    local check_interval=5
    
    echo "Waiting for CARLA to be ready (max ${max_wait}s)..."
    
    while [ $waited -lt $max_wait ]; do
        if check_carla_ready; then
            echo "CARLA is ready!"
            return 0
        fi
        
        echo "  Still waiting... (${waited}s / ${max_wait}s)"
        sleep $check_interval
        waited=$((waited + check_interval))
    done
    
    echo "ERROR: CARLA did not become ready within ${max_wait} seconds"
    return 1
}

# Check if CARLA is already running
if pgrep -f "CarlaUE4" > /dev/null; then
    echo "CARLA is already running!"
    
    # Give it a moment and check if it's ready
    if check_carla_ready; then
        echo "CARLA is ready to accept connections."
    else
        echo "CARLA is running but not ready yet. Waiting..."
        if ! wait_for_carla 60; then
            echo "CARLA is not responding. You may need to restart it."
            exit 1
        fi
    fi
else
    echo "CARLA is not running. Starting CARLA..."
    
    # Check if CARLA directory exists
    if [ ! -d "${CARLA_ROOT}" ]; then
        echo "ERROR: CARLA directory not found: ${CARLA_ROOT}"
        echo ""
        echo "Please set CARLA_ROOT environment variable:"
        echo "  export CARLA_ROOT=/path/to/carla"
        echo ""
        echo "Or edit this script and set CARLA_ROOT directly."
        exit 1
    fi
    
    # Check if CarlaUE4.sh exists
    if [ ! -f "${CARLA_ROOT}/CarlaUE4.sh" ]; then
        echo "ERROR: CarlaUE4.sh not found in ${CARLA_ROOT}"
        exit 1
    fi
    
    # Start CARLA
    echo "Starting CARLA from ${CARLA_ROOT}..."
    cd "${CARLA_ROOT}"
    ./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=low &
    CARLA_PID=$!
    echo "CARLA started with PID: ${CARLA_PID}"
    
    # Wait for CARLA to be ready
    echo ""
    if ! wait_for_carla 120; then
        echo "Failed to start CARLA. Check the logs."
        exit 1
    fi
fi

# Run data collection
echo ""
echo "=============================================="
echo "Starting Data Collection"
echo "=============================================="
echo ""

cd "${SCRIPT_DIR}"
python carla_data_collector.py \
    --config config.yaml \
    --skeleton skeleton.txt \
    --host ${CARLA_HOST} \
    --port ${CARLA_PORT} \
    --duration ${DURATION}

echo ""
echo "=============================================="
echo "Collection Complete!"
echo "=============================================="
