#!/bin/bash
# =============================================================================
# CARLA Data Collection - Basic Run Script
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DURATION=${1:-300}

echo "=============================================="
echo "CARLA Stereo Pedestrian Data Collection"
echo "=============================================="
echo "Duration: ${DURATION} seconds"
echo ""

# Check if CARLA is running (use -f for pattern matching)
if ! pgrep -f "CarlaUE4" > /dev/null; then
    echo "WARNING: CARLA server does not appear to be running!"
    echo ""
    echo "Start CARLA with one of these commands:"
    echo "  ./CarlaUE4.sh -RenderOffScreen -nosound"
    echo "  ./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=low"
    echo ""
    echo "Then wait at least 60 seconds before running this script."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "CARLA process detected!"
    echo "Giving CARLA a moment to ensure it's ready..."
    sleep 5
fi

cd "${SCRIPT_DIR}"
python carla_data_collector.py \
    --config config.yaml \
    --skeleton skeleton.txt \
    --duration ${DURATION}

echo ""
echo "Collection complete!"
