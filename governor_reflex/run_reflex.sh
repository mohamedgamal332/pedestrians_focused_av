#!/bin/bash
# Run the Reflex process

echo "=========================================="
echo "Starting Reflex Process (CaRL)"
echo "=========================================="

# Activate PCLA environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PCLA

# Add CARLA Python API to path (adjust version if needed)
export CARLA_ROOT=/opt/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.8-linux-x86_64.egg

# Change to governor_reflex directory  
cd ~/trajectory-system/governor_reflex

# Run reflex
python -u reflex/main_reflex.py "$@"
