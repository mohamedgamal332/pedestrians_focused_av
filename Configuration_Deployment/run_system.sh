#!/bin/bash

# 1. Start CARLA Server in background (Off-screen for HPC performance)
./CarlaUE4.sh -RenderOffScreen -nosound -benchmark -fps=20 &
sleep 10 # Wait for world to load

# 2. Start Milvus Vector DB (Local Memory)
sudo docker-compose up -d milvus_db

# 3. Launch the Main Bridge (Orchestrator)
# This will use GPU 0 for the Reflex/CaRL model
export CUDA_VISIBLE_DEVICES=0
python3 main_bridge.py --config config.yaml &

# 4. Launch the Auditor (Llama 3.3)
# This will use GPU 1 + a portion of the 10.5TB RAM
export CUDA_VISIBLE_DEVICES=1
python3 auditor.py --config config.yaml &

echo "🚀 Autonomous System is LIVE."
echo "Monitoring reasoning traces and physical safety..."

# Keep script alive to monitor background processes
trap "kill 0" EXIT
wait