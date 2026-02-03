#!/bin/bash
# Run complete experiment with both conditions

EXPERIMENT_DIR=~/trajectory-system/experiments
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "Governor-Reflex Experiment Runner"
echo "Timestamp: $TIMESTAMP"
echo "=========================================="

# Create experiment directories
mkdir -p "$EXPERIMENT_DIR/with_pedestrian_info/$TIMESTAMP"
mkdir -p "$EXPERIMENT_DIR/without_pedestrian_info/$TIMESTAMP"

# Function to run a single experiment
run_experiment() {
    local condition=$1
    local ped_flag=$2
    local output_dir="$EXPERIMENT_DIR/$condition/$TIMESTAMP"
    
    echo ""
    echo "=========================================="
    echo "Running experiment: $condition"
    echo "Output: $output_dir"
    echo "=========================================="
    
    # Clear runtime directory
    rm -rf ~/trajectory-system/runtime/input/*
    rm -rf ~/trajectory-system/runtime/output/*
    rm -rf ~/trajectory-system/runtime/cameras/*
    
    # Update config for this condition
    if [ "$condition" == "with_pedestrian_info" ]; then
        sed -i 's/include_pedestrian_info:.*/include_pedestrian_info: true/' ~/trajectory-system/governor_reflex/config.yaml
    else
        sed -i 's/include_pedestrian_info:.*/include_pedestrian_info: false/' ~/trajectory-system/governor_reflex/config.yaml
    fi
    
    # Start Governor in background
    echo "Starting Governor..."
    gnome-terminal --title="Governor - $condition" -- bash -c "
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate alpo
        cd ~/trajectory-system/governor_reflex
        python -u governor/main_governor.py 2>&1 | tee $output_dir/governor.log
        exec bash
    " &
    
    # Wait for Governor to initialize
    sleep 30
    
    # Start Reflex
    echo "Starting Reflex..."
    gnome-terminal --title="Reflex - $condition" -- bash -c "
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate PCLA
        cd ~/trajectory-system/governor_reflex
        python -u reflex/main_reflex.py $ped_flag 2>&1 | tee $output_dir/reflex.log
        exec bash
    " &
    
    echo "Experiment '$condition' started in separate terminals"
    echo "Monitor the terminals and close them when done"
}

# Parse arguments
case "$1" in
    "with")
        run_experiment "with_pedestrian_info" "--with-pedestrians"
        ;;
    "without")
        run_experiment "without_pedestrian_info" "--without-pedestrians"
        ;;
    "both")
        echo "Running both conditions sequentially..."
        echo "First: WITH pedestrian info"
        run_experiment "with_pedestrian_info" "--with-pedestrians"
        echo ""
        echo "Press Enter when first experiment is complete..."
        read
        echo ""
        echo "Second: WITHOUT pedestrian info"
        run_experiment "without_pedestrian_info" "--without-pedestrians"
        ;;
    *)
        echo "Usage: $0 {with|without|both}"
        echo ""
        echo "  with    - Run with pedestrian info enabled"
        echo "  without - Run with pedestrian info disabled"
        echo "  both    - Run both conditions sequentially"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Experiment setup complete"
echo "=========================================="
