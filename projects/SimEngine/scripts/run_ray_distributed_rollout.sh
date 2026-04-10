#!/usr/bin/env bash

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH." >&2
  echo "Hint: please install conda and run 'conda init' (then restart your shell)." >&2
  exit 1
fi

SIMENGINE_ENV_NAME="simengine"
ALGENGINE_ENV_NAME="algengine"

CFG=$1
CKPT=$2
MODEL_NAME=$3
DATA_TYPE=$4
ASSET_NAME=${5:-$DATA_TYPE}

# Resume flag - set to true to skip already completed scenarios
ENABLE_RESUME=true

# Convert relative paths to absolute paths based on WORLDENGINE_ROOT
SIMENGINE_ROOT="$WORLDENGINE_ROOT/projects/SimEngine"
ALGENGINE_ROOT="$WORLDENGINE_ROOT/projects/AlgEngine"
PYTHONPATH=$SIMENGINE_ROOT:$ALGENGINE_ROOT:$PYTHONPATH

# SimEngine setting 
ASSET_FOLDER_PATH="$WORLDENGINE_ROOT/data/sim_engine/assets/${ASSET_NAME}/assets"
DATAFILE_FOLDER_PATH="data/sim_engine/scenarios/original/${DATA_TYPE}" # DEFAULT original, can choose augmented

# Test path (absolute, relative to WORLDENGINE_ROOT)
test_path="$WORLDENGINE_ROOT/experiments/closed_loop_exps/${MODEL_NAME}/${DATA_TYPE}_NR"

# Check if test_path already exists (skip check if resume is enabled)
if [ -d "$test_path" ] && [ "$ENABLE_RESUME" = false ]; then
    echo "ERROR: Test path already exists!"
    echo "Run the following command to remove it:"
    echo "rm -rf $test_path"
    echo "Or set ENABLE_RESUME=true to resume from where you left off."
    exit 1
fi

# Set error handling
set -euo pipefail

cleanup() {
  echo "Cleaning up processes..."
  trap - SIGINT SIGTERM EXIT

  kill 0 || true
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM EXIT

# Main execution
echo "Starting distributed simulation with 8 splits..."
echo "Model: $MODEL_NAME, Data: $DATA_TYPE, Asset: $ASSET_NAME"
echo "Resume mode: $ENABLE_RESUME"

cd $SIMENGINE_ROOT
conda run --no-capture-output -n $SIMENGINE_ENV_NAME python worldengine/runner/run_simulation.py \
    debug_mode=True \
    debug_scene_name=null \
    data_file_folder_path=$DATAFILE_FOLDER_PATH \
    asset_folder_path=$ASSET_FOLDER_PATH \
    data_pkl_file_name=all_scenarios.pkl \
    output_dir=$test_path/__WORKER_ID__/WE_output \
    job_name=${DATA_TYPE}_NR_${MODEL_NAME} \
    use_planner_actions=true \
    ego_policy=env_input_policy \
    ego_client=navformer_client \
    ego_controller=log_play_controller \
    ego_navigation=trajectory_navigation \
    planner_data_path=$test_path/__WORKER_ID__/plan_traj \
    planner_client_folder=$test_path/__WORKER_ID__/frames \
    with_metric_manager=true \
    with_dense_reward_manager=true \
    distributed_mode=SCENARIO_BASED \
    worker=ray_distributed \
    worker_id_prefix=split_ \
    enable_resume=$ENABLE_RESUME \
    completed_scenarios_dir=$test_path/__WORKER_ID__/completed_scenarios &

we_pid=$!
echo "WorldEngine started with PID: $we_pid with ray distributed mode!"

sleep 30

# Function to run single simulation
run_planner() {
    local split_id=$1
    local gpu_id=$1

    # Set GPU environment
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    
    local split_suffix="split_${split_id}"
    local test_path_worker="$test_path/${split_suffix}"

    mkdir -p $test_path_worker/WE_output
    mkdir -p $test_path_worker/plan_traj
    mkdir -p $test_path_worker/frames
    mkdir -p $test_path_worker/merged_ann_files

    rm -rf $test_path_worker/merged_ann_files/*.pkl
    rm -rf $test_path_worker/frames/*.pkl
    rm -rf $test_path_worker/plan_traj/*.npy

    # Clean up previous simulation completed flag if exists
    rm -f $test_path_worker/WE_output/simulation_completed.flag

    # Start AlgEngine client
    cd $ALGENGINE_ROOT
    conda run --no-capture-output -n $ALGENGINE_ENV_NAME python closed_loop/sim_test.py \
        $CFG \
        $CKPT \
        --log-dir $test_path_worker \
        --cfg-options sim.monitored_folder="$test_path_worker/frames" \
        sim.plan_save_path="$test_path_worker/plan_traj" \
        sim.merged_ann_save_dir="$test_path_worker/merged_ann_files" \
        sim.clean_temp_files=True \
        sim.clean_record_data=False \
        data_root="$test_path_worker/WE_output/openscene_format/" &

    local alg_pid=$!
    echo "AlgEngine started with PID: $alg_pid for split ${split_id}"

    wait $alg_pid
    local alg_exit_code=$?

    # Check exit codes
    if [ $alg_exit_code -eq 0 ]; then
        echo "Split ${split_id} completed successfully on GPU $gpu_id"
    else
        echo "Split ${split_id} failed - WE exit code: $we_exit_code - Alg exit code: $alg_exit_code"
        return 1
    fi
}

# Run simulations in parallel
for i in {0..7}; do
    run_planner $i &
done

# Wait for all simulations to complete
wait

echo "All simulation splits completed successfully."

trap - SIGINT SIGTERM EXIT

# Merge results
cd $SIMENGINE_ROOT
conda run --no-capture-output -n $SIMENGINE_ENV_NAME \
    python scripts/merge_simulation_results.py \
    --test_path "$test_path" \
    --react_type NR

TIME=`date +"%y%m%d"`
conda run --no-capture-output -n $SIMENGINE_ENV_NAME \
    python scripts/export_simulation_data.py \
    --test_path "$test_path" \
    --appendix $TIME
