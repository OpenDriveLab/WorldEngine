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
REACT_TYPE=$5
ASSET_NAME=${6:-$DATA_TYPE}

# Convert relative paths to absolute paths based on WORLDENGINE_ROOT
SIMENGINE_ROOT="$WORLDENGINE_ROOT/projects/SimEngine"
ALGENGINE_ROOT="$WORLDENGINE_ROOT/projects/AlgEngine"
PYTHONPATH=$SIMENGINE_ROOT:$ALGENGINE_ROOT:$PYTHONPATH

# SimEngine setting
ASSET_FOLDER_PATH="$WORLDENGINE_ROOT/data/sim_engine/assets/${ASSET_NAME}/assets"
DATAFILE_FOLDER_PATH="data/sim_engine/scenarios/original/${DATA_TYPE}"

# Test path (absolute, relative to WORLDENGINE_ROOT)
TEST_PATH="$WORLDENGINE_ROOT/experiments/closed_loop_exps/${MODEL_NAME}/${DATA_TYPE}_${REACT_TYPE}"

# Check if test_path already exists (skip check if resume is enabled)
if [ -d "$TEST_PATH" ] && [ "$ENABLE_RESUME" = false ]; then
    echo "ERROR: Test path already exists!"
    echo "Run the following command to remove it:"
    echo "rm -rf $TEST_PATH"
    echo "Or set ENABLE_RESUME=true to resume from where you left off."
    exit 1
fi

mkdir -p $TEST_PATH/plan_traj
mkdir -p $TEST_PATH/frames
mkdir -p $TEST_PATH/merged_ann_files
mkdir -p $TEST_PATH/WE_output

rm -rf $TEST_PATH/merged_ann_files/*.pkl
rm -rf $TEST_PATH/frames/*.pkl
rm -rf $TEST_PATH/plan_traj/*.npy
rm -f $TEST_PATH/WE_output/simulation_completed.flag

# Run SimEngine simulation
cd $SIMENGINE_ROOT
conda run --no-capture-output -n $SIMENGINE_ENV_NAME \
python worldengine/runner/run_simulation.py \
    debug_mode=True \
    debug_scene_name=null \
    data_file_folder_path=$DATAFILE_FOLDER_PATH \
    asset_folder_path=$ASSET_FOLDER_PATH \
    data_pkl_file_name=all_scenarios.pkl \
    output_dir=$TEST_PATH/WE_output \
    job_name=${DATA_TYPE}_${REACT_TYPE}_${MODEL_NAME} \
    use_planner_actions=true \
    ego_controller=log_play_controller \
    ego_policy=env_input_policy \
    ego_client=navformer_client \
    ego_navigation=trajectory_navigation \
    $([ "$REACT_TYPE" = "R" ] && echo "agent_policy=idm_policy") \
    $([ "$REACT_TYPE" = "R" ] && echo "agent_navigation=idm_navigation") \
    planner_data_path=$TEST_PATH/plan_traj \
    planner_client_folder=$TEST_PATH/frames \
    with_metric_manager=true \
    with_dense_reward_manager=false \
    distributed_mode=SINGLE_NODE \
    enable_resume=true \
    completed_scenarios_dir=$TEST_PATH/completed_scenarios &

# Start AlgEngine client
cd $ALGENGINE_ROOT
conda run --no-capture-output -n $ALGENGINE_ENV_NAME \
python closed_loop/sim_test.py \
    $CFG \
    $CKPT \
    --log-dir $TEST_PATH/WE_output \
    --cfg-options sim.monitored_folder="$TEST_PATH/frames" \
    sim.plan_save_path="$TEST_PATH/plan_traj" \
    sim.merged_ann_save_dir="$TEST_PATH/merged_ann_files" \
    sim.clean_temp_files=True \
    sim.clean_record_data=False \
    data_root="$TEST_PATH/WE_output/openscene_format/"
