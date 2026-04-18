# SimEngine Usage Guide

This guide covers how to use SimEngine for closed-loop simulation, rollout generation, and model testing. SimEngine provides photorealistic simulation environments powered by 3D Gaussian Splatting.

## Table of Contents

- [Quick Reference](#quick-reference)
- [Running Simulations](#running-simulations)
- [Rollout Scripts](#rollout-scripts)
- [Testing Scripts](#testing-scripts)
- [Configuration](#configuration)
- [Output Format](#output-format)
- [Troubleshooting](#troubleshooting)

---

## Quick Reference

```bash

cd projects/SimEngine

# Single-GPU testing
bash scripts/run_testing.sh <config> <checkpoint> <model_name> <data_type> <react_type> [<asset_name>]

# Multi-GPU distributed testing (8 GPUs)
bash scripts/run_ray_distributed_testing.sh <config> <checkpoint> <model_name> <data_type> <react_type> [<asset_name>]

# Multi-GPU distributed rollout (data generation)
bash scripts/run_ray_distributed_rollout.sh <config> <checkpoint> <model_name> <data_type> [<asset_name>]
```

---

## Running Simulations

### Basic Simulation

Run a basic simulation with default settings:

```bash
conda activate simengine
cd projects/SimEngine

python worldengine/runner/run_simulation.py \
    data_pkl_file_name=your_scenario.pkl
```

**Parameters:**
- `data_pkl_file_name`: Scene pickle file in `data/sim_engine/scenarios/`

### Custom Scenario Count

```bash
python worldengine/runner/run_simulation.py \
    data_pkl_file_name=your_scenario.pkl \
    num_scenarios=50 
```
---

## Rollout Scripts

Rollout scripts run closed-loop simulation with an E2E planning model and **reward computation**. Useful for:
- Generating training data for RL-based fine-tuning (with reward)
- Rollout data augmentation/synthesis

### Multi-GPU Distributed Rollout

For large-scale data generation:

```bash
export WORLDENGINE_ROOT=/path/to/WorldEngine

cd  projects/SimEngine
bash scripts/run_ray_distributed_rollout.sh \
    $WORLDENGINE_ROOT/projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py \
    $WORLDENGINE_ROOT/data/alg_engine/ckpts/e2e_vadv2_50pct_ep8.pth \
    e2e_vadv2_50pct \
    navtrain_50pct_collision \ 
    navtrain
```

**Arguments:**
1. `<config>`: Model configuration file
2. `<checkpoint>`: Trained model checkpoint
3. `<model_name>`: Experiment name (creates output folder)
4. `<data_type>`: Scenario split (e.g., `navtrain_50pct_collision`). Must match a directory under `data/sim_engine/scenarios/original/`
5. `<asset_name>` (optional): Asset folder name under `data/sim_engine/assets/`. Defaults to `<data_type>` if not provided

**Features:**
- Distributes scenarios across 8 GPUs (8 splits)
- Each GPU processes 1/8 of scenarios in parallel
- Auto-merges results at the end
- Resumes from previous runs (`ENABLE_RESUME=true`)

### Customizing Rollout Splits

Edit the rollout script to change GPU count:

```bash
# In run_ray_distributed_rollout.sh, line ~140
# Change from:
for i in {0..7}; do    # 8 GPUs
    run_planner $i &
done

# To (for 4 GPUs):
for i in {0..3}; do    # 4 GPUs
    run_planner $i &
done
```

---

## Testing Scripts

Testing scripts run closed-loop evaluation with a **planning model**. Used for:
- Model performance evaluation
- Rare case testing

### Single-GPU Testing

For quick tests or debugging:

```bash
export WORLDENGINE_ROOT=/path/to/WorldEngine

cd  projects/SimEngine
bash scripts/run_testing.sh \
    $WORLDENGINE_ROOT/projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py \
    $WORLDENGINE_ROOT/data/alg_engine/ckpts/e2e_vadv2_50pct_ep8.pth \
    e2e_vadv2_50pct \
    navtest_failures \
    NR
```

**Arguments:**
1. `<config>`: Model configuration file
2. `<checkpoint>`: Trained model checkpoint
3. `<model_name>`: Experiment name (creates output folder)
4. `<data_type>`: Scenario split (e.g., `navtest_failures`, `navtest`)
5. `<react_type>`: Reactive mode (`NR` or `R`)
   - `NR` (Non-Reactive): Other agents replay logged trajectories
   - `R` (Reactive): Other agents use IDM policy to react
6. `<asset_name>` (optional): Asset folder name under `data/sim_engine/assets/`. Defaults to `<data_type>` if not provided


**Output:**
```
experiments/closed_loop_exps/e2e_vadv2_50pct/navtest_failures_NR/
├── WE_output/
│   └── openscene_format/
│       └── all_scenes_pdm_averages_NR.csv  # Main metrics CSV
├── plan_traj/              # Model-predicted trajectories (.npy)
├── frames/                 # Communication files (deleted after run)
└── merged_ann_files/       # Merged annotations (.pkl)
```

### Multi-GPU Distributed Testing (Recommended)

For faster evaluation on large test sets:

```bash
export WORLDENGINE_ROOT=/path/to/WorldEngine

cd  projects/SimEngine
bash scripts/run_ray_distributed_testing.sh \
    $WORLDENGINE_ROOT/projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py \
    $WORLDENGINE_ROOT/data/alg_engine/ckpts/e2e_vadv2_50pct_ep8.pth \
    e2e_vadv2_50pct \
    navtest_failures \
    NR
```

**How it works:**
1. Launches **SimEngine ray distributed server** (handles scenario distribution)
2. Spawns **8 AlgEngine clients** (one per GPU)
3. Each client processes 1/8 of scenarios in parallel
4. Auto-merges results after completion


### Testing on Different Scenario Splits

```bash
# Test on all navtest scenarios (not just failures)
bash scripts/run_ray_distributed_testing.sh \
    ... \
    navtest \  # Changed from navtest_failures
    NR

# Test on training set
bash scripts/run_ray_distributed_testing.sh \
    ... \
    navtrain \
    NR
```

### Testing with Reactive Agents

```bash
# Non-reactive (default): Other agents replay logged paths
bash scripts/run_ray_distributed_testing.sh \
    ... \
    NR

# Reactive: Other agents use IDM to react to ego
bash scripts/run_ray_distributed_testing.sh \
    ... \
    R  # Changed to R
```

**Reactive mode impact:**
- More challenging (agents respond to ego's mistakes)
- Lower scores expected
- Tests robustness to unexpected agent behaviors

---

## Configuration

SimEngine uses Hydra for configuration management.

### Configuration Files

Default config: `projects/SimEngine/worldengine/configs/default_runner.yaml`

Override with:
```bash
python worldengine/runner/run_simulation.py \
    --config-name custom_config.yaml
```


## Output Format

### Directory Structure

```
experiments/closed_loop_exps/<model_name>/<data_type>_<react_type>/
├── split_0/ ... split_7/
├── plan_traj/
└── WE_output/
    └── openscene_format/
        ├── meta_datas/
        ├── pdms_pkl/
        ├── sensor_blobs/
        └── all_scenes_pdm_averages_NR.csv
```

### OpenScene Format

```
WE_output/openscene_format/
├── sensor_blobs/
│   ├── CAM_F0/                 # Front camera
│   ├── CAM_L0/                 # Left camera
│   ├── CAM_R0/                 # Right camera
│   └── LIDAR_TOP/              # LiDAR point clouds
├── meta_datas/
│   └── {scenario_token}.pkl         # Per-scenario metadata
└── all_scenes_pdm_averages_NR.csv
```

---

## Utility Scripts

### Convert nuPlan Data to SimEngine Format

Convert nuPlan dataset to SimEngine scenario format with navsim filter-based scene selection.

Example:
```bash
conda activate simengine

python projects/SimEngine/worldengine/utils/dataset_utils/nuplan/digitaltwin_nuplan_converter_navsim_filter.py \
    --navsim-filters $ALGENGINE_ROOT/configs/navsim_splits/navtrain_split/e2e_vadv2_50pct_rare/navtrain_50pct_collision.yaml \
        $ALGENGINE_ROOT/configs/navsim_splits/navtrain_split/e2e_vadv2_50pct_rare/navtrain_50pct_ep_1pct.yaml \
        $ALGENGINE_ROOT/configs/navsim_splits/navtrain_split/e2e_vadv2_50pct_rare/navtrain_50pct_off_road.yaml \
    --out-dir data/sim_engine/scenarios/original/navtrain_vadv2_50pct_rare \
    --num-processes 8
```

**Key Parameters:**
- `--navsim-filters`: Path(s) to navsim config file(s) for scenario filtering (supports multiple files)
- `--digitaltwin-asset-root`: Digital Twin asset root directory (default: `data/sim_engine/assets/navtrain`)
- `--nuplan-root-path`: nuPlan dataset root directory
- `--nuplan-db-path`: nuPlan database file directory
- `--openscene-dataroot`: OpenScene data root directory
- `--out-dir`: Output directory for converted scenario files
- `--sample-interval`: Keyframe sampling interval (default: 10)
- `--num-processes`: Number of parallel processes

**Output:**
- `{out-dir}/all_scenarios.pkl` - Converted WorldEngine scenario file

The script extracts from Digital Twin config and nuPlan raw data:
- Ego and agent vehicle trajectories
- Camera and LiDAR calibration parameters
- Traffic light states
- Map features (lanes, intersections, etc.)

Converted scenarios can be directly used for SimEngine simulation.

---

### Merge Distributed Results

After distributed simulation, merge results:
(Run Automatically)
```bash
conda activate simengine

python projects/SimEngine/scripts/merge_simulation_results.py \
    --test_path $WORLDENGINE_ROOT/experiments/closed_loop_exps/e2e_vadv2_50pct/navtrain_ep_per1_NR  \
    --react_type NR
```

### Export OpenScene Format Data

Export simulation data for training:

```bash
conda activate simengine

python projects/SimEngine/scripts/export_simulation_data.py \
    --test_path $WORLDENGINE_ROOT/experiments/closed_loop_exps/e2e_vadv2_50pct/navtrain_ep_per1_NR 
```

**Output:** `data/alg_engine/openscene-synthetic/`

---

## Troubleshooting

### Issue 1: Ray initialization fails

**Error:** `ray.exceptions.RaySystemError: System error: Failed to start Ray`

**Solution:**
```bash
# Kill existing Ray processes
ray stop

# Check port availability
netstat -tulpn | grep 6379

# Restart simulation
bash scripts/run_ray_distributed_testing.sh ...
```

### Issue 2: GPU out of memory

**Error:** `CUDA out of memory`

**Solution:**
```bash
# Reduce GPU allocation (fewer scenarios per GPU)
# Edit the script and change:
number_of_gpus_allocated_per_simulation=1.0  # From 0.5

# Or reduce batch size in model config
```

### Issue 3: "No scenarios found"

**Error:** `FileNotFoundError: all_scenarios.pkl`

**Solution:**
```bash
# Check scenario data path
ls -lh data/sim_engine/scenarios/original/navtest_failures/

# Verify pickle file exists
file data/sim_engine/scenarios/original/navtest_failures/all_scenarios.pkl
```

### Issue 4: Simulation hangs

**Symptoms:** Process runs but no progress

**Solution:**
```bash
# Check if AlgEngine client is connected
tail -f experiments/closed_loop_exps/*/WE_output/*.log

# Check for stuck frames
ls -lt experiments/closed_loop_exps/*/frames/ | head -20

# Kill and restart with debug mode
debug_mode=true num_scenarios=1
```

### Issue 5: Resume not working

**Error:** Scenarios restart from beginning

**Solution:**
```bash
# Ensure resume flag is set
enable_resume=true

# Check completed scenarios file
cat experiments/closed_loop_exps/*/completed_scenarios/completed_scenarios.txt

# Verify scenario IDs match
```

---

## Performance Tips

1. **GPU Allocation:** Use `0.5` for most scenarios (2 workers per GPU)
2. **Resume Mode:** Always enable `enable_resume=true` for long runs
3. **Distributed Mode:** Both testing and rollout use `SCENARIO_BASED` mode by default
4. **Cleanup:** Set `sim.clean_temp_files=True` to save disk space
5. **Ray Resources:** Limit Ray memory with `RAY_memory_limit=30GB` if needed

---

## Next Steps

- **Train models:** See [AlgEngine Usage Guide](algengine_usage.md)
- **Understand data:** See [Data Organization](data_organization.md)

For questions, visit [GitHub Discussions](https://github.com/OpenDriveLab/WorldEngine/discussions).
