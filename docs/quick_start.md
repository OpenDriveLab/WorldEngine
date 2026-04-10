# Quick Start Guide

This guide will help you run your first WorldEngine experiment in minutes. We'll start with a quick test using pre-trained models, then point you to detailed guides for each subsystem.

## Prerequisites

Before starting, ensure you have:
- ✅ Completed [Installation](installation.md) for both environments
- ✅ Set up environment variables (`WORLDENGINE_ROOT`)
- ✅ Downloaded pre-trained model checkpoint
- ✅ Prepared scenario data (see [Data Organization](data_organization.md))

---

## Quick Test (5 Minutes)

The fastest way to verify your installation and see WorldEngine in action is to run the quick test script.

### What the Quick Test Does

The quick test script:
1. Loads a pre-trained end-to-end driving model
2. Runs closed-loop simulation on test scenarios
3. Evaluates the model's performance with PDM metrics
4. Saves results to `experiments/closed_loop_exps/`

### Option 1: Single GPU Test

For systems with 1 GPU or for quick testing:

```bash
cd /path/to/WorldEngine

# Set your WorldEngine root path
export WORLDENGINE_ROOT=$(pwd)

# Run quick test
bash scripts/closed_loop_test.sh
```

**Expected output:**
```
Starting simulation...
AlgEngine client connected
Processing scenario 1/10...
Processing scenario 2/10...
...
All scenarios completed!
Results saved to: experiments/closed_loop_exps/e2e_vadv2_50pct/navtest_failures_NR/
```

**Time:** ~5-10 minutes (depends on GPU and scenario count)

### Option 2: Multi-GPU Test (Recommended)

For systems with 8 GPUs (faster parallel execution):

```bash
cd /path/to/WorldEngine
export WORLDENGINE_ROOT=$(pwd)

# Run multi-GPU quick test (8 splits in parallel)
bash scripts/multigpu_closed_loop_test.sh
```

**Expected output:**
```
Starting distributed simulation with 8 splits...
WorldEngine started with PID: 12345 with ray distributed mode!
AlgEngine started with PID: 12346 for split 0
AlgEngine started with PID: 12347 for split 1
...
All simulation splits completed successfully.
Merging results...
Results merged to: experiments/closed_loop_exps/e2e_vadv2_50pct/navtest_failures_NR/
```

**Time:** ~2-3 minutes with 8 GPUs

**Note:** If you have fewer than 8 GPUs, edit `scripts/multigpu_closed_loop_test.sh` and change the loop `{0..7}` to match your GPU count (e.g., `{0..3}` for 4 GPUs).

---

## Understanding Quick Test Results

After the test completes, check your results:

```bash
cd experiments/closed_loop_exps/e2e_vadv2_50pct/navtest_failures_NR/

# View aggregated metrics
cat WE_output/openscene_format/all_scenes_pdm_averages_NR.csv
```


---

## What Happens Under the Hood?

The quick test script calls `scripts/run_testing.sh` (or `run_ray_distributed_testing.sh` for multi-GPU), which:

1. **Launches SimEngine** (in `simengine` conda env)
   - Loads scenario data from `data/sim_engine/scenarios/`
   - Loads 3DGS scene assets from `data/sim_engine/assets/`
   - Starts simulation server

2. **Launches AlgEngine Client** (in `algengine` conda env)
   - Loads pre-trained model checkpoint
   - Connects to SimEngine via socket
   - Receives observations, outputs actions

3. **Runs Closed-Loop Simulation**
   - SimEngine sends camera images + sensor data to AlgEngine
   - AlgEngine predicts trajectory
   - SimEngine executes trajectory and renders next frame
   - Repeat for 20 steps per scenario

4. **Computes Metrics**
   - SimEngine evaluates using PDM (Planning Deviation Metric)
   - Results saved as CSV files

For more details on the testing pipeline, see:
- **[SimEngine Usage Guide](simengine_usage.md)** - Rollout and testing scripts
- **[AlgEngine Usage Guide](algengine_usage.md)** - Model inference and evaluation

---

## Customizing the Quick Test

### Change Test Scenarios

Edit `scripts/closed_loop_test.sh` to test on different scenarios:

```bash
# Original (navtest rare cases num: 288)
bash scripts/run_testing.sh \
    ... \
    navtest_failures \
    NR

# Test on all navtest scenarios
bash scripts/run_testing.sh \
    ... \
    navtest \  # Changed from navtest_failures
    NR
```

### Change Model Checkpoint

Edit the checkpoint path in `scripts/closed_loop_test.sh`:

```bash
bash scripts/run_testing.sh \
    .../configs/worldengine/e2e_vadv2_100pct.py \  # Changed config
    .../ckpts/e2e_vadv2_100pct_ep20.pth \  # Changed checkpoint
    e2e_vadv2_100pct \  # Changed experiment name
    navtest_failures \
    NR
```

### Change Reactive Mode

Test with **reactive agents** (other vehicles respond to ego):

```bash
bash scripts/run_testing.sh \
    ... \
    NR  # Change to R for Reactive mode
```

- `NR` (Non-Reactive): Other agents replay logged trajectories (default)
- `R` (Reactive): Other agents use IDM policy to react to ego vehicle

---

## Next Steps: Deep Dive into Subsystems

Now that you've verified your installation, dive deeper into each subsystem:

### 🎮 SimEngine - Closed-Loop Simulation

Learn how to:
- Run simulations on custom scenarios
- Use different rollout scripts
- Configure simulation parameters
- Export simulation data
- Debug simulation issues

👉 **[SimEngine Usage Guide](simengine_usage.md)**

### 🧠 AlgEngine - Model Training & Evaluation

Learn how to:
- Train models from scratch
- Fine-tune with long tail cases
- Use different model architectures
- Configure training hyperparameters

👉 **[AlgEngine Usage Guide](algengine_usage.md)**

---

## Complete Pipeline Example

For a complete workflow from data to deployment, follow these steps:

### 1. Prepare Data

```bash
# Symlink datasets
cd WorldEngine/data
ln -s /path/to/openscene-v1.1 raw/
ln -s /path/to/ckpts alg_engine/
ln -s /path/to/sim_assets sim_engine/assets/

# Set environment variables
export WORLDENGINE_ROOT=/path/to/WorldEngine
export NUPLAN_MAPS_ROOT=$WORLDENGINE_ROOT/data/raw/nuplan/maps
```

### 2. Train a Model (AlgEngine)

```bash
conda activate algengine
cd projects/AlgEngine

# Train on 50% data
./scripts/e2e_dist_train.sh configs/worldengine/e2e_vadv2_50pct.py 8
```

See [AlgEngine Usage Guide](algengine_usage.md#training) for details.

### 3. Evaluate Open-Loop (AlgEngine)

```bash
conda activate algengine
cd projects/AlgEngine

# Evaluate on test set
./scripts/e2e_dist_eval.sh \
    configs/worldengine/e2e_vadv2_50pct.py \
    work_dirs/e2e_vadv2_50pct/epoch_20.pth \
    8
```

**Time:** ~30 minutes on 8 GPUs

See [AlgEngine Usage Guide](algengine_usage.md#evaluation) for details.

### 4. Extract Rare Cases

```bash
conda activate algengine
cd projects/AlgEngine

# Extract failure scenarios
python scripts/rare_case_sampling_by_pdms.py \
    --pdm-result work_dirs/e2e_vadv2_50pct/navtest.csv \
    --base-split configs/navsim_splits/navtest_split/navtest.yaml \
    --output-dir configs/navsim_splits/navtest_split/rare_cases
```

See [AlgEngine Usage Guide](algengine_usage.md#rare-case-extraction) for details.

### 5. Run Closed-Loop Simulation (SimEngine + AlgEngine)

```bash
cd WorldEngine
export WORLDENGINE_ROOT=$(pwd)

cd projects/SimEngine
# Run distributed testing
bash scripts/run_ray_distributed_testing.sh \
    $WORLDENGINE_ROOT/projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py \
    $WORLDENGINE_ROOT/projects/AlgEngine/work_dirs/e2e_vadv2_50pct/epoch_20.pth \
    e2e_vadv2_50pct \
    navtrain_ep_per1 \
    NR
```

See [SimEngine Usage Guide](simengine_usage.md#distributed-testing) for details.

### 6. Fine-Tune on Rare Cases (AlgEngine)

```bash
conda activate algengine
cd projects/AlgEngine

# Fine-tune with RL on rare cases
./scripts/e2e_dist_train.sh \
    configs/worldengine/e2e_vadv2_50pct_rlft_rare_log.py \
    8 \
    work_dirs/e2e_vadv2_50pct/epoch_20.pth
```


See [AlgEngine Usage Guide](algengine_usage.md#fine-tuning) for details.

---

## Summary

You've learned how to:
- ✅ Run quick tests with pre-trained models
- ✅ Understand simulation outputs and metrics
- ✅ Customize test parameters
- ✅ Navigate to detailed subsystem guides

**Next:** Choose your path:
- 🎮 **Want to run more simulations?** → [SimEngine Usage Guide](simengine_usage.md)
- 🧠 **Want to train/posttrain your own models?** → [AlgEngine Usage Guide](algengine_usage.md)
