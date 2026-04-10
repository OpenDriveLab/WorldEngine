# AlgEngine Usage Guide

This guide covers how to use AlgEngine for training end-to-end autonomous driving models, running evaluations, extracting rare cases, and fine-tuning. AlgEngine is built on MMDetection3D and supports UniAD, VADv2, and HydraMDP architectures.

## Table of Contents

- [Quick Reference](#quick-reference)
- [Training](#training)
- [Evaluation](#evaluation)
- [Rare Case Extraction](#rare-case-extraction)
- [Fine-Tuning](#fine-tuning)
- [Configuration](#configuration)
- [Model Architectures](#model-architectures)
- [Troubleshooting](#troubleshooting)

---

## Quick Reference

```bash
cd projects/AlgEngine
# Training (8 GPUs)
./scripts/e2e_dist_train.sh <config> <num_gpus> [resume_checkpoint]

# Open-loop evaluation
./scripts/e2e_dist_eval.sh <config> <checkpoint> <num_gpus>

# Rare case extraction
python scripts/rare_case_sampling_by_pdms.py \
    --pdm-result <csv_file> \
    --base-split <yaml_file> \
    --output-dir <output_dir>

# Closed-loop evaluation
bash scripts/run_ray_distributed_testing.sh <config> <checkpoint> <model_name> <data_type> <react_type>
```

---

## Training

### Prerequisites

Before training, ensure:
- ✅ AlgEngine environment installed (`algengine` conda env)
- ✅ Data prepared (see [Data Organization](data_organization.md))
- ✅ Pre-trained backbone weights downloaded

### Training from Scratch

Train a model on 50% of the training data:

```bash
conda activate algengine
cd projects/AlgEngine

# Train VADv2 with 50% data (8 GPUs)
./scripts/e2e_dist_train.sh configs/worldengine/e2e_vadv2_50pct.py 8
```

**Arguments:**
1. `<config>`: Configuration file path
2. `<num_gpus>`: Number of GPUs to use
3. `[resume_checkpoint]` (optional): Checkpoint to resume from

### Training with 100% Data

```bash
./scripts/e2e_dist_train.sh configs/worldengine/e2e_vadv2_100pct.py 8
```

### Resume Training

Resume from a checkpoint:

```bash
./scripts/e2e_dist_train.sh \
    configs/worldengine/e2e_vadv2_50pct.py \
    8 \
    work_dirs/e2e_vadv2_50pct/latest.pth
```

**Auto-resume:** If `latest.pth` exists in `work_dirs/`, training will auto-resume.

### Monitor Training

```bash
# Watch training log
tail -f work_dirs/e2e_vadv2_50pct/logs/train.*

# TensorBoard (if enabled)
tensorboard --logdir work_dirs/e2e_vadv2_50pct/tf_logs
```

**Key metrics to monitor:**
- `loss`: Total training loss (should decrease)
- `loss_planning`: Planning loss
- `loss_track`: Tracking loss
- `ade_4s`: Average displacement error at 4 seconds
- `fde_4s`: Final displacement error at 4 seconds

### Training Output

```
work_dirs/e2e_vadv2_50pct/
├── e2e_vadv2_50pct.py          # Config backup
├── logs/
│   └── train.26040614*         # Training logs
├── epoch_1.pth                 # Checkpoints
├── epoch_2.pth
...
├── epoch_20.pth
└── latest.pth                  # Symlink to latest checkpoint
```

---

## Evaluation

### Open-Loop Evaluation

Evaluate model predictions against ground truth trajectories.

#### Full Test Set Evaluation

```bash
conda activate algengine
cd projects/AlgEngine

# Evaluate on navtest (8 GPUs)
./scripts/e2e_dist_eval.sh \
    configs/worldengine/e2e_vadv2_50pct.py \
    work_dirs/e2e_vadv2_50pct/epoch_20.pth \
    8
```

**Output:**
```
work_dirs/e2e_vadv2_50pct/
└── navtest.csv                 # Evaluation results
```

#### Rare Navtest Cases Only

Evaluate on known rare navtest cases:

```bash
./scripts/e2e_dist_eval_navtest_failures.sh \
    configs/worldengine/e2e_vadv2_50pct.py \
    work_dirs/e2e_vadv2_50pct/epoch_20.pth \
    8
```

**Output:**
```
work_dirs/e2e_vadv2_50pct/
└── navtest_failures.csv        # rare navtest cases only
```

#### Understanding Evaluation Metrics

Open-loop metrics CSV format:

```csv
token,ade_4s,fde_4s,no_at_fault_collisions,drivable_area_compliance,ego_progress,comfort,score
abc123,0.42,0.85,1.0,0.95,0.88,0.92,0.89
...
```

**Key metrics:**
- `ade_4s`: Average trajectory error over 4 seconds (meters, lower is better)
- `fde_4s`: Final position error at 4 seconds (meters, lower is better)
- `no_at_fault_collisions`: Collision avoidance rate (0-1, higher is better)
- `drivable_area_compliance`: Stay in drivable area (0-1, higher is better)
- `ego_progress`: Route completion (0-1, higher is better)
- `comfort`: Comfort metric (0-1, higher is better)
- `score`: Overall PDM score (0-1, higher is better)

### Closed-Loop Evaluation

Evaluate model in simulation (requires SimEngine).

See [SimEngine Usage Guide](simengine_usage.md#testing-scripts) for:
- Single-GPU testing
- Multi-GPU distributed testing
- Reactive vs non-reactive modes

**Quick example:**

```bash
cd projects/AlgEngine

bash scripts/run_ray_distributed_testing.sh \
    $WORLDENGINE_ROOT/projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py \
    $WORLDENGINE_ROOT/projects/AlgEngine/work_dirs/e2e_vadv2_50pct/epoch_20.pth \
    e2e_vadv2_50pct_epoch20 \
    navtest_failures \
    NR
```

---

## Rare Case Extraction

Extract failure scenarios from evaluation results for targeted fine-tuning.

### Basic Extraction

```bash
conda activate algengine
cd projects/AlgEngine

python scripts/rare_case_sampling_by_pdms.py \
    --pdm-result work_dirs/e2e_vadv2_50pct/navtest.csv \
    --base-split configs/navsim_splits/navtest_split/navtest.yaml \
    --output-dir configs/navsim_splits/navtest_split/e2e_vadv2_50pct_rare
```

**Arguments:**
- `--pdm-result`: CSV file with evaluation metrics
- `--base-split`: Base scenario split YAML file
- `--output-dir`: Directory to save extracted split files

### Extracted Splits

The script generates three rare case split files:

```
configs/navsim_splits/navtest_split/e2e_vadv2_50pct_rare/
├── navtest_collision.yaml         # Collision scenarios
├── navtest_off_road.yaml          # Off-road scenarios
└── navtest_ep_1pct.yaml           # Low ego-progress (bottom 1%)
```

### Using Custom Thresholds

Edit the script to customize:

```python
# In rare_case_sampling_by_pdms.py

# Change collision threshold
collision_scenarios = df[df['no_at_fault_collisions'] < 0.95]  # From 1.0

# Change ego-progress percentile
ep_threshold = df['ego_progress'].quantile(0.05)  # From 0.01 (1% -> 5%)
```

### Verify Extracted Scenarios

```bash
# Check how many scenarios were extracted
wc -l configs/navsim_splits/navtest_split/e2e_vadv2_50pct_rare/*.yaml

# View first few scenarios
head -20 configs/navsim_splits/navtest_split/e2e_vadv2_50pct_rare/navtest_collision.yaml
```

---

## Fine-Tuning

Fine-tune a trained model on rare cases using reinforcement learning.

### Prerequisites: Generating Rollouts with SimEngine

**Important:** Rollout data must be generated by SimEngine before fine-tuning. This involves:

1. **Run SimEngine Rollout** to generate trajectory data:
   ```bash
   # Multi-GPU distributed rollout (recommended for large datasets)
   export WORLDENGINE_ROOT=/path/to/WorldEngine
   cd projects/SimEngine
   bash scripts/run_ray_distributed_rollout.sh \
       $WORLDENGINE_ROOT/projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py \
       $WORLDENGINE_ROOT/data/alg_engine/ckpts/e2e_vadv2_50pct_ep8.pth \
       e2e_vadv2_50pct \
       navtrain_50pct_collision \ 
       navtrain
   ```

2. **Rollout Output** is saved to:
   ```
   experiments/closed_loop_exps/e2e_vadv2_50pct/navtrain_NR/
   └── WE_output/
       └── openscene_format/
           ├── sensor_blobs/        # Camera images, LiDAR
           ├── meta_datas/          # Per-scenario metadata
           ├── pdms_pkl/            # Metric pdms pkl
           └── all_scenes_pdm_averages_NR.csv
   ```

3. **Reorganize to AlgEngine Format** (creates `openscene-synthetic` dataset):
   ```bash
   conda activate simengine
   
   cd projects/SimEngine
   python scripts/export_simulation_data.py \
       --test_path experiments/closed_loop_exps/e2e_vadv2_50pct/navtrain_NR \
       --appendix 260406  # Date suffix for versioning, Default None
   ```
   
   **Output location:** `data/alg_engine/openscene-synthetic/`

4. **Verify Data Structure:**
   ```bash
   data/alg_engine/openscene-synthetic/
   ├── sensor_blobs/              # Replayed scenario sensor data
   ├── meta_datas/                # Metadata
   └── pdms_pkl/                  # Metric pdms pkl
   ```

For detailed SimEngine usage, see [SimEngine Usage Guide](simengine_usage.md#rollout-scripts).

### RL-Based Fine-Tuning

```bash
conda activate algengine
cd projects/AlgEngine

# Fine-tune on extracted rare cases (8 GPUs)
./scripts/e2e_dist_train.sh \
    configs/worldengine/e2e_vadv2_50pct_rlft_rare_log.py \
    8 \
    work_dirs/e2e_vadv2_50pct/epoch_20.pth
```

**Arguments:**
1. Config with `_rlft_rare_log` suffix (uses rare case splits)
2. Number of GPUs
3. Base checkpoint to fine-tune from


### Fine-Tuning Configuration

The `_rlft_rare_log` config typically includes:

```python
# configs/worldengine/e2e_vadv2_50pct_rlft_rare_log.py

# Use rare case splits
data = dict(
    train=dict(
        ann_file='merged_infos_navformer/nuplan_openscene_navtrain.pkl',
        scenario_filter=[
            'configs/navsim_splits/navtrain_split/e2e_vadv2_50pct_ep8/navtrain_50pct_collision.yaml',
            'configs/navsim_splits/navtrain_split/e2e_vadv2_50pct_ep8/navtrain_50pct_off_road.yaml',
            'configs/navsim_splits/navtrain_split/e2e_vadv2_50pct_ep8/navtrain_50pct_ep_1pct.yaml',
        ]
    )
)

# RL training settings
optimizer = dict(type='AdamW', lr=5e-5)  # Lower learning rate
total_epochs = 8  # Fewer epochs for fine-tuning
```

### Fine-Tuning Output

```
work_dirs/e2e_vadv2_50pct_rlft_rare_log/
├── e2e_vadv2_50pct_rlft_rare_log.py
├── logs/
│   └── train.*
├── epoch_1.pth
...
└── epoch_8.pth
```

### Evaluate Fine-Tuned Model

```bash
cd projects/AlgEngine
# Open-loop evaluation
./scripts/e2e_dist_eval.sh \
    configs/worldengine/e2e_vadv2_50pct_rlft_rare_log.py \
    work_dirs/e2e_vadv2_50pct_rlft_rare_log/epoch_8.pth \
    8

# Closed-loop evaluation
bash scripts/run_ray_distributed_testing.sh \
    $WORLDENGINE_ROOT/projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct_rlft_rare_log.py \
    $WORLDENGINE_ROOT/projects/AlgEngine/work_dirs/e2e_vadv2_50pct_rlft_rare_log/epoch_8.pth \
    e2e_vadv2_50pct_rlft \
    navtest_failures \
    NR
```

---

## Configuration

AlgEngine uses hierarchical configuration with MMDetection3D. For a detailed reference of all config parameters, variants, and their relationships, see the [Configuration Guide](config_guide.md).

### Configuration Hierarchy

```
configs/
├── _base_/
│   └── default_runtime.py              # Base runtime settings
├── worldengine/
│   ├── e2e_vadv2_50pct.py              # 50% data training
│   ├── e2e_vadv2_100pct.py             # 100% data training
│   ├── e2e_vadv2_50pct_rlft_rare_log.py  # Rare case fine-tuning
│   └── ...
└── navsim_splits/
    ├── navtrain_split/
    │   ├── navtrain.yaml               # Full training set
    │   ├── navtrain_50pct.yaml         # 50% subset
    │   └── e2e_vadv2_50pct_ep8/        # Rare case splits
    │       ├── navtrain_50pct_collision.yaml
    │       ├── navtrain_50pct_off_road.yaml
    │       └── navtrain_50pct_ep_1pct.yaml
    └── navtest_split/
        ├── navtest.yaml                # Full test set
        └── navtest_failures.yaml       # Failure subset
```

### Key Configuration Parameters

```python
# Model architecture
model = dict(
    type='VADv2',  # or 'UniAD', 'HydraMDP'
    num_query=900,
    num_classes=7,
    planning_steps=8,
    img_backbone=dict(type='ResNet50', ...),
    img_neck=dict(type='FPN', ...),
)

# BEV configuration
bev_h_, bev_w_ = 200, 200
patch_size = [102.4, 102.4]  # Physical range (meters)

# Input modality
input_modality = dict(
    use_lidar=False,
    use_camera=True,  # 8 cameras
    use_radar=False,
    use_external=True  # CAN bus
)

# Training
total_epochs = 20
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
)

# Data
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        ann_file='merged_infos_navformer/nuplan_openscene_navtrain.pkl',
        scenario_filter='configs/navsim_splits/navtrain_split/navtrain_50pct.yaml',
    ),
    val=dict(
        ann_file='merged_infos_navformer/nuplan_openscene_navtest.pkl',
        scenario_filter='configs/navsim_splits/navtest_split/navtest.yaml',
    ),
)
```

### Override Configuration

Override config parameters at runtime:

```bash
./scripts/e2e_dist_train.sh \
    configs/worldengine/e2e_vadv2_50pct.py \
    8 \
    --cfg-options \
    optimizer.lr=1e-4 \
    total_epochs=30 \
    data.samples_per_gpu=2
```

---

## Model Architectures

AlgEngine supports multiple end-to-end autonomous driving architectures.

### VADv2 (Default)

**Features:**
- Vector-based scene representation
- Planning-oriented perception
- Efficient trajectory prediction

**Config:** `configs/worldengine/e2e_vadv2_*.py`

**Best for:** General driving scenarios, fast inference

### UniAD

**Features:**
- Unified perception-prediction-planning
- Multi-task learning
- Strong generalization

**Config:** `configs/worldengine/e2e_uniad_*.py`

**Best for:** Complex scenarios, research

### HydraMDP

**Features:**
- Multi-modal trajectory prediction
- Distribution-aware planning
- Behavior world model integration

**Config:** `configs/worldengine/e2e_hydramdp_*.py`

**Best for:** Safety-critical scenarios, rare cases

### Switching Architectures

```bash
# Train UniAD instead of VADv2
./scripts/e2e_dist_train.sh configs/worldengine/e2e_uniad_50pct.py 8

# Evaluate HydraMDP
./scripts/e2e_dist_eval.sh \
    configs/worldengine/e2e_hydramdp_50pct.py \
    work_dirs/e2e_hydramdp_50pct/epoch_20.pth \
    8
```

---

## Advanced Training

### Multi-Node Training

For very large models or datasets:

```bash
# Node 0 (master)
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=28567
export WORLD_SIZE=16  # Total GPUs
export RANK=0  # Node rank

./scripts/e2e_dist_train.sh configs/worldengine/e2e_vadv2_100pct.py 8

# Node 1 (worker)
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=28567
export WORLD_SIZE=16
export RANK=8

./scripts/e2e_dist_train.sh configs/worldengine/e2e_vadv2_100pct.py 8
```

### Mixed Precision Training

Enable automatic mixed precision (AMP) for faster training:

```python
# In config
fp16 = dict(loss_scale='dynamic')
```

### Gradient Accumulation

For large batch sizes with limited GPU memory:

```python
# In config
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
)

# Set gradient accumulation steps
runner = dict(
    max_epochs=20,
    gradient_accumulation_steps=4,  # Effective batch size = 1 * 8 GPUs * 4 = 32
)
```

---

## Troubleshooting

### Issue 1: CUDA out of memory during training

**Solution:**
```bash
# Reduce batch size
# Edit config: data.samples_per_gpu = 1 (from 2)

# Reduce BEV resolution
# Edit config: bev_h_, bev_w_ = 150, 150 (from 200, 200)

# Use gradient checkpointing
# Edit config: model.img_backbone.with_cp = True
```

### Issue 2: Training loss not decreasing

**Possible causes:**
- Learning rate too high/low
- Data loading issues
- Incorrect pre-trained weights

**Solution:**
```bash
# Check data loading
python tools/analysis_tools/browse_dataset.py configs/worldengine/e2e_vadv2_50pct.py

# Verify pre-trained weights loaded
grep "load checkpoint" work_dirs/*/logs/train.*

# Try different learning rate
./scripts/e2e_dist_train.sh ... --cfg-options optimizer.lr=1e-4
```

### Issue 3: Evaluation hangs

**Solution:**
```bash
# Check if processes are stuck
ps aux | grep python

# Kill stuck processes
pkill -f "test.py"

# Restart evaluation with fewer GPUs
./scripts/e2e_dist_eval.sh ... 4  # Use 4 instead of 8
```

### Issue 4: "ModuleNotFoundError: No module named mmdet3d"

**Solution:**
```bash
# Ensure you're in the right environment
conda activate algengine

# Verify MMCV installation
python -c "import mmcv; print(mmcv.__version__)"

# Reinstall MMDetection3D if needed
pip uninstall mmdet3d -y
pip install mmdet3d==1.0.0rc6
```

### Issue 5: Checkpoint file corrupted

**Solution:**
```bash
# Use a previous checkpoint
./scripts/e2e_dist_train.sh ... work_dirs/*/epoch_18.pth  # Instead of epoch_20

# Or train from scratch
rm work_dirs/e2e_vadv2_50pct/latest.pth
./scripts/e2e_dist_train.sh ...
```

---

## Performance Optimization

### Training Speed

1. **Increase workers:** `data.workers_per_gpu = 8` (if CPU/RAM allows)
2. **Use SSD:** Store data on fast NVMe SSD
3. **Mixed precision:** Enable `fp16 = dict(loss_scale='dynamic')`
4. **Persistent workers:** `data.persistent_workers = True`

### Memory Optimization

1. **Reduce batch size:** `data.samples_per_gpu = 1`
2. **Lower BEV resolution:** `bev_h_, bev_w_ = 150, 150`
3. **Gradient checkpointing:** `model.img_backbone.with_cp = True`
4. **Clear cache:** `torch.cuda.empty_cache()` in code

### Distributed Training Tips

1. **Even GPU allocation:** Use same GPU type across nodes
2. **InfiniBand:** Use high-speed interconnect for multi-node
3. **Shared filesystem:** Use NFS/Lustre for data loading
4. **Monitor network:** Watch for communication bottlenecks

---

## Next Steps

- **Run simulations:** See [SimEngine Usage Guide](simengine_usage.md)
- **Understand evaluation:** See [Quick Start Guide](quick_start.md)

For questions, visit [GitHub Discussions](https://github.com/OpenDriveLab/WorldEngine/discussions).
