# AlgEngine Configuration Guide

This guide provides a comprehensive reference for all configuration files under `projects/AlgEngine/configs/worldengine/`. Each config defines a complete training or evaluation experiment, covering model architecture, data pipeline, optimizer, and training schedule.

## Table of Contents

- [Configuration Overview](#configuration-overview)
- [Base IL Training Configs](#base-il-training-configs)
- [RL Fine-Tuning Configs (RLFT)](#rl-fine-tuning-configs-rlft)
- [IL Fine-Tuning Config (ILFT)](#il-fine-tuning-config-ilft)
- [Detection/Tracking Config](#detectiontracking-config)
- [Common Parameters Reference](#common-parameters-reference)
- [Config Comparison Table](#config-comparison-table)

---

## Configuration Overview

All configs inherit from `configs/_base_/default_runtime.py` and are organized into four categories:

```
configs/worldengine/
│
├── Base IL Training (data scaling experiments)
│   ├── e2e_vadv2_13pct.py
│   ├── e2e_vadv2_25pct.py
│   ├── e2e_vadv2_50pct.py          ← reference config
│   ├── e2e_vadv2_60pct.py
│   ├── e2e_vadv2_70pct.py
│   ├── e2e_vadv2_80pct.py
│   ├── e2e_vadv2_90pct.py
│   └── e2e_vadv2_100pct.py
│
├── RL Fine-Tuning (RLFT)
│   ├── e2e_vadv2_50pct_rlft_common_log.py   ← real logs, normal data only
│   ├── e2e_vadv2_50pct_rlft_rare_log.py     ← real logs, includes hard cases
│   ├── e2e_vadv2_50pct_rlft_rare_rollout.py ← synthetic rollout (single source)
│   ├── e2e_vadv2_50pct_rlft_rare_rollout_bwm.py ← synthetic rollout (multi-source)
│   └── e2e_vadv2_50pct_rlft_rare_syn_replay.py  ← synthetic replay, no real failures
│
├── IL Fine-Tuning (ILFT)
│   └── e2e_vadv2_50pct_ilft_rare_log.py     ← IL-only on rare cases
│
└── Detection/Tracking
    └── track_map_nuplan_r50_navtrain_50pct.py ← perception-only (UniAD)
```

---

## Base IL Training Configs

**Files:** `e2e_vadv2_{13,25,50,60,70,80,90,100}pct.py`

These configs train the NAVFormer model end-to-end with Imitation Learning on different percentages of the NavSim training set. They are used for **data scaling experiments**.

### Key Characteristics

- **Model:** `NAVFormer` with `TrajScoringHead` (standard IL planning head)
- **Backbone:** ResNet-50 (caffe style), frozen (`freeze_img_backbone=True`)
- **Other modules:** img_neck, BN, BEV encoder are **NOT** frozen
- **Pretrained weights:** `track_map_nuplan_r50_navtrain_50pct_bs1x8.pth`
- **Dataset:** `NavSimOpenSceneE2E`
- **Epochs:** 8, with evaluation at epoch 8

### Data Percentage Variants

The **only** difference between these configs is `nav_filter_path_train`:

| Config | Training Split |
|--------|---------------|
| `e2e_vadv2_13pct.py` | `navtrain_13pct.yaml` |
| `e2e_vadv2_25pct.py` | `navtrain_25pct.yaml` |
| `e2e_vadv2_50pct.py` | `navtrain_50pct.yaml` |
| `e2e_vadv2_60pct.py` | `navtrain_60pct.yaml` |
| `e2e_vadv2_70pct.py` | `navtrain_70pct.yaml` |
| `e2e_vadv2_80pct.py` | `navtrain_80pct.yaml` |
| `e2e_vadv2_90pct.py` | `navtrain_90pct.yaml` |
| `e2e_vadv2_100pct.py` | `navtrain.yaml` (full set) |

### Planning Head (IL)

```python
planning_head=dict(
    type='TrajScoringHead',       # standard IL head
    reward_shaping=False,         # no reward shaping in IL
    num_poses=40,                 # trajectory vocabulary size
    vocab_path="data/alg_engine/test_8192_kmeans.npy",  # K-means trajectory clusters
    num_commands=4,               # driving command types
    # No LoRA, no RL loss
)
```

---

## RL Fine-Tuning Configs (RLFT)

All RLFT configs fine-tune a **pretrained IL model** (`e2e_vadv2_50pct_ep8.pth`) using Reinforcement Learning, with LoRA adapters to keep parameter-efficient.

### Common RLFT Changes (vs Base IL)

| Parameter | Base IL | RLFT |
|-----------|---------|------|
| `planning_head.type` | `TrajScoringHead` | `TrajScoringHeadRL` |
| `lora_finetuning` | N/A | `True` |
| `freeze_img_backbone` | `True` | `True` |
| `freeze_img_neck` | `False` | `True` |
| `freeze_bn` | `False` | `True` |
| `freeze_bev_encoder` | `False` | `True` |
| `reward_shaping` | `False` | `True` |
| `rl_finetuning` | N/A | `True` |
| `importance_sampling` | N/A | `True` |
| `orig_IL` | N/A | `True` (keep IL loss component) |
| `load_from` | perception ckpt | IL e2e ckpt |
| train pipeline keys | no `fail_mask` | includes `fail_mask` |

### RL Loss Weights

All RLFT configs use the same loss configuration:

```python
rl_loss_weight=dict(
    bce=0.0,       # binary cross-entropy (disabled)
    rank=0.0,      # ranking loss (disabled)
    PG=0.01,       # policy gradient loss
    entropy=1.0    # entropy regularization
)
```

### LoRA Configuration

LoRA adapters are applied in the planning head's transformer decoder:

```python
use_lora=True,          # in planning_head
trans_use_lora=True,    # in planning_head
# In MotionTransformerAttentionLayer:
use_lora=True, lora_rank=16
# In MotionDeformableAttention:
use_lora=True, lora_rank=16
```

### RLFT Variant Comparison

| Config | Data Source | Dataset Type | `normal_only` | `hard_case_no_imi` | Special Fields |
|--------|-----------|-------------|--------------|-------------------|----------------|
| `rlft_common_log` | Real logs | `FineTune` | `True` | `False` (default) | - |
| `rlft_rare_log` | Real logs | `FineTune` | `False` (default) | `True` | - |
| `rlft_rare_rollout` | Synthetic | `FineTuneSynthetic` | - | - | `folder_name`, `customized_filter="v1"` |
| `rlft_rare_rollout_bwm` | Synthetic (multi) | `FineTuneSynthetic` | - | - | `folder_name` (4 sources), `customized_filter="v1"` |
| `rlft_rare_syn_replay` | Synthetic replay | `FineTuneSynthetic` | - | - | `customized_filter="v2"`, `include_real_failures=False` |

---

### rlft_common_log

**Purpose:** Ablation baseline -- RL fine-tuning using only **normal** (non-failure) data.

Key fields:
- `normal_only=True` -- dataset only loads normal samples, excluding hard/failure cases
- `hard_case_no_imi=False` (default) -- N/A since there are no hard cases in the data

**Downstream effect:** In `navsim_openscene_finetuning.py`, when `normal_only=True`, the dataset's `index_map` only contains normal samples. No failure cases from `finetune_yaml` enter the training loop.

---

### rlft_rare_log

**Purpose:** Full RLFT on rare/hard failure scenarios from **real driving logs**.

Key fields:
- `normal_only=False` (default) -- dataset mixes normal + failure samples
- `hard_case_no_imi=True` -- for hard cases (`fail_mask != 0`), imitation learning loss is zeroed out; only RL losses (PG + entropy) are applied

**Downstream effect:** In `traj_scoring_head_RL.py`, when `hard_case_no_imi=True`, the imitation mask is set to 0 for all samples where `fail_mask != 0` (both real failures and synthetic cases). This forces the model to learn from RL rewards rather than imitating expert behavior on difficult scenarios.

---

### rlft_rare_rollout

**Purpose:** RLFT using **synthetic rollout** trajectories generated from a single source.

Key fields:
- `train_dataset_type = "NavSimOpenSceneE2EFineTuneSynthetic"` -- loads synthetic trajectory data
- `synthetic_folder_names = ["e2e_vadv2_50pct_navtrain_50pct_collision_NR_250911"]` -- single synthetic source
- `customized_filter="v1"` -- filtering strategy for synthetic data
- `folder_name=synthetic_folder_names` -- passed to dataset for loading

---

### rlft_rare_rollout_bwm

**Purpose:** Extended version of `rare_rollout` with **multiple augmented synthetic sources** (backward-masked trajectories).

Key fields:
- `synthetic_folder_names` -- 4 sources covering collision, ego progress, and off-road scenarios:
  ```python
  synthetic_folder_names = [
      "e2e_vadv2_50pct_navtrain_50pct_collision_NR_250911",
      "e2e_vadv2_50pct_aug_navtrain_50pct_collision_NR_250928",
      "e2e_vadv2_50pct_aug_navtrain_50pct_ep_1pct_NR_250928",
      "e2e_vadv2_50pct_aug_navtrain_50pct_offroad_NR_250928",
  ]
  ```
  PS: You need to produce your own rollouts and organize them into `data/alg_engine/openscene-synthetic`. See [SimEngine Usage Guide - Rollout Scripts](simengine_usage.md#rollout-scripts) for how to generate augmented rollouts.
- `customized_filter="v1"` -- same filtering as `rare_rollout`

---

### rlft_rare_syn_replay

**Purpose:** RLFT with synthetic replay data, **excluding real failure cases**.

Key fields:
- `customized_filter="v2"` -- different filtering strategy from v1
- `include_real_failures=False` -- explicitly excludes real failure data, training only on synthetic replays
- Uses same multi-source `synthetic_folder_names` as `rare_rollout_bwm` but with single source

---

## IL Fine-Tuning Config (ILFT)

**File:** `e2e_vadv2_50pct_ilft_rare_log.py`

**Purpose:** Fine-tune with **Imitation Learning only** (no RL) on rare failure cases. Serves as an ablation to compare against RLFT approaches.

### Key Differences from RLFT

| Parameter | RLFT | ILFT |
|-----------|------|------|
| `rl_finetuning` | `True` | `False` |
| `reward_shaping` | `True` | `False` |
| `rl_loss_weight` | `dict(bce=0, rank=0, PG=0.01, entropy=1.0)` | N/A |
| `orig_IL` | `True` | N/A |
| `evaluation.interval` | 8 | 1 (every epoch) |

The model still uses `TrajScoringHeadRL` as the head type (for code compatibility) and LoRA adapters, but all RL-specific losses are disabled. The model learns purely from imitation.

---

## Detection/Tracking Config

**File:** `track_map_nuplan_r50_navtrain_50pct.py`

**Purpose:** Train perception-only model for **object detection + map segmentation** (no planning).

### Key Differences from E2E Configs

| Parameter | E2E (NAVFormer) | Detection (UniAD) |
|-----------|----------------|-------------------|
| `model.type` | `NAVFormer` | `UniAD` |
| `dataset_type` | `NavSimOpenSceneE2E` | `NavSimOpenSceneE2EDet` |
| `queue_length` | 4 | 3 |
| `total_epochs` | 8 | 40 |
| `samples_per_gpu` | 2 | 1 |
| `freeze_*` | varies | all `False` (full training) |
| `planning_head` | yes | no |
| `seg_head` | no | yes (`PansegformerHead`) |
| `eval_mod` | `[]` | `['det', 'map']` |
| `load_from` | varies | `bevformerv2-r50-t1-base_epoch_48.pth` |

Additional features in detection config:
- **Segmentation head** (`PansegformerHead`): lane detection and map segmentation
- **3D annotations**: `gt_bboxes_3d`, `gt_labels_3d`, `gt_lane_labels`, `gt_lane_bboxes`, `gt_lane_masks`
- **Image scaling**: `RandomScaleImageMultiViewImage` with scale 0.5
- **Loading**: `LoadMultiViewImageFromFilesInCeph` (vs `LoadMultiViewImageFromFilesWithDownsample` in E2E)

---

## Common Parameters Reference

### Spatial & BEV Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| `point_cloud_range` | `[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]` | 3D detection range (meters) |
| `voxel_size` | `[0.2, 0.2, 8]` | Voxel grid resolution |
| `bev_h_, bev_w_` | `200, 200` | BEV feature map size |
| `patch_size` | `[102.4, 102.4]` | Spatial patch size for BEV |

### Temporal Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| `queue_length` | 4 (E2E) / 3 (Det) | Number of frames per sequence |
| `past_steps` | 3 | Historical tracking steps |
| `fut_steps` | 4 | Future prediction steps |
| `planning_steps` | 8 | Planning horizon steps |

### Model Architecture

| Parameter | Value | Description |
|-----------|-------|-------------|
| `_dim_` | 256 | Embedding dimension |
| `_ffn_dim_` | 512 | Feed-forward network dimension |
| `_num_levels_` | 4 | Multi-scale feature levels |
| `num_query` | 900 | Number of detection queries |
| `num_cams` | 8 | Number of camera views |

### Tracking (QIM & Memory Bank)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `qim_type` | `QIMBase` | Query interaction module type |
| `fp_ratio` | 0.3 | False positive ratio |
| `random_drop` | 0.1 | Random query drop rate |
| `memory_bank_len` | 4 | Frames to keep in memory bank |

### Optimizer & Schedule

| Parameter | Value | Description |
|-----------|-------|-------------|
| `optimizer.type` | `AdamW` | Optimizer type |
| `optimizer.lr` | 2e-4 | Base learning rate |
| `img_backbone lr_mult` | 0.1 | Backbone learning rate multiplier |
| `weight_decay` | 0.01 | Weight decay |
| `lr_config.policy` | `CosineAnnealing` | LR schedule |
| `warmup_iters` | 500 | Linear warmup iterations |
| `grad_clip.max_norm` | 35 | Gradient clipping threshold |

### Freeze Strategy

| Parameter | Base IL | RLFT/ILFT | Detection |
|-----------|---------|-----------|-----------|
| `freeze_img_backbone` | `True` | `True` | `False` |
| `freeze_img_neck` | `False` | `True` | `False` |
| `freeze_bn` | `False` | `True` | `False` |
| `freeze_bev_encoder` | `False` | `True` | `False` |

---

## Config Comparison Table

### All E2E Configs at a Glance

| Config | Model | Head | Dataset | Freeze | LoRA | RL | Epochs | Eval Interval | Pretrained |
|--------|-------|------|---------|--------|------|----|--------|--------------|------------|
| `e2e_vadv2_Xpct` | NAVFormer | TrajScoringHead | NavSimOpenSceneE2E | backbone only | No | No | 8 | 8 | perception ckpt |
| `rlft_common_log` | NAVFormer | TrajScoringHeadRL | FineTune | all except planning | Yes | Yes | 8 | 8 | IL e2e ckpt |
| `rlft_rare_log` | NAVFormer | TrajScoringHeadRL | FineTune | all except planning | Yes | Yes | 8 | 8 | IL e2e ckpt |
| `rlft_rare_rollout` | NAVFormer | TrajScoringHeadRL | FineTuneSynthetic | all except planning | Yes | Yes | 8 | 8 | IL e2e ckpt |
| `rlft_rare_rollout_bwm` | NAVFormer | TrajScoringHeadRL | FineTuneSynthetic | all except planning | Yes | Yes | 8 | 8 | IL e2e ckpt |
| `rlft_rare_syn_replay` | NAVFormer | TrajScoringHeadRL | FineTuneSynthetic | all except planning | Yes | Yes | 8 | 8 | IL e2e ckpt |
| `ilft_rare_log` | NAVFormer | TrajScoringHeadRL | FineTune | all except planning | Yes | No | 8 | 1 | IL e2e ckpt |
| `track_map` | UniAD | N/A (det only) | NavSimOpenSceneE2EDet | none | No | No | 40 | 40 | BEVFormerV2 ckpt |

### Failure Data Sources

The `finetune_yaml` files define which failure scenarios are included:

| YAML File | Failure Type |
|-----------|-------------|
| `navtrain_50pct_collision.yaml` | Collision scenarios |
| `navtrain_50pct_ep_1pct.yaml` | Bottom 1% ego progress (near-stationary) |
| `navtrain_50pct_off_road.yaml` | Off-road / drivable area violations |
