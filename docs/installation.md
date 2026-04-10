# Installation Guide

WorldEngine consists of two main subsystems that require **separate conda environments** due to different Python version requirements:

- **SimEngine** (`simengine` env, Python 3.9) - Closed-loop simulation with photorealistic rendering
- **AlgEngine** (`algengine` env, Python 3.9) - End-to-end model training and evaluation

> **Note:** Scene Reconstruction uses the same environment as SimEngine since it's based on MTGS and shares dependencies.

---

## System Requirements

### Hardware Requirements

**Minimum:**
- GPU: NVIDIA GPU with 8GB VRAM (e.g., RTX 2080)
- RAM: 32GB
- Storage: 500GB SSD
- CPU: 8 cores

**Recommended:**
- GPU: NVIDIA GPU with 24GB+ VRAM (e.g., RTX 3090, A100)
- RAM: 64GB+
- Storage: 5TB+ SSD
- CPU: 16+ cores

### Software Requirements

- **OS:** Linux (Ubuntu 20.04/22.04 recommended)
- **CUDA:** 11.8
- **Conda/Miniconda:** Latest version

---

## Environment 1: SimEngine (`simengine`)

This environment is used for:
- Closed-loop simulation
- Photorealistic rendering
- Behavior world model (coming soon)

### Step-by-Step Installation

#### 1. Create Conda Environment

```bash
conda create --name simengine python=3.9 -y
conda activate simengine
```

#### 2. Install CUDA Toolkit

```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
```

#### 3. Install PyTorch

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**Verify PyTorch installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
# Expected: PyTorch: 2.0.1+cu118, CUDA: True
```

#### 4. Install gsplat (Gaussian Splatting Library)

```bash
pip install ninja   # For build acceleration
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0 \
--no-build-isolation
```

#### 5. Install SimEngine Dependencies

```bash
cd projects/SimEngine
pip install -r requirements.txt
```

#### 6. Verify Installation

```bash
conda activate simengine
python -c "
import torch
import ray
import hydra
import gsplat
print('✓ All SimEngine dependencies OK')
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
"
```

---

## Environment 2: AlgEngine (`algengine`)

This environment is used for:
- End-to-end model training
- End-to-end model testing
- Fine-tuning with rare cases

### Step-by-Step Installation

#### 1. Create Conda Environment

```bash
conda create --name algengine python=3.9 -y
conda activate algengine
```

#### 2. Install PyTorch

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**Verify PyTorch:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

#### 3. Install MMCV (Build from Source - Required!)

MMCV must be built from source to include custom CUDA operators:

```bash
# Clone MMCV repository
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.6.2

# Build with custom ops (this will take 10-15 minutes)
# You may downgrade setuptools if errors occur. (Recommend 75.1.0)
MMCV_WITH_OPS=1 pip install -v -e .

# Verify installation
python .dev_scripts/check_installation.py
cd ..
```

**Verify MMCV:**
```bash
python -c "import mmcv; print(f'MMCV: {mmcv.__version__}')"
# Expected: MMCV: 1.6.2
```

#### 4. Install OpenMMLab Ecosystem

```bash
pip install mmcls==0.25.0
pip install mmdet==2.25.3
pip install mmdet3d==1.0.0rc6
pip install mmsegmentation==0.29.1
```

**Verify MMDetection3D:**
```bash
python -c "import mmdet3d; print(f'MMDetection3D: {mmdet3d.__version__}')"
# Expected: MMDetection3D: 1.0.0rc6
```

#### 5. Install AlgEngine Dependencies

```bash
cd projects/AlgEngine
pip install -r requirements.txt
pip install shapely==2.0.4
```

#### 6. Verify Installation

```bash
conda activate algengine
python -c "
import torch
import mmcv
import mmdet
import mmdet3d
import numpy
import hydra
print('✓ All AlgEngine dependencies OK')
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ MMCV {mmcv.__version__}')
print(f'✓ MMDetection3D {mmdet3d.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
"
```
---

## Environment Variables

we rely on NAVSIM devkit, please git clone and switch to `v1.1` branch:
```bash
git clone -b v1.1 https://github.com/autonomousvision/navsim.git
```
Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# AlgEngine Environment Variables
export NAVSIM_DEVKIT_ROOT="/path/to/your/navsim/v1.1"
export WORLDENGINE_ROOT="/path/to/WorldEngine"
export SIMENGINE_ROOT="${WORLDENGINE_ROOT}/projects/SimEngine"
export ALGENGINE_ROOT="${WORLDENGINE_ROOT}/projects/AlgEngine"
export NUPLAN_MAPS_ROOT="${WORLDENGINE_ROOT}/data/raw/nuplan/maps"

PYTHONPATH=$WORLDENGINE_ROOT:$SIMENGINE_ROOT:$ALGENGINE_ROOT:$NAVSIM_DEVKIT_ROOT:$PYTHONPATH
```

Apply changes:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

---

## Environment Summary

| Feature | SimEngine (`simengine`) | AlgEngine (`algengine`) |
|---------|---------------------------|-------------------------|
| **Python Version** | 3.9 | 3.9 |
| **PyTorch** | 2.0.1+cu118 | 2.0.1+cu118 |
| **Key Libs** | gsplat, ray | MMCV, MMDet3D, MMCls |
| **Use Cases** | Simulation, Rendering, BWM | Training, Evaluation |
| **Disk Space** | ~10 GB | ~15 GB |
| **Install Time** | ~30 min | ~45 min (MMCV build) |


