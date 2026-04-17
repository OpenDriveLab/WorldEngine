<div align="center">
<img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/WorldEngine/imgs/WE_title.png" width="800px">

# Towards the Era of Post-Training for Physical AI
> *The missing infrastructure for Physical AI post-training in AD. Open-source. Production-validated.*

[![Paper](https://img.shields.io/badge/Paper-Coming_Soon-b31b1b.svg?style=for-the-badge&logo=arxiv)](https://github.com/OpenDriveLab/WorldEngine)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Dataset-ffc107.svg?style=for-the-badge&logo=huggingface)](https://huggingface.co/datasets/OpenDriveLab/WorldEngine)
[![ModelScope](https://img.shields.io/badge/ModelScope-Dataset-orange.svg?style=for-the-badge)](https://www.modelscope.cn/datasets/OpenDriveLab/WorldEngine)
<br>
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/python-3.9-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)

</div>

<div id="top" align="center">
<p align="center">
<img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/WorldEngine/imgs/README_overall.jpg" width="800px" >
</p>
</div>

>  Joint effort by OpenDriveLab at The University of Hong Kong, Huawei Inc. and Shanghai Innovation Institute (SII).

## Table of Contents
- [Highlights](#highlights)
- [News](#news)
- [Benchmark](#benchmark)
  - [Qualitative Results — Closed-Loop Simulation on nuPlan](#qualitative-results--closed-loop-simulation-on-nuplan)
  - [On-Road Deployment — Night Urban Driving](#on-road-deployment--night-urban-driving)
- [System Architecture](#system-architecture)
- [Roadmap](#roadmap)
- [Getting Started](#getting-started)
  - [Documentation Overview](#documentation-overview)
  - [Installation](#installation)
  - [Quick Test](#quick-test)
  - [Deep Dive by Module](#deep-dive-by-module)
    - [SimEngine - Photorealistic Closed-Loop Simulation](#simengine---photorealistic-closed-loop-simulation)
    - [AlgEngine - End-to-End Model Training \& Fine-Tuning](#algengine---end-to-end-model-training--fine-tuning)
    - [Scene Reconstruction - 3D Gaussian Splatting-based method, MTGS](#scene-reconstruction---3d-gaussian-splatting-based-method-mtgs)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Related Resources](#related-resources)

## Highlights

- **WorldEngine** is a post-training framework for Physical AI that systematically addresses the long-tail safety-critical data scarcity problem in autonomous driving.
- **Data-driven long-tail discovery**: Failure-prone scenarios are automatically identified from real-world driving logs by the pre-trained agent itself — no manual design, no synthetic perturbations.
- **Photorealistic interactive simulation** via 3D Gaussian Splatting (3DGS): Each discovered scenario is reconstructed into a fully controllable, real-time-renderable simulation environment with independent dynamic agent manipulation.
- **Behavior-driven scenario generation**: Leverages Behavior World Model (BWM) to generalize and synthesize diverse traffic variations from existing long-tail scenarios, expanding sparse safety-critical events into a dense, learnable distribution.
- **RL-based post-training** on synthesized safety-critical rollouts substantially outperforms scaling pre-training data alone — competitive with a ~10× increase in pre-training data.
- **Production-scale validation**: Deployed on a mass-produced ADAS platform trained on 80,000+ hours of real-world driving logs, reducing simulated collision rate by up to **45.5%** and achieving zero disengagements in a 200 km on-road test.


## News
- **[2026/04/09]** Official dataset released. See [OpenDriveLab/WorldEngine](https://huggingface.co/datasets/OpenDriveLab/WorldEngine) or [OpenDriveLab/WorldEngine (ModelScope)](https://www.modelscope.cn/datasets/OpenDriveLab/WorldEngine)
- **[2026/04/10]** Official code repository established.


## Benchmark

We compare different post-training paradigms on the nuPlan dataset, evaluating on both open-loop and closed-loop metrics across common and rare driving scenarios.

> **Metric notes:**
> **Early stage**. Stable ckpts and corresponding results coming soon.
> - **Open-loop PDMS** is aligned with [NAVSIM v1.1](https://github.com/autonomousvision/navsim) PDM Score. *Common* denotes the standard `navtest` split; *Rare* denotes the `navtest_failures` subset — failure-prone rare-case scenarios extracted from `navtest`.
> - **Closed-loop Success Rate** is defined as the fraction of simulated driving episodes completed without collision or off-road failure.
> - **Closed-loop PDMS*** is the PDM Score obtained via **SimEngine closed-loop testing**, where the planner interacts with reactive agents in simulation under real-time rendering.
>
> **Training notes:**
> - **Rare logs** are failure-prone scenarios automatically extracted from `navtrain` by the pre-trained agent itself (see [Rare Case Extraction](docs/algengine_usage.md#rare-case-extraction)). 
> - **Common logs** are the standard cases in `navtrain`.

| Method | Open-loop PDMS ↑ (common) | Open-loop PDMS ↑ (rare) | Closed-loop Success Rate ↑ | Closed-loop PDMS* ↑ |
|:-------|:-------------------------:|:-----------------------:|:--------------------------:|:--------------------:|
| Base model | 85.62 | 47.15 | 73.61 | 60.28 |
| Supervised fine-tuning on rare logs | 87.03 | 49.68 | 73.26 | 62.26 |
| Post-training on common logs | 86.15 | 51.49 | 64.58 | 56.66 |
| Post-training on rare logs | 89.29 | 62.56 | 74.31 | 62.55 |
| Post-training on rare synthetic replays | 88.01 | 56.62 | 76.39 | 62.11 |
| Post-training on rare rollouts w/o Behaviour WM | 88.99 | 59.69 | 85.07 | 68.29 |
| **Post-training with WorldEngine** | **88.95** | **59.83** | **88.89** | **70.12** |

**Key findings:**
- Post-training on **rare logs** significantly outperforms supervised fine-tuning (62.56 vs 49.68 open-loop rare PDMS), demonstrating the advantage of reward-guided optimization over imitation.
- Post-training on **common logs** provides limited benefit and even degrades closed-loop performance (success rate drops from 73.61% to 64.58%), confirming that long-tail event discovery is essential.
- The full **WorldEngine** pipeline achieves the best closed-loop performance (**88.89%** success rate, **70.12** PDMS*), a **+15.28%** absolute improvement in success rate over the base model.

### Qualitative Results — Closed-Loop Simulation on nuPlan

Each pair shows the **Base model** vs **WorldEngine post-trained model** on the same rare-case scenario. Left: front-camera rendering; Right: BEV trajectory visualization.

<div align="center">
<table>
<tr>
<td><img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/WorldEngine/imgs/nuplan_1.png" width="400px"></td>
<td><img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/WorldEngine/imgs/nuplan_2.png" width="400px"></td>
</tr>
<tr>
<td><img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/WorldEngine/imgs/nuplan_3.png" width="400px"></td>
<td><img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/WorldEngine/imgs/nuplan_4.png" width="400px"></td>
</tr>
</table>
</div>

### On-Road Deployment — Night Urban Driving

Zero disengagements in 200 km on-road testing on a mass-produced ADAS platform.

<div align="center">
<img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/WorldEngine/gif/WE_road_night_01.gif" width="270px">
<img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/WorldEngine/gif/WE_road_night_02.gif" width="270px">
<img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/refs/heads/master/WorldEngine/gif/WE_road_night_03.gif" width="270px">
</div>


## System Architecture

WorldEngine consists of two tightly coupled subsystems:


| Module | Function | Core Technology |
|--------|----------|----------------|
| **[SimEngine](docs/WE_simulation.md)** | Closed-loop simulation with ego & agents | Hydra, Ray, rendering |
| **[AlgEngine](docs/AlgEngine.md)** | End-to-end model training & evaluation | MMDetection3D, UniAD/VADv2/HydraMDP |


## Roadmap

- [x] Core platform integration (SimEngine + AlgEngine)
- [x] Multi-GPU distributed simulation and training
- [x] Rare case extraction and fine-tuning pipeline
- [x] Comprehensive documentation and usage guides
- [x] Hugging Face / ModelScope dataset
- [x] Open-source release (code, data, early pre-trained models)
- [ ] arXiv preprint
- [ ] Behavior World Model integration
- [ ] Stable pre-trained models


## Getting Started

### Documentation Overview

WorldEngine provides comprehensive guides for each stage of your workflow:

| Guide | Purpose | Key Topics |
|-------|---------|------------|
| **[Installation](docs/installation.md)** | Set up both conda environments | Two-environment setup (simengine + algengine), dependencies, troubleshooting |
| **[Data Organization](docs/data_organization.md)** | Prepare datasets and checkpoints | Data structure, Hugging Face/ModelScope downloads, symlinks |
| **[Quick Start](docs/quick_start.md)** | Run your first experiment in 5 min | Quick test tutorial, understanding results, complete pipeline |
| **[SimEngine Usage](docs/simengine_usage.md)** | Master closed-loop simulation | Rollout scripts, distributed testing, configuration, metrics |
| **[AlgEngine Usage](docs/algengine_usage.md)** | Train and fine-tune models | Training from scratch, evaluation, rare case extraction, RL fine-tuning |

### Installation

WorldEngine requires **two separate conda environments** due to different Python requirements.

**Full installation guide:** [docs/installation.md](docs/installation.md)

### Quick Test

Verify your installation with a pre-trained model:

```bash
# Set up environment variable
export WORLDENGINE_ROOT=$(pwd)

# Option 1: Single GPU test 
bash scripts/closed_loop_test.sh

# Option 2: Multi-GPU test (Default 8 GPUs)
bash scripts/multigpu_closed_loop_test.sh
```

**What this does:**
- Loads a pre-trained VADv2 model (50% training data, epoch 8)
- Runs closed-loop simulation on 288 rare-case test scenarios
- Evaluates with navsim v1 PDMS (collision avoidance, progress, comfort, etc.)
- Saves results to `experiments/closed_loop_exps/e2e_vadv2_50pct/navtest_failures_NR/`

**Detailed quick start tutorial:** [docs/quick_start.md](docs/quick_start.md)

### Deep Dive by Module

After the quick test, explore each subsystem in detail:

#### SimEngine - Photorealistic Closed-Loop Simulation

Learn how to run simulations, generate rollouts, and test models:

- **Rollout scripts** for data generation (no model required)
- **Testing scripts** for model evaluation (single/multi-GPU)
- **Ray distributed simulation** for large-scale testing
- **Reactive vs non-reactive** agent modes
- **Configuration guide** for all Hydra parameters

**[SimEngine Usage Guide](docs/simengine_usage.md)**

#### AlgEngine - End-to-End Model Training & Fine-Tuning

Learn how to train models, extract rare cases, and fine-tune:

- **Training from scratch**
- **Open-loop evaluation** on test sets
- **Rare case extraction** from evaluation failures
- **RL-based fine-tuning** on long-tail scenarios
- **Multi-GPU training** with distributed data parallel

**[AlgEngine Usage Guide](docs/algengine_usage.md)**

#### Scene Reconstruction - 3D Gaussian Splatting-based method, MTGS

WorldEngine's simulation environments are powered by 3D Gaussian Splatting (MTGS):

- **Multi-traversal reconstruction** from nuPlan data
- **Photorealistic rendering** for closed-loop simulation
- **Asset generation** for SimEngine scenes

**[MTGS Repository](https://github.com/OpenDriveLab/MTGS)**


## Citation

If any parts of our work help your research, please consider citing us and giving a star to our repository:

```bibtex


```

If you use the Render Assets (MTGS), please also cite:
```bibtex
@article{li2025mtgs,
  title={MTGS: Multi-Traversal Gaussian Splatting},
  author={Li, Tianyu and Qiu, Yihang and Wu, Zhenhua and Lindstr{\"o}m, Carl and Su, Peng and Nie{\ss}ner, Matthias and Li, Hongyang},
  journal={arXiv preprint arXiv:2503.12552},
  year={2025}
}
```
If you use the augmented scenarios data, please cite as well:
```bibtex
@inproceedings{zhou2025nexus,
  title={Decoupled Diffusion Sparks Adaptive Scene Generation},
  author={Zhou, Yunsong and Ye, Naisheng and Ljungbergh, William and Li, Tianyu and Yang, Jiazhi and Yang, Zetong and Zhu, Hongzi and Petersson, Christoffer and Li, Hongyang},
  booktitle={ICCV},
  year={2025}
}
```
```bibtex
@article{li2025optimization,
  title={Optimization-Guided Diffusion for Interactive Scene Generation},
  author={Li, Shihao and Ye, Naisheng and Li, Tianyu and Chitta, Kashyap and An, Tuo and Su, Peng and Wang, Boyang and Liu, Haiou and Lv, Chen and Li, Hongyang},
  journal={arXiv preprint arXiv:2512.07661},
  year={2025}
}
```
If you find AlgEngine well, please cite as well:
```bibtex
@ARTICLE{11353028,
  author={Liu, Haochen and Li, Tianyu and Yang, Haohan and Chen, Li and Wang, Caojun and Guo, Ke and Tian, Haochen and Li, Hongchen and Li, Hongyang and Lv, Chen},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Reinforced Refinement With Self-Aware Expansion for End-to-End Autonomous Driving}, 
  year={2026},
  volume={48},
  number={5},
  pages={5774-5792},
  keywords={Adaptation models;Self-aware;Autonomous vehicles;Pipelines;Planning;Training;Reinforcement learning;Uncertainty;Data models;Safety;End-to-end autonomous driving;reinforced finetuning;imitation learning;motion planning},
  doi={10.1109/TPAMI.2026.3653866}}
```
If you find data scaling infos helpful, please also cite:
```bibtex
@article{tian2025simscale,
        title={SimScale: Learning to Drive via Real-World Simulation at Scale},
        author={Haochen Tian and Tianyu Li and Haochen Liu and Jiazhi Yang and Yihang Qiu and Guang Li and Junli Wang and Yinfeng Gao and Zhang Zhang and Liang Wang and Hangjun Ye and Tieniu Tan and Long Chen and Hongyang Li},
        journal={arXiv preprint arXiv:2511.23369},
        year={2025}
      }
```

## Contributing

We welcome contributions from the community! Whether you want to:

- **Report bugs** - Open an [Issue](https://github.com/OpenDriveLab/WorldEngine/issues)
- **Improve documentation** - Submit a [Pull Request](https://github.com/OpenDriveLab/WorldEngine/pulls)
- **Contribute code** - Fork, develop, and submit a PR

Please read our contributing guidelines before submitting PRs.

**For questions:**
1. Check the [documentation](docs/) first
2. Search existing [Issues](https://github.com/OpenDriveLab/WorldEngine/issues)


## License

All content in this repository is under the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0).

The released data is based on [nuPlan](https://www.nuscenes.org/nuplan) and is under the [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.


<!-- ## 👥 Team & Acknowledgements

WorldEngine is developed by **Shanghai Innovation Institute (SII)** and **OpenDriveLab** at The University of Hong Kong.

**Core Contributors:**
- Scene Reconstruction Team
- Simulation Platform Team
- Algorithm Development Team

We would like to thank all contributors and the open-source community for their support. -->


## Related Resources

We acknowledge all the open-source contributors for the following projects to make this work possible:

<div align="center">

| Project | Description |
|:-------:|:------------|
| [![MTGS](https://img.shields.io/badge/MTGS-Multi--Traversal_GS-blue?style=flat-square&logo=github)](https://github.com/OpenDriveLab/MTGS) | Multi-traversal Gaussian Splatting for scene reconstruction |
| [![SimScale](https://img.shields.io/badge/SimScale-Driving_Simulation-AD9BC2?style=flat-square&logo=github)](https://github.com/OpenDriveLab/SimScale) | Large scale driving simulation |
| [![nerfstudio](https://img.shields.io/badge/nerfstudio-NeRF_Framework-green?style=flat-square&logo=github)](https://github.com/nerfstudio-project/nerfstudio) | Collaboration-friendly NeRF toolkit |
| [![MMDetection3D](https://img.shields.io/badge/MMDetection3D-3D_Detection-orange?style=flat-square&logo=github)](https://github.com/open-mmlab/mmdetection3d) | 3D detection framework |
| [![UniAD](https://img.shields.io/badge/UniAD-End--to--End_AD-red?style=flat-square&logo=github)](https://github.com/OpenDriveLab/UniAD) | End-to-end autonomous driving framework |
| [![VADv2](https://img.shields.io/badge/VADv2-End--to--End_AD-crimson?style=flat-square&logo=github)](https://github.com/priest-yang/VADv2) | Vectorized autonomous driving framework |
| [![NAVSIM](https://img.shields.io/badge/NAVSIM-AD_Benchmark-teal?style=flat-square&logo=github)](https://github.com/autonomousvision/navsim) | Non-reactive autonomous vehicle simulation benchmark |
| [![nuPlan](https://img.shields.io/badge/nuPlan-Dataset-purple?style=flat-square&logo=github)](https://www.nuscenes.org/nuplan) | Large-scale autonomous driving dataset |
| [![MetaDrive](https://img.shields.io/badge/MetaDrive-Driving_Simulation-ff69b4?style=flat-square&logo=github)](https://github.com/metadriverse/metadrive) | Compositional driving simulation platform |
| [![Ray](https://img.shields.io/badge/Ray-Distributed_Computing-yellow?style=flat-square&logo=ray)](https://github.com/ray-project/ray) | Distributed execution framework |
| [![Hydra](https://img.shields.io/badge/Hydra-Config_Management-lightblue?style=flat-square&logo=python)](https://github.com/facebookresearch/hydra) | Configuration management framework |

</div>

---

<div align="center">

**If you find WorldEngine useful, please consider giving us a star!**

**Quick Links:** [Documentation](docs/) | [Installation](docs/installation.md) | [Quick Start](docs/quick_start.md) | [Issues](https://github.com/OpenDriveLab/WorldEngine/issues) | [Discussions](https://github.com/OpenDriveLab/WorldEngine/discussions)

**Contact:** For research collaboration or questions, visit our [Discussions](https://github.com/OpenDriveLab/WorldEngine/discussions)

</div>
