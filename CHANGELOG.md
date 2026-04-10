# Changelog

All notable changes to WorldEngine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial open source release preparation
- LICENSE file (Apache 2.0)
- CONTRIBUTING.md with contribution guidelines
- CODE_OF_CONDUCT.md for community standards
- GitHub issue and PR templates
- CI/CD workflows for automated testing and linting
- Pre-commit hooks configuration

### Changed
- Fixed installation scripts to use auto-detected paths instead of placeholders
- Updated quick test scripts with automatic WORLDENGINE_ROOT detection

### Security
- Removed internal source URLs from configuration files

## [0.1.0] - 2026-04-06

### Added
- **SimEngine**: Photorealistic closed-loop simulation engine
  - 3D Gaussian Splatting (3DGS) based rendering
  - Real-time interactive simulation
  - Ray distributed testing support
  - Hydra-based configuration management
  
- **AlgEngine**: End-to-end model training and evaluation framework
  - Support for UniAD and NavFormer architectures
  - MMDetection3D integration
  - nuPlan dataset support
  - Closed-loop evaluation capabilities
  - RL-based post-training pipeline

- **Documentation**:
  - Comprehensive installation guide
  - Quick start tutorial
  - Detailed usage documentation for both engines
  - Data organization guidelines
  - FAQ section

- **Features**:
  - Automatic long-tail scenario discovery
  - Behavior world model for traffic generation
  - Multi-GPU distributed training and testing
  - Production-scale ADAS validation

### Notes
- Requires Python 3.8+ (AlgEngine) or 3.9+ (SimEngine)
- CUDA 11.8 and PyTorch 2.0.1 required
- Tested on Linux systems with NVIDIA GPUs

---

## Release History

### Version Numbering

- **MAJOR** version for incompatible API changes
- **MINOR** version for new functionality in a backward compatible manner
- **PATCH** version for backward compatible bug fixes

### Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for information on how to contribute to this changelog and the project.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
