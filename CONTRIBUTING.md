# Contributing to WorldEngine

Thank you for your interest in contributing to WorldEngine! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## Getting Started

### Prerequisites

- Python 3.8+ (AlgEngine) or Python 3.9+ (SimEngine)
- CUDA 11.8
- PyTorch 2.0.1
- Git

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub

2. **Clone your fork locally**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/WorldEngine.git
   cd WorldEngine
   ```

3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/OpenDriveLab/WorldEngine.git
   ```

4. **Install dependencies**:
   
   For SimEngine (Python 3.9):
   ```bash
   bash scripts/install_simengine.sh
   ```
   
   For AlgEngine (Python 3.8):
   ```bash
   bash scripts/install_algengine.sh
   ```

5. **Verify installation**:
   ```bash
   # Run quick test
   bash scripts/closed_loop_test.sh
   ```

## Development Workflow

### Creating a Branch

Always create a new branch for your work:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or updates

### Keeping Your Branch Updated

Regularly sync with the upstream repository:

```bash
git fetch upstream
git rebase upstream/main
```

## Code Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications:

- **Line length**: Maximum 120 characters
- **Imports**: Organize imports in the following order:
  1. Standard library imports
  2. Related third-party imports
  3. Local application/library imports
- **Docstrings**: Use Google-style docstrings
- **Type hints**: Add type annotations for function signatures

Example:

```python
from typing import List, Optional
import numpy as np
from .utils import helper_function


def process_data(
    input_data: np.ndarray,
    threshold: float = 0.5,
    debug: bool = False
) -> List[float]:
    """Process input data and return filtered results.
    
    Args:
        input_data: Input numpy array to process
        threshold: Minimum value threshold for filtering
        debug: Enable debug logging
        
    Returns:
        List of processed values that exceed the threshold
        
    Raises:
        ValueError: If input_data is empty
    """
    if len(input_data) == 0:
        raise ValueError("Input data cannot be empty")
    
    results = []
    for value in input_data:
        if value > threshold:
            results.append(float(value))
    
    return results
```

### Code Formatting

We recommend using automated formatters:

- **black**: For code formatting (line length 120)
- **isort**: For import sorting
- **flake8**: For linting

Install pre-commit hooks to automatically format code:

```bash
pip install pre-commit
pre-commit install
```

### Documentation

- Add docstrings to all public classes, methods, and functions
- Update relevant documentation when adding features
- Keep comments clear and concise
- Use inline comments sparingly - prefer self-documenting code

## Testing

### Running Tests

Before submitting a pull request, ensure all tests pass:

```bash
# Run quick test
bash scripts/closed_loop_test.sh

# Run distributed test (multi-GPU)
bash scripts/multigpu_closed_loop_test.sh
```

### Writing Tests

- Add tests for new features
- Ensure tests are deterministic and reproducible
- Test edge cases and error conditions
- Use descriptive test names that explain what is being tested

Example:

```python
def test_process_data_with_valid_input():
    """Test process_data with valid input returns expected results."""
    input_data = np.array([0.3, 0.6, 0.9])
    result = process_data(input_data, threshold=0.5)
    assert len(result) == 2
    assert result == [0.6, 0.9]


def test_process_data_with_empty_input_raises_error():
    """Test process_data raises ValueError for empty input."""
    with pytest.raises(ValueError):
        process_data(np.array([]))
```

## Submitting Changes

### Pull Request Process

1. **Ensure your code follows the style guidelines**

2. **Update documentation** if needed

3. **Add or update tests** as appropriate

4. **Commit your changes** with clear, descriptive messages:
   ```bash
   git add .
   git commit -m "feat: add trajectory prediction module
   
   - Implement LSTM-based trajectory predictor
   - Add unit tests for predictor
   - Update documentation with usage examples"
   ```

   Commit message format:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `refactor:` - Code refactoring
   - `test:` - Test updates
   - `chore:` - Maintenance tasks

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub:
   - Use a clear, descriptive title
   - Reference related issues (e.g., "Fixes #123")
   - Describe your changes in detail
   - Include screenshots for UI changes
   - List any breaking changes

### Pull Request Review

- Be responsive to reviewer feedback
- Make requested changes in new commits (don't force push)
- Once approved, a maintainer will merge your PR

## Reporting Issues

### Before Submitting an Issue

1. Check if the issue already exists
2. Verify you're using the latest version
3. Collect relevant information:
   - Python version
   - PyTorch version
   - CUDA version
   - Operating system
   - Error messages and stack traces

### Creating an Issue

Use our [issue templates](.github/ISSUE_TEMPLATE/) to report:
- **Bug reports**: Problems with the code
- **Feature requests**: Suggestions for new features

Provide as much detail as possible to help us understand and reproduce the issue.

## Feature Requests

We welcome feature requests! When suggesting a feature:

1. **Search existing issues** to avoid duplicates
2. **Describe the problem** your feature would solve
3. **Explain your proposed solution** in detail
4. **Consider alternatives** and explain why your solution is best
5. **Provide examples** of how the feature would be used

## Questions?

If you have questions about contributing:

- Open a [Discussion](https://github.com/OpenDriveLab/WorldEngine/discussions)
- Check the [FAQ](README.md#-faq) in the README
- Review existing [Issues](https://github.com/OpenDriveLab/WorldEngine/issues)

## License

By contributing to WorldEngine, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for contributing to WorldEngine! 🚀
