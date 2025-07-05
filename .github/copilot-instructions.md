# GitHub Copilot Instructions for the TARS Project

Welcome to the TARS (Trust-Aware Reinforcement Selection) project! This document provides comprehensive guidelines for using GitHub Copilot effectively within this repository. Following these instructions will help ensure that Copilot's suggestions are consistent with our project's architecture, coding style, research objectives, and best practices.

## 1. Project Overview

TARS is a research prototype for Trust-Aware Reinforcement Selection in robust federated learning systems. It combines trust-aware client evaluation with Q-learning-based aggregation rule selection to defend against adaptive Byzantine attacks in federated learning environments.

**Key Goals:**

- Implement a mathematically principled defense against Byzantine attacks in federated learning
- Achieve superior robustness (97.7% accuracy on MNIST, 80.5% on CIFAR-10) under 20% Byzantine clients
- Provide real-time trust score monitoring and visualization capabilities
- Enable comprehensive benchmarking against baseline methods (FedAvg, Krum, Trimmed Mean/Median, FLTrust, SARA)
- Support research into adaptive aggregation strategies using reinforcement learning

**Research Focus:**

- Defend against 4 specific attack patterns: label flipping, sign flipping, Gaussian noise, and pretense attacks
- Evaluate on MNIST and CIFAR-10 datasets with non-IID data partitioning
- Demonstrate adaptive learning through Q-learning-based aggregation rule selection

## 2. Tech Stack & Dependencies

- **Primary Language:** Python 3.10+
- **Core ML Libraries:**
  - **PyTorch:** `torch>=1.12.0` for neural networks, model training, and tensor operations
  - **TorchVision:** `torchvision>=0.13.0` for MNIST and CIFAR-10 dataset handling
  - **NumPy:** `numpy>=1.21.0` for numerical computations and array operations
  - **Pandas:** `pandas>=1.4.0` for data manipulation and results analysis
- **Visualization:**
  - **Matplotlib:** `matplotlib>=3.5.0` for real-time plotting and trust score visualization
  - **Plotly:** `plotly>=5.0.0` for interactive dashboards (optional advanced features)
- **Testing:** `pytest>=7.0.0`, `pytest-cov>=3.0.0` for comprehensive testing with coverage
- **Code Quality:** `black>=22.0.0`, `flake8>=4.0.0`, `mypy>=0.950` for formatting, linting, and type checking
- **Configuration:** `PyYAML>=6.0` for YAML configuration file support

**Dependency Management:**

- Use `requirements.txt` for exact version pinning
- Update dependencies only when necessary for security or critical features
- Test thoroughly after any dependency updates

## 3. Coding Standards & Style Guide

### 3.1 Code Formatting (PEP 8 Compliance)

- **Black Formatting:** All Python code MUST be formatted with `black` using default settings
  ```bash
  black . --line-length 88 --target-version py310
  ```
- **Line Length:** Maximum 88 characters (Black default)
- **Imports:** Use `isort` for import organization:
  ```bash
  isort . --profile black
  ```

### 3.2 Linting & Code Quality

- **Flake8:** Zero tolerance for linting errors. Configuration in `.flake8`:
  ```ini
  [flake8]
  max-line-length = 88
  extend-ignore = E203, W503, E501
  exclude = .git,__pycache__,docs/source/conf.py,old,build,dist
  ```
- **MyPy:** Strict type checking required for all new code:
  ```ini
  [mypy]
  python_version = 3.10
  warn_return_any = True
  warn_unused_configs = True
  disallow_untyped_defs = True
  ```

### 3.3 Type Hints (Mandatory)

- **All functions and methods** MUST include comprehensive type hints
- Use `from typing import` for complex types:
  ```python
  from typing import Dict, List, Optional, Tuple, Union, Any, Callable
  from collections.abc import Sequence
  ```
- For PyTorch tensors, use `torch.Tensor` type hints
- For model state dictionaries, use `Dict[str, torch.Tensor]`

### 3.4 Docstring Standards (Google Style)

All modules, classes, and functions MUST have comprehensive docstrings:

```python
def calculate_trust_score(
    client_update: Dict[str, torch.Tensor],
    global_model: Dict[str, torch.Tensor],
    validation_loader: DataLoader
) -> float:
    """Calculate trust score for a client update using multi-criteria evaluation.

    Implements the trust scoring mechanism from TARS framework combining:
    - Loss divergence analysis
    - Cosine similarity computation
    - Gradient magnitude evaluation

    Args:
        client_update: Client's model state dictionary after local training
        global_model: Current global model state dictionary
        validation_loader: DataLoader for validation set evaluation

    Returns:
        Trust score in range [0, 1] where 1 indicates highest trust

    Raises:
        ValueError: If model architectures don't match
        RuntimeError: If validation evaluation fails

    Example:
        >>> trust = calculate_trust_score(client_state, global_state, val_loader)
        >>> print(f"Client trust score: {trust:.4f}")
    """
```

### 3.5 Naming Conventions

- **Variables & Functions:** `snake_case` (e.g., `trust_score`, `calculate_reward`)
- **Classes:** `PascalCase` (e.g., `TARSAgent`, `ByzantineClient`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `MAX_CLIENTS`, `DEFAULT_LEARNING_RATE`)
- **Private Methods:** `_leading_underscore` (e.g., `_update_q_table`)
- **Protected Attributes:** `_single_underscore` (e.g., `_trust_memory`)

### 3.6 Logging Standards

Use Python's built-in `logging` module with structured formatting:

```python
import logging

# Configure logger at module level
logger = logging.getLogger(__name__)

# Usage examples
logger.info("Starting TARS simulation with %d clients", num_clients)
logger.warning("Trust score below threshold: %.4f for client %d", score, client_id)
logger.error("Q-learning convergence failed after %d rounds", max_rounds)
logger.debug("Q-table state: %s, action: %d, reward: %.4f", state, action, reward)
```

## 4. Federated Learning Architecture Patterns

### 4.1 Core Components Structure

```
app/
├── simulation.py          # Main simulation orchestrator
├── components/
│   ├── server.py         # Central FL server implementation
│   └── client.py         # FL client with Byzantine attack support
├── defense/
│   ├── tars_agent.py     # Q-learning agent for rule selection
│   └── aggregation_rules.py # Byzantine-robust aggregation methods
├── attacks/
│   └── poisoning.py      # Implementation of 4 attack patterns
└── shared/
    ├── interfaces.py     # Abstract base classes and protocols
    ├── models.py         # CNN architectures for MNIST/CIFAR-10
    └── data_utils.py     # Dataset loading and partitioning
```

### 4.2 Interface Design Patterns

- **Dependency Injection:** Pass dependencies through constructor parameters
- **Abstract Base Classes:** Use `abc.ABC` for defining interfaces
- **Protocol Classes:** Use `typing.Protocol` for structural typing

```python
from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any

class IAggregationRule(Protocol):
    """Protocol for aggregation rule implementations."""

    def __call__(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client updates into global model."""
        ...

class IAttack(ABC):
    """Abstract base class for Byzantine attacks."""

    @abstractmethod
    def apply(
        self,
        model_update: Dict[str, torch.Tensor],
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """Apply attack to model update."""
        pass
```

### 4.3 Federated Learning Simulation Design

- **Stateless Components:** Clients and server maintain minimal state
- **Immutable Model States:** Use deep copies for model state sharing
- **Event-Driven Updates:** Trust scores and Q-learning updates triggered by events

## 5. Testing Framework & Standards

### 5.1 Test Structure & Coverage

```
tests/
├── unit/
│   ├── test_tars_agent.py      # Q-learning agent unit tests
│   ├── test_aggregation_rules.py # Aggregation methods testing
│   ├── test_attacks.py         # Byzantine attack validation
│   └── test_trust_scoring.py   # Trust calculation verification
├── integration/
│   ├── test_simulation_flow.py # End-to-end simulation testing
│   └── test_benchmarking.py    # Baseline comparison tests
└── fixtures/
    ├── conftest.py             # Shared pytest fixtures
    └── mock_data.py            # Test data generation
```

### 5.2 Testing Requirements

- **Minimum Coverage:** 90% code coverage for core components
- **Test Isolation:** Each test must be independent and reproducible
- **Mock External Dependencies:** Use `unittest.mock` for external services
- **Parameterized Tests:** Use `@pytest.mark.parametrize` for multiple scenarios

```python
import pytest
import torch
from unittest.mock import Mock, patch

@pytest.fixture
def mock_mnist_data():
    """Fixture providing mock MNIST data for testing."""
    return torch.randn(100, 1, 28, 28), torch.randint(0, 10, (100,))

@pytest.mark.parametrize("attack_type", ["label_flip", "sign_flip", "gaussian"])
def test_tars_robustness_against_attacks(attack_type, mock_mnist_data):
    """Test TARS performance under different Byzantine attacks."""
    # Test implementation here
    pass

def test_trust_score_calculation():
    """Test trust score computation with known inputs."""
    # Arrange
    client_update = {...}
    global_model = {...}

    # Act
    trust_score = calculate_trust_score(client_update, global_model, val_loader)

    # Assert
    assert 0.0 <= trust_score <= 1.0
    assert isinstance(trust_score, float)
```

### 5.3 Performance & Benchmark Testing

- **Accuracy Validation:** Verify research paper results (97.7% MNIST, 80.5% CIFAR-10)
- **Convergence Testing:** Ensure Q-learning convergence within 30 rounds
- **Baseline Comparisons:** Automated testing against all baseline methods
- **Memory & Speed Tests:** Monitor resource usage during simulation

## 6. Real-Time Visualization Standards

### 6.1 Matplotlib Integration

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class TrustScoreVisualizer:
    """Real-time trust score visualization using matplotlib."""

    def __init__(self, num_clients: int):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.trust_data = {i: [] for i in range(num_clients)}

    def update_plot(self, round_num: int, trust_scores: Dict[int, float]) -> None:
        """Update trust score plot with new data."""
        # Implementation with real-time updates
        pass
```

### 6.2 Interactive Dashboard Features

- **Real-time Updates:** Update plots every simulation round
- **Multi-panel Layout:** Separate panels for trust, accuracy, Q-learning progress
- **Export Capabilities:** Save plots in publication-ready formats (PNG, PDF, SVG)

## 7. Configuration Management

### 7.1 YAML Configuration Structure

```yaml
# config/tars_config.yaml
simulation:
  dataset: "mnist" # mnist | cifar10
  num_clients: 10
  byzantine_percentage: 0.2
  num_rounds: 30
  attack_type: "label_flip" # label_flip | sign_flip | gaussian | pretense

q_learning:
  learning_rate: 0.1
  discount_factor: 0.9
  epsilon_start: 1.0
  epsilon_decay: 0.995
  epsilon_min: 0.01

trust_scoring:
  similarity_weight: 0.5
  loss_weight: 0.3
  magnitude_weight: 0.2
  temporal_decay: 0.5

visualization:
  real_time: true
  save_plots: true
  output_format: ["png", "pdf"]
```

### 7.2 Configuration Validation

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class TARSConfig:
    """Type-safe configuration dataclass."""

    dataset: Literal["mnist", "cifar10"]
    num_clients: int
    byzantine_percentage: float
    attack_type: Literal["label_flip", "sign_flip", "gaussian", "pretense"]

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.byzantine_percentage <= 1.0:
            raise ValueError("Byzantine percentage must be between 0 and 1")
        if self.num_clients < 2:
            raise ValueError("Minimum 2 clients required")
```

## 8. Performance Optimization Guidelines

### 8.1 PyTorch Best Practices

- **Device Management:** Always specify device (CPU/GPU) explicitly
- **Memory Efficiency:** Use `torch.no_grad()` for inference operations
- **Batch Processing:** Vectorize operations when possible
- **Model State Management:** Use `model.eval()` and `model.train()` appropriately

```python
def efficient_trust_calculation(
    client_updates: List[Dict[str, torch.Tensor]],
    device: torch.device
) -> torch.Tensor:
    """Vectorized trust score calculation for multiple clients."""
    with torch.no_grad():
        # Vectorized computation implementation
        pass
```

### 8.2 Simulation Optimization

- **Lazy Loading:** Load datasets only when needed
- **Parallel Processing:** Use `torch.multiprocessing` for client simulations
- **Memory Management:** Clear unused tensors explicitly with `del`

## 9. Error Handling & Debugging

### 9.1 Exception Handling Patterns

```python
class TARSException(Exception):
    """Base exception for TARS-related errors."""
    pass

class ConvergenceError(TARSException):
    """Raised when Q-learning fails to converge."""
    pass

class TrustScoreError(TARSException):
    """Raised when trust score calculation fails."""
    pass

def robust_q_learning_update(
    q_table: Dict,
    state: Tuple,
    action: int,
    reward: float
) -> None:
    """Q-learning update with comprehensive error handling."""
    try:
        # Q-learning implementation
        pass
    except (ValueError, KeyError) as e:
        logger.error("Q-learning update failed: %s", e)
        raise ConvergenceError(f"Failed to update Q-table: {e}") from e
```

### 9.2 Debugging & Profiling

- **Detailed Logging:** Log all critical decision points
- **Performance Profiling:** Use `cProfile` for performance bottlenecks
- **Memory Profiling:** Monitor memory usage with `memory_profiler`

## 10. Research Reproducibility Standards

### 10.1 Seed Management

```python
import random
import numpy as np
import torch

def set_random_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 10.2 Experiment Tracking

- **Results Logging:** Save all experimental results with timestamps
- **Parameter Tracking:** Log all hyperparameters and configuration settings
- **Baseline Reproduction:** Ensure exact reproduction of baseline method results

## 11. Deployment & Distribution

### 11.1 Research Prototype Packaging

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="tars-fl",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "PyYAML>=6.0",
    ],
    python_requires=">=3.10",
    author="TARS Research Team",
    description="Trust-Aware Reinforcement Selection for Federated Learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
```

### 11.2 Documentation Standards

- **README.md:** Comprehensive setup and usage instructions
- **API Documentation:** Sphinx-generated documentation from docstrings
- **Research Notes:** Document all experimental findings and insights

## 12. Git Workflow & Collaboration

### 12.1 Commit Message Standards

```
feat(tars): implement Q-learning aggregation selection
fix(trust): correct cosine similarity calculation
docs(readme): add installation instructions
test(attacks): add comprehensive Byzantine attack tests
refactor(viz): optimize real-time plotting performance
```

### 12.2 Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: black
        name: black
        entry: black
        language: system
        types: [python]
      - id: flake8
        name: flake8
        entry: flake8
        language: system
        types: [python]
      - id: mypy
        name: mypy
        entry: mypy
        language: system
        types: [python]
```

## 13. Security & Privacy Considerations

### 13.1 Federated Learning Privacy

- **No Raw Data Sharing:** Ensure client data never leaves local environment
- **Model Parameter Security:** Validate all received model updates
- **Trust Score Privacy:** Avoid leaking information through trust calculations

### 13.2 Research Ethics

- **Reproducible Results:** All experiments must be reproducible
- **Fair Comparison:** Baseline implementations must be faithful to original papers
- **Open Research:** Code and results should be openly available

---

**Remember:** This is a research prototype focused on advancing federated learning security. Prioritize correctness, reproducibility, and clear documentation over premature optimization. When in doubt, consult the research paper and existing literature for implementation guidance.

For questions or clarifications about these guidelines, please refer to the PRD document or create an issue in the repository.
