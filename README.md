# TARS: Trust-Aware Reinforcement Selection for Robust Federated Learning

A research prototype implementing TARS, a federated learning framework that combines trust-aware client evaluation with reinforcement learning-based aggregation rule selection to defend against adaptive Byzantine attacks.

## Features

- **Trust-Aware Client Evaluation**: Multi-criteria trust scoring using loss divergence, cosine similarity, and gradient magnitude
- **Q-Learning Aggregation Selection**: Dynamic selection of optimal aggregation rules using reinforcement learning
- **Byzantine Attack Simulation**: Support for sign flipping, Gaussian noise, and pretense attacks
- **Multiple Aggregation Methods**: FedAvg, Krum, Trimmed Mean, Median, and FLTrust
- **Dataset Support**: MNIST and CIFAR-10 with non-IID data partitioning

## Quick Start

### Prerequisites

- Python 3.8 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shafiqahmeddev/tars-fl-sim.git
   cd tars-fl-sim
   ```

2. **One-time setup** - Install all dependencies:
   ```bash
   ./scripts/setup.sh
   ```

3. **Run the simulation**:
   ```bash
   ./scripts/run.sh
   ```

That's it! The simulation will run and save results to `simulation_results.csv`.

### Alternative Usage

If you prefer manual commands:

```bash
# Install dependencies
uv sync

# Run simulation
uv run python main.py

# Or activate environment manually
source .venv/bin/activate
python main.py
```

## Configuration

Edit the configuration in `main.py` (lines 6-19) to customize:

- **Dataset**: `'mnist'` or `'cifar10'`
- **Clients**: Number of federated learning clients
- **Byzantine Percentage**: Fraction of malicious clients (0.0-1.0)
- **Attack Type**: `'sign_flipping'` or `'gaussian'`
- **Training Rounds**: Number of federated learning rounds
- **Q-Learning Parameters**: Learning rate, discount factor, epsilon values
- **Trust Parameters**: Temporal smoothing factor (beta)

## Project Structure

```
tars-fl-sim/
├── main.py                    # Entry point with configuration
├── scripts/
│   ├── setup.sh              # One-time setup script
│   └── run.sh                # Simulation runner
├── app/
│   ├── simulation.py          # Main simulation orchestrator
│   ├── components/
│   │   ├── server.py          # Central FL server
│   │   └── client.py          # FL client with attack support
│   ├── defense/
│   │   ├── tars_agent.py      # Q-learning agent with trust scoring
│   │   └── aggregation_rules.py # Byzantine-robust aggregation methods
│   ├── attacks/
│   │   └── poisoning.py       # Attack implementations
│   └── shared/
│       ├── models.py          # CNN architectures
│       ├── data_utils.py      # Dataset utilities
│       └── interfaces.py      # Abstract base classes
└── pyproject.toml             # Project dependencies and metadata
```

## Research Context

This implementation targets the performance metrics from the TARS research paper:
- **97.7% accuracy on MNIST** with 20% Byzantine clients
- **80.5% accuracy on CIFAR-10** with 20% Byzantine clients
- Superior performance compared to static aggregation methods

## Development

### Dependencies

Core dependencies are managed in `pyproject.toml`:
- **PyTorch**: Neural networks and tensor operations
- **torchvision**: Dataset loading and transforms
- **NumPy**: Numerical computations
- **Pandas**: Results export and analysis
- **Matplotlib**: Visualization support

### Optional Development Tools

Install development dependencies:
```bash
uv sync --extra dev
```

This includes pytest, black, flake8, mypy, and jupyter for development.

## Results

The simulation outputs:
- **Console logs**: Real-time progress and metrics
- **CSV file**: Detailed results saved to `simulation_results.csv`
- **Trust scores**: Per-client trust evolution
- **Aggregation choices**: Q-learning rule selection history

## Author

**Shafiq Ahmed**  
University of Essex  
Email: s.ahmed@essex.ac.uk

## License

This is a research prototype for academic use.

## Contributing

See `github_issues.md` for detailed implementation roadmap and development tasks.