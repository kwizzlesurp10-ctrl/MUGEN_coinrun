# CoinRun Package

This directory contains the CoinRun reinforcement learning environment that was merged with the MusicGen project.

## Installation

The CoinRun package is not automatically installed with MusicGen. If you need to use CoinRun, you can install it separately:

### Dependencies

CoinRun requires the following packages:
- gym
- baselines (OpenAI baselines)
- tensorflow
- mpi4py

### Manual Installation

```bash
# Install dependencies first
pip install gym tensorflow mpi4py
pip install git+https://github.com/openai/baselines.git

# Then you can import coinrun from the coinrun/ directory
export PYTHONPATH="${PYTHONPATH}:/path/to/MUGEN_coinrun"
```

## Usage

```python
from coinrun import make

# Create a CoinRun environment
env = make('standard', num_envs=1)
```

## Note

The main package in this repository is MusicGen. CoinRun is included as a secondary component and is not installed by default when installing the MusicGen package.
