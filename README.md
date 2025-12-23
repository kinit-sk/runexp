# runexp

A Python package for experiment configuration and running using Hydra and OmegaConf.

## Features
- Modular experiment runner
- Hydra-based configuration
- Command-line interface: `python -m runexp`

## Installation

For editable install in your development environment, run:
```sh
pip install -e .
```

For standard installation, run:
```sh
pip install .
```

## Usage

### Run from CLI

```sh
python -m runexp
```

E.g. to run experiment configured in configs/experiment1.yaml, one should run:

```
python -m runexp -cn experiment1.yaml
```

### Configuration

Find examples of how to setup your experiments and configuration files in `examples/`.

### Logging the Results

The example configuration automatically logs results into MLFlow. For simplicity, we are using the sqlite backend with this `tracking_uri`: `"sqlite:///mlruns/mlruns.db"`.

However, for practical purposes, logging is usually faster when MLFlow is run as a separate process, e.g. like this:
```
mkdir -p mlruns
mlflow server --backend-store-uri=sqlite:///mlruns/mlruns.db --default-artifact-root=file:mlruns --host 0.0.0.0 --port 5000
```

The `tracking_uri` would then be `"http://localhost:5000"`.

Wandb logging is also supported.

## Requirements
- Python >= 3.7
- hydra-core
- omegaconf

## License
MIT
