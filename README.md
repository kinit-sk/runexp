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

## Requirements
- Python >= 3.7
- hydra-core
- omegaconf

## License
MIT
