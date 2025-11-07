<p align="middle"><img src="./docs/assets/logo_banner.png" alt="exalsius banner" width="250"></p>

<h1 align="center">Distributed Low-Communication Training</h1>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![CI](https://img.shields.io/github/actions/workflow/status/exalsius/exalsius-operator/ci.yml?label=CI) [![Docker Image](https://img.shields.io/badge/docker-ghcr.io%2Fexalsius%2Fdiloco--training-blue)](https://github.com/exalsius/diloco-training/pkgs/container/diloco-training)

</div>

## Overview


This repository provides an extended implementation of **[DiLoCo (Distributed Low-Communication Training)](https://arxiv.org/abs/2311.08105)** and several **communication-efficient optimizers** for large-scale model training.  
It is part of the broader **[exalsius stack](https://github.com/exalsius)**, which enables **scheduling and orchestrating distributed training workloads** across **geo-distributed GPU resources**.

Traditional large-model training assumes high-bandwidth interconnects within data centers.  
This work explores how to train effectively across **heterogeneous, geographically distributed clusters** by reducing synchronization frequency and communication volume between model replicas.


## Highlights
- Extends the original **DiLoCo** implementation with **additional optimizers** and **momentum compression techniques**
- Integrates seamlessly into the **[exalsius framework](https://github.com/exalsius)** for cross-cluster and cross-cloud scheduling
- Reduces communication cost by combining **infrequent synchronization** with **frequency-based momentum decomposition**
- Supports transformer-based and CNN architectures for NLP and vision workloads
- To be published in the **NeurIPs 2025 - DynaFront Workshop** ([Preprint](https://arxiv.org/abs/2510.03371))

---

## Supported Models

This implementation supports a variety of architectures across domains:

| Domain | Model | Description |
|:--------|:-------|:-------------|
| **Vision** | **BigGAN** | Generative Adversarial Network for high-fidelity image synthesis |
| | **ResNet** | Convolutional neural network for image classification and feature extraction |
| **Language** | **GPT-Neo** | Transformer-based autoregressive language model |
| | **GPT-NeoX** | Large-scale, distributed GPT variant optimized for scalability |
| **Speech** | **Wav2Vec 2.0** | Self-supervised speech representation model |

Additional models can be integrated in `diloco_training/models/`.

---

## Supported Optimizers

The following optimizers are (or will be) supported in this repository:

| Optimizer | Status | Description |
|:-----------|:--------|:-------------|
| **DiLoCo** | ✅ | Distributed Low-Communication baseline optimizer |
| **DCT-Momentum** | ✅ | Momentum decomposition via Discrete Cosine Transform (DCT) |
| **TBA** | ⏳ | Additional optimizers under development (to be announced) |

---


## Integration within the exalsius Stack

This repository can be used standalone or as part of the exalsius distributed AI platform, which coordinates and scales training workloads across multiple geo-distributed GPU resources.

Within that context, training jobs can be:

- Scheduled automatically across geographically distributed GPU nodes

- Monitored through exalsius observability components

- Executed efficiently on heterogeneous infrastructures with low-bandwidth interconnects

This enables scalable, communication-efficient training beyond the boundaries of traditional data centers.

For more details on the exalsius platform, visit the [exalsius documentation](https://docs.exalsius.ai).



## Prerequisites

Before getting started, ensure you have the following installed:

- **Python 3.12**: Required for running the application and dependencies.
- **uv**: Dependency management and packaging tool.

To maintain code quality and enforce consistent style, we suggest to use a pre-commit hook. Follow these steps to set it up:

1. Install the pre-commit hook:

   ```bash
   pre-commit install
   ```

2. Run the hooks manually on all files (optional):
   ```bash
   pre-commit run --all-files
   ```

---

## Development

### Prerequisites

Before getting started, ensure you have the following installed:

* **Python 3.12** — Required for running the application and dependencies
* **uv** — Dependency management and packaging tool

To maintain code quality and enforce consistent style, we suggest using a pre-commit hook.
Install and use it as follows:

```bash
# Install the pre-commit hook
pre-commit install

# (Optional) Run the hooks manually on all files
pre-commit run --all-files
```

---

### Makefile Targets

The Makefile includes several targets to streamline common development and deployment tasks:

| Target     | Description                                      | Command       |
| :--------- | :----------------------------------------------- | :------------ |
| **format** | Format codebase using `black` + `isort`          | `make format` |
| **lint**   | Run `ruff` linter to check for code style issues | `make lint`   |
| **test**   | Execute formatter, linter, and test suite        | `make test`   |
| **build**  | Build the Docker image                  | `make build`  |

---

### UV Workflow

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies (dev + test)
uv sync --dev --extra test
```

---

### Tests

To execute the test suite after setting up uv:

```bash
uv run pytest --dev
```

---


## Publication

An initial version of this implementation was used for the following publication. If you use this code in your research, please cite:

```
@article{nedelkoski2025distributed,
  title={Distributed Low-Communication Training with Decoupled Momentum Optimization},
  author={Nedelkoski, Sasho and Acker, Alexander and Kao, Odej and Becker, Soeren and Scheinert, Dominik},
  journal={NeurIPS 2025 - DynaFront 2025: Dynamics at the Frontiers of Optimization, Sampling, and Games Workshop},
  year={2025}
}
```