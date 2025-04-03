# Implementation of Diloco and Different Optimizers

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

## Makefile Targets

The Makefile includes several targets to streamline common development and deployment tasks:

- **`format`**: Formats the codebase using `black` and `isort`.

  ```bash
  make format
  ```

- **`lint`**: Runs the linter using `ruff` to check for coding style issues.

  ```bash
  make lint
  ```

- **`test`**: Executes the formatter, linter, and test suite using `pytest`.

  ```bash
  make test
  ```

- **`build`**: Builds the Exalsius Docker image using the provided Dockerfile.

  ```bash
  make build
  ```

## UV

1. Install UV
```
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```
2. Create venv, activate, and install dependecies including dev and test dependencies
```
uv venv
source .venv/bin/activate
uv sync --dev --extra test
```

## Tests
 
To execute the tests one needs to perform UV installation step and then:
```
uv run pytest --dev
```