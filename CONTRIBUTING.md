# Contributing to Repeat Task Gym

## Setup

```bash
bash scripts/install.sh
source .venv/bin/activate
```

The package supports Python 3.11+. The install script creates `.venv` with
free-threaded Python 3.15 (`3.15t`) only if it does not already exist, then
installs the package, development tools, and example dependencies, including
Jupyter and the Python notebook kernel. Set `REPEAT_TASK_PYTHON` to choose a
different interpreter.

Alternatively, use `pip install -e ".[dev,examples]"` in your own virtual
environment, or `pip install -e ".[dev]"` for development tools only.

## Layout

- `src/repeat_task_gym/wrapper.py`: episode repetition, task boundaries, and seeding.
- `src/repeat_task_gym/__init__.py`: public package API.
- `tests/`: task behavior and Gymnasium integration tests.
- `examples/`: runnable notebooks.
- `scripts/install.sh`: development setup.

Keep the wrapper focused. Environment construction, reward shaping, metrics,
and vectorization belong in ordinary Gymnasium environments and wrappers.

## Checks

```bash
pyright
pytest -q
python -m build
python -m twine check dist/*
```

CI runs these checks on Python 3.11, 3.15, and free-threaded 3.15. Update tests and documentation
together when changing the public contract. Keep notebook outputs cleared.
The test suite executes the notebook code.

## Releases

Update the version in `pyproject.toml` and record the change in `CHANGELOG.md`.
Build and validate the distributions before publishing through the project's
chosen release process.
