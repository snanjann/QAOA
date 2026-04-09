# QAOA for Max-Cut

This repository contains a Qiskit-based `qaoa` package and a set of notebooks and scripts for Max-Cut experiments covering:

- ideal statevector QAOA,
- edge-color parallelized QAOA circuits,
- transpiled noisy Aer experiments,
- Recursive QAOA (RQAOA),
- Extended Recursive QAOA (ERQAOA).

## Repository Layout

### Package

- `qaoa/core.py`: graph helpers, Max-Cut Hamiltonian construction, sequential QAOA circuits, parallelized QAOA circuits, edge coloring, and circuit statistics.
- `qaoa/algorithms.py`: variational optimization helpers, INTERP warm start, adaptive shot schedules, RQAOA, and ERQAOA.
- `qaoa/__init__.py`: public package exports.

### Tests

- `test_qaoa.py`: unit tests for core QAOA, RQAOA, ERQAOA, and a basic Aer execution path.

### Scripts

- `scripts/run_ideal.py`: scriptable ideal statevector QAOA experiment.
- `scripts/run_transpiled.py`: scriptable transpiled noisy Aer experiment.
- `scripts/run_rqaoa.py`: scriptable RQAOA experiment.
- `scripts/run_erqaoa.py`: scriptable ERQAOA experiment.

### Notebooks

- `qaoa_ideal.ipynb`: ideal noiseless QAOA experiments for Max-Cut.
- `qaoa_improved.ipynb`: edge-coloring parallelism and INTERP warm-start experiments.
- `qaoa_transpiled.ipynb`: transpilation and noisy Aer simulation.
- `rqaoa.ipynb`: RQAOA experiments.
- `erqaoa.ipynb`: ERQAOA experiments.

## Paper-Oriented Mapping

Use the following artifacts for each implementation claim:

- Ideal clean Max-Cut QAOA:
  - code: `qaoa.build_qaoa_circuit(...)`
  - notebook: `qaoa_ideal.ipynb`
  - script: `scripts/run_ideal.py`
- Optimized parallel circuit construction:
  - code: `qaoa.build_qaoa_circuit_parallel(...)`
  - notebook: `qaoa_improved.ipynb`
- Transpilation and noisy Aer evaluation:
  - notebook: `qaoa_transpiled.ipynb`
  - script: `scripts/run_transpiled.py`
- Recursive QAOA:
  - code: `qaoa.rqaoa_solve(...)`
  - notebook: `rqaoa.ipynb`
  - script: `scripts/run_rqaoa.py`
- Extended Recursive QAOA:
  - code: `qaoa.erqaoa_solve(...)`
  - notebook: `erqaoa.ipynb`
  - script: `scripts/run_erqaoa.py`

## Reproducible Environment

The repository includes a pinned local environment file:

```bash
pip install -r requirements.txt
pip install -e .
```

Optional extras:

```bash
pip install -e .[dev,notebooks]
```

For fake-backend notebook paths that use IBM Runtime fake providers:

```bash
pip install -e .[fake-backend]
```

## Running Reproduction Scripts

Ideal QAOA:

```bash
python scripts/run_ideal.py --nodes 8 --degree 3 --depth-max 3 --restarts 5 --seed 42
```

Transpiled noisy Aer:

```bash
python scripts/run_transpiled.py --nodes 6 --degree 3 --depth-max 3 --shots 2048 --seed 42
```

RQAOA:

```bash
python scripts/run_rqaoa.py --graph regular --nodes 8 --degree 3 --depth 1 --threshold 2 --restarts 5 --seed 42
```

ERQAOA:

```bash
python scripts/run_erqaoa.py --graph regular --nodes 8 --degree 3 --depth 1 --threshold 2 --k-max 2 --restarts 5 --seed 42
```

Each script prints a JSON summary to stdout so results can be captured in files or integrated into a plotting pipeline.

## Running Tests

```bash
python -m pytest test_qaoa.py -v
```

## Notes

- Reusable implementation code lives in the `qaoa/` package.
- Notebooks are retained as experiment records and visual analysis artifacts.
- The scripts provide non-interactive entry points better suited for a paper repository.
