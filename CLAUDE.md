# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when
working with code in this repository.

## Project Overview

Bootstrap-stat is a Python library implementing bootstrap methods for
statistical inference, based on Efron & Tibshirani (1993). It provides
standard error estimation, bias estimation, confidence intervals,
prediction error estimation, and significance testing.

## Development Commands

```bash
# Install dependencies
uv sync --all-extras

# Run all tests with coverage
uv run pytest

# Run a single test file
uv run pytest tests/test_bootstrap_stat.py

# Run tests matching a pattern
uv run pytest -k "test_standard_error"

# Lint with ruff
uv run ruff check bootstrap_stat/ tests/

# Format code
uv run black bootstrap_stat/ tests/

# Type check
uv run mypy bootstrap_stat/

# Build documentation
cd docs && uv run make html
```

## Code Architecture

The package is organized into submodules by functionality. All public
API is re-exported from `bootstrap_stat/__init__.py` for convenient
imports:

```python
import bootstrap_stat as bp
# or
from bootstrap_stat import EmpiricalDistribution, bcanon_interval
```

### Module Structure

```
bootstrap_stat/
├── __init__.py           # Re-exports all public API
├── _utils.py             # Private utilities and type aliases
├── distributions.py      # EmpiricalDistribution classes
├── sampling.py           # bootstrap_samples, jackknife_values
├── standard_error.py     # SE estimation methods
├── confidence.py         # Confidence interval methods
├── bias.py               # Bias estimation and correction
├── significance.py       # ASL and power analysis
├── prediction.py         # Prediction error and intervals
└── datasets.py           # Example datasets
```

### Key Classes (`distributions.py`)

- `EmpiricalDistribution` - Represents the empirical distribution of a
  sample; provides `sample()` method for drawing bootstrap samples
- `MultiSampleEmpiricalDistribution` - Extension for two-sample
  problems (treatment/control comparisons)

### Function Categories

**Standard error** (`standard_error.py`):
- `standard_error()`, `jackknife_standard_error()`, `infinitesimal_jackknife()`

**Bias** (`bias.py`):
- `bias()`, `jackknife_bias()`, `better_bootstrap_bias()`, `bias_corrected()`

**Confidence intervals** (`confidence.py`):
- `percentile_interval()` - Simple percentile method
- `bcanon_interval()` - BCa (Bias-Corrected and Accelerated), recommended default
- `abcnon_interval()` - Analytical BCa approximation
- `t_interval()` - Bootstrap-t method
- `calibrate_interval()` - Calibrated coverage adjustment

**Significance testing** (`significance.py`):
- `bootstrap_asl()`, `percentile_asl()`, `bcanon_asl()` - Achieved significance levels
- `bootstrap_power()` - Power analysis

**Prediction** (`prediction.py`):
- `prediction_error_optimism()`, `prediction_error_632()`
- `prediction_interval()`

**Sampling** (`sampling.py`):
- `bootstrap_samples()`, `multithreaded_bootstrap_samples()`
- `jackknife_values()`

### Statistic Functions

All bootstrap methods take a `statistic` parameter: a callable that
accepts a sample (array or DataFrame) and returns a scalar. Example:
```python
def correlation(df):
    return np.corrcoef(df["LSAT"], df["GPA"])[0, 1]
```

### Parallelization

Multicore support via pathos (uses dill instead of pickle, allowing
locally-defined functions). Specify `num_threads` parameter in
applicable functions.

### Datasets (`datasets.py`)

Example datasets for testing: `law_data()`, `mouse_data()`,
`rainfall_data()`, `spatial_test_data()`, `hormone_data()`,
`patch_data()`.

## CI/CD

- GitHub Actions runs tests on Python 3.10, 3.11, 3.12
- PRs to master trigger build workflow (ruff, black check, tests)
- Pushes to master trigger PyPI deployment via `uv publish`

## Documentation deployment

Docs are served from a DigitalOcean droplet via nginx. The deploy
workflow (`.github/workflows/deploy_docs.yml`) fires on pushes to
master that touch `docs/**`. It SSHes to the droplet and runs
`git pull --ff-only` — it does **not** run `make html`.

**`docs/_build/html/` must therefore be committed to git.** The
`.gitignore` does not exclude it. When editing docs:

```bash
# 1. Edit RST sources under docs/
# 2. Rebuild
cd docs && uv run make html

# 3. Stage sources and build artifacts together
git add docs/<changed-source-files>
git add -f docs/_build/html/   # -f needed for any new files
git rm --cached docs/_build/html/.buildinfo.bak 2>/dev/null || true  # exclude ephemeral artifact

# 4. Commit and push to master
```

`bootstrap-stat` is a submodule of the nginx-proxy repo on the
droplet. Each pull updates the working tree, but the parent repo's
submodule pointer goes stale — cosmetic only; nginx serves from the
working tree directly.

## References

[ET93] Bradley Efron and Robert J. Tibshirani, "An Introduction to the
Bootstrap". Chapman & Hall, 1993.
