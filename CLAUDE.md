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
# Install dependencies and activate environment
poetry install
poetry shell

# Run all tests with coverage
python -m pytest

# Run a single test file
python -m pytest tests/test_bootstrap_stat.py

# Run tests matching a pattern
python -m pytest -k "test_standard_error"

# Format code (79-char line width)
poetry run black bootstrap_stat/ tests/

# Build documentation
cd docs && make html
```

## Code Architecture

### Core Module: `bootstrap_stat/bootstrap_stat.py`

**Key Classes:**
- `EmpiricalDistribution` - Represents the empirical distribution of a
  sample; provides `sample()` method for drawing bootstrap samples
- `MultiSampleEmpiricalDistribution` - Extension for two-sample
  problems (treatment/control comparisons)

**Function Categories:**

Standard error and bias:
- `standard_error()`, `jackknife_standard_error()`, `infinitesimal_jackknife()`
- `bias()`, `jackknife_bias()`, `better_bootstrap_bias()`, `bias_corrected()`

Confidence intervals:
- `percentile_interval()` - Simple percentile method
- `bcanon_interval()` - BCa (Bias-Corrected and Accelerated), recommended default
- `abcnon_interval()` - Analytical BCa approximation
- `t_interval()` - Bootstrap-t method
- `calibrate_interval()` - Calibrated coverage adjustment

Significance testing:
- `bootstrap_asl()`, `percentile_asl()`, `bcanon_asl()` - Achieved significance levels

Prediction:
- `prediction_error_optimism()`, `prediction_error_632()`
- `prediction_interval()`

**Statistic Functions:**

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

### Datasets: `bootstrap_stat/datasets.py`

Example datasets for testing: `law_data()`, `mouse_data()`,
`rainfall_data()`, `spatial_test_data()`, `hormone_data()`,
`patch_data()`.

## CI/CD

- GitHub Actions runs tests on Python 3.6, 3.7, 3.8
- PRs to master trigger build workflow (format check + tests)
- Pushes to master trigger PyPI deployment via `poetry publish`

## References

[ET93] Bradley Efron and Robert J. Tibshirani, "An Introduction to the
Bootstrap". Chapman & Hall, 1993.
