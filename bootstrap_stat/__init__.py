# pyre-unsafe
"""Bootstrap methods for standard errors, confidence intervals, and more.

This package implements bootstrap methods for statistical inference, based on
Efron & Tibshirani [ET93]_. It provides standard error estimation, bias
estimation, confidence intervals, prediction error estimation, and
significance testing.

"""

# Type aliases (public)
# Public utility function
from bootstrap_stat._utils import ArrayLike, JackknifeValues, Statistic

# Bias estimation and correction
from bootstrap_stat.bias import (
    better_bootstrap_bias,
    bias,
    bias_corrected,
    jackknife_bias,
)

# Confidence intervals
from bootstrap_stat.confidence import (
    abcnon_interval,
    bcanon_interval,
    calibrate_interval,
    percentile_interval,
    t_interval,
)

# Distribution classes
from bootstrap_stat.distributions import (
    EmpiricalDistribution,
    MultiSampleEmpiricalDistribution,
)

# Prediction
from bootstrap_stat.prediction import (
    prediction_error_632,
    prediction_error_optimism,
    prediction_interval,
)

# Sampling functions
from bootstrap_stat.sampling import (
    bootstrap_samples,
    jackknife_values,
    multithreaded_bootstrap_samples,
)

# Significance testing
from bootstrap_stat.significance import (
    bcanon_asl,
    bootstrap_asl,
    bootstrap_power,
    percentile_asl,
)

# Standard error estimation
from bootstrap_stat.standard_error import (
    infinitesimal_jackknife,
    jackknife_standard_error,
    standard_error,
)

__all__ = [
    # Type aliases
    "ArrayLike",
    "JackknifeValues",
    "Statistic",
    # Distributions
    "EmpiricalDistribution",
    "MultiSampleEmpiricalDistribution",
    # Sampling
    "bootstrap_samples",
    "jackknife_values",
    "multithreaded_bootstrap_samples",
    # Standard error
    "infinitesimal_jackknife",
    "jackknife_standard_error",
    "standard_error",
    # Bias
    "better_bootstrap_bias",
    "bias",
    "bias_corrected",
    "jackknife_bias",
    # Confidence intervals
    "abcnon_interval",
    "bcanon_interval",
    "calibrate_interval",
    "percentile_interval",
    "t_interval",
    # Significance testing
    "bcanon_asl",
    "bootstrap_asl",
    "bootstrap_power",
    "percentile_asl",
    # Prediction
    "prediction_error_632",
    "prediction_error_optimism",
    "prediction_interval",
]
