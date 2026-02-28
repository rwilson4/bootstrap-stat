# pyre-unsafe
"""Bootstrap methods for standard errors, confidence intervals, and more.

This package implements bootstrap methods for statistical inference, based on
Efron & Tibshirani (1993). It provides standard error estimation, bias
estimation, confidence intervals, prediction error estimation, and
significance testing.

References
----------
[ET93]:  Bradley Efron and Robert J. Tibshirani, "An Introduction to the
         Bootstrap". Chapman & Hall, 1993.
[Koh95]: Ron Kohavi, "A Study of Cross-Validation and Bootstrap for
         Accuracy Estimation and Model Selection".  International
         Joint Conference on Artificial Intelligence, 1995.
[ET97]:  Bradley Efron and Robert Tibshirani. "Improvements on
         Cross-Validation: The .632+ Bootstrap Method". Journal of the
         American Statistical Association, Vol. 92, No. 438.  June
         1997, pp. 548--560.
[Arl10]: Sylvain Arlot, "A Survey of Cross-Validation Procedures for
         Model Selection". Statistics Surveys, Vol. 4, 2010.

"""

# Type aliases (public)
# Public utility function
from bootstrap_stat._utils import ArrayLike, JackknifeValues, Statistic, loess

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
    # Utility
    "loess",
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
