bootstrap-stat
==============

Bootstrap-stat implements bootstrap methods for statistical inference,
following Efron & Tibshirani's *An Introduction to the Bootstrap*
[ET93]_. The bootstrap provides distribution-free estimates of standard
errors, confidence intervals, and significance levels by resampling the
observed data rather than assuming a parametric model.

The library provides:

- **Standard errors**: bootstrap, jackknife, and infinitesimal jackknife
- **Confidence intervals**: percentile, BCa (bias-corrected and
  accelerated), ABC, bootstrap-t, and calibrated
- **Bias estimation and correction**
- **Significance testing**: achieved significance levels and power
  analysis
- **Prediction error**: optimism method, .632 and .632+ bootstrap

Quickstart
----------

The core workflow: wrap observed data in an
:class:`~bootstrap_stat.distributions.EmpiricalDistribution`, define a statistic
function, then call whichever inference function you need.
:func:`~bootstrap_stat.confidence.bcanon_interval` is the recommended default for
confidence intervals.

.. code-block:: python

    import numpy as np
    import bootstrap_stat as bp

    # Law school data: LSAT/GPA for n=15 schools [ET93], Table 3.2
    df = bp.law_data()

    def correlation(df):
        return np.corrcoef(df["LSAT"], df["GPA"])[0, 1]

    dist = bp.EmpiricalDistribution(df)
    theta_hat = correlation(df)
    lo, hi = bp.bcanon_interval(dist, correlation, theta_hat)
    # 90% BCa confidence interval for the correlation

The methods in this library are described in detail in [ET93]_. The
.632+ bootstrap is from [ET97]_; [Koh95]_ and [Arl10]_ discuss its
limitations relative to cross-validation. Full citations are on the
:doc:`references` page.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   guide.rst
   api.rst
   references.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
