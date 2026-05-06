Confidence Intervals
====================

The bootstrap is a general technique for quantifying uncertainty that
replaces distributional assumptions with computation. Classical
confidence intervals rely on asymptotic normality or a known parametric
form for the sampling distribution of the statistic --- assumptions that
are reasonable for the sample mean but fail for nonlinear statistics,
small samples, or statistics near the boundary of their parameter space.
The bootstrap sidesteps these assumptions by treating the observed data
as a stand-in for the unknown population and resampling from it
repeatedly to approximate the sampling distribution directly. What
follows illustrates the three main bootstrap confidence interval methods
on a concrete example.

We wish to estimate a confidence interval for the Pearson correlation
between LSAT scores and undergraduate GPA at 15 American law schools
--- the data from Table 3.2 of [ET93]_. The example is a natural
starting point: the sample is small, the statistic is nonlinear, and
the sampling distribution is skewed when the true correlation is near 1.
Each of those features is a problem for normal-theory intervals and
an opportunity for the bootstrap.

Setting up
----------

The core workflow is always the same. We wrap the observed data in an
:class:`~bootstrap_stat.distributions.EmpiricalDistribution`, which places
probability :math:`1/n` on each observation, and define the statistic
as a plain Python function.

.. code-block:: python

    import numpy as np
    import bootstrap_stat as bp
    from bootstrap_stat.datasets import law_data

    data = law_data()

    def correlation(df):
        return np.corrcoef(df["LSAT"], df["GPA"])[0, 1]

    dist = bp.EmpiricalDistribution(data)
    theta_hat = correlation(data)   # 0.7764

We have :math:`n = 15` observations and an observed correlation of
:math:`\hat{\theta} = 0.776`. All three confidence intervals below use
the same ``dist`` and ``correlation`` objects.

.. note::

   All intervals in this library use a *per-tail* :math:`\alpha`
   convention. The default ``alpha=0.05`` produces a **90%** interval
   --- :math:`100(1 - 2 \times 0.05)\% = 90\%` --- not 95%. Pass
   ``alpha=0.025`` for a 95% interval.

Percentile interval
-------------------

:func:`~bootstrap_stat.confidence.percentile_interval` reads the :math:`\alpha`
and :math:`1 - \alpha` quantiles of the bootstrap distribution of
:math:`\hat{\theta}^*` directly.

.. code-block:: python

    lo, hi = bp.percentile_interval(dist, correlation, B=2000)
    # (0.521, 0.949)   ~0.23 s

The percentile interval is the method most commonly described in
introductions to the bootstrap, and its appeal is clear: draw bootstrap
samples, evaluate the statistic on each, take quantiles. No additional
machinery required.

Its limitation is accuracy. The coverage error is :math:`O(n^{-1/2})`:
the true coverage probability deviates from the nominal
:math:`1 - 2\alpha` by a term proportional to :math:`1/\sqrt{n}`. For
:math:`n = 15`, that is roughly a 26% scaling factor on any bias or
skewness in the bootstrap distribution. Concretely, an interval stated
as 90% might actually cover the true parameter 82% or 97% of the time,
depending on direction. The larger the skewness and the smaller the
sample, the worse this gets.

BCa interval
------------

:func:`~bootstrap_stat.confidence.bcanon_interval` adjusts the percentile
endpoints before reading them off. Two quantities are estimated from
the data: the *bias-correction* :math:`\hat{z}_0`, which accounts for
median bias in the bootstrap distribution, and the *acceleration*
:math:`\hat{a}`, which accounts for the rate at which the standard
error of the statistic changes with the parameter.

.. code-block:: python

    lo, hi = bp.bcanon_interval(dist, correlation, data, B=2000)
    # (0.438, 0.927)   ~0.23 s

BCa requires the original data ``data`` as a third argument, beyond
the empirical distribution and the statistic, because computing
:math:`\hat{a}` requires jackknife values.

The BCa interval achieves :math:`O(n^{-1})` coverage error, an order
of magnitude more accurate than the percentile interval for large
:math:`n` --- and at essentially the same computational cost. The
difference is not merely theoretical here: the lower bound shifts from
0.521 to 0.438, reflecting the left skewness of the correlation's
sampling distribution when the true value is near 1. The percentile
interval, insensitive to that skewness, overstates how far the
correlation could plausibly be below :math:`\hat{\theta}`.

ABC interval
------------

:func:`~bootstrap_stat.confidence.abcnon_interval` is an analytical
approximation to the BCa interval that skips bootstrap sampling
entirely. Rather than drawing replicates, it evaluates the statistic at
nearby perturbed weight vectors to estimate the influence components and
second derivatives that BCa would compute from jackknife and bootstrap
samples. The result is second-order accurate and arrives in
milliseconds.

The catch is that the statistic must be supplied in *resampling form*:
a function of both the data and a weight vector ``p``, where
``p = np.ones(n) / n`` corresponds to equal weights on all observations.
For the correlation this amounts to a weighted version of the formula:

.. code-block:: python

    def correlation_p(df, p):
        lsat = df["LSAT"].values
        gpa  = df["GPA"].values
        mean_l = np.dot(p, lsat)
        mean_g = np.dot(p, gpa)
        cov    = np.dot(p, lsat * gpa) - mean_l * mean_g
        var_l  = np.dot(p, lsat**2)    - mean_l**2
        var_g  = np.dot(p, gpa**2)     - mean_g**2
        return cov / np.sqrt(var_l * var_g)

    lo, hi = bp.abcnon_interval(data, correlation_p)
    # (0.443, 0.921)   ~0.001 s

Note that ``abcnon_interval`` takes the raw data directly --- no
:class:`~bootstrap_stat.distributions.EmpiricalDistribution` is needed.

The interval (0.44, 0.92) is nearly identical to BCa's (0.44, 0.93)
and arrives about 200 times faster. The resampling-form requirement is
the tradeoff: writing the weighted statistic is straightforward for
smooth algebraic statistics like the correlation, but infeasible for
statistics computed by external libraries or that involve sorting (the
median, for instance). When it applies, ABC is the most efficient route
to a second-order accurate interval.

Bootstrap-t interval
--------------------

:func:`~bootstrap_stat.confidence.t_interval` also achieves :math:`O(n^{-1})`
accuracy, via pivotalization rather than endpoint adjustment. For each of the
:math:`B_\mathrm{outer}` bootstrap samples it forms

.. math::

   z^* = \frac{\hat{\theta}^* - \hat{\theta}}{\widehat{\mathrm{se}}^*},

where :math:`\widehat{\mathrm{se}}^*` is estimated by an inner bootstrap
of size :math:`B_\mathrm{inner}` on the bootstrap sample itself. The
interval inverts the :math:`\alpha` and :math:`1 - \alpha` quantiles of
:math:`z^*`.

.. code-block:: python

    lo, hi = bp.t_interval(dist, correlation, theta_hat, Bouter=2000, Binner=25)
    # (0.012, 0.960)   ~6.1 s

With 2000 outer and 25 inner samples, the bootstrap-t runs 50,000
bootstrap evaluations versus 2,000 for BCa --- roughly 26 times slower.
And the lower bound of 0.01 is implausibly small for data showing clear
positive correlation. Both problems trace back to the same cause.

**Why the pivot fails for correlation.** The bootstrap-t is well
calibrated only when :math:`z^*` has approximately the same distribution
regardless of the true parameter value --- that is, when the pivot is
*pivotal*. For the sample correlation this fails badly: the standard
error is approximately :math:`(1 - \rho^2)/\sqrt{n}`, which equals 0.26
at :math:`\rho = 0.1` but only 0.025 at :math:`\rho = 0.95` --- a
tenfold variation. Bootstrap samples that happen to land near
:math:`\hat{\rho} = 1` have a near-zero denominator
:math:`\widehat{\mathrm{se}}^*`, producing enormous values of
:math:`z^*` that drag the lower quantile far to the left. The result is
a wildly conservative lower bound.

**Fisher's z-transform.** The classical fix is Fisher's
:math:`z`-transform, :math:`z = \mathrm{arctanh}(r)`, whose variance is
approximately :math:`1/(n - 3)` regardless of :math:`\rho`. We apply
the bootstrap-t in :math:`z`-space and transform the resulting endpoints
back.

.. code-block:: python

    def fisher_z(df):
        return np.arctanh(correlation(df))

    theta_hat_z = np.arctanh(theta_hat)

    lo_z, hi_z = bp.t_interval(dist, fisher_z, theta_hat_z, Bouter=2000, Binner=25)
    lo, hi = np.tanh(lo_z), np.tanh(hi_z)
    # (0.153, 0.925)   ~6.1 s

The lower bound rises from 0.01 to 0.15 --- a substantial improvement,
though still below the BCa estimate of 0.44. Some residual imprecision
is expected with only :math:`n = 15` observations: even a
variance-stabilized pivot is not exactly pivotal in small samples.

**Automated variance stabilization.** Knowing the right transformation
analytically requires statistical knowledge that may not always be at
hand. The ``stabilize_variance=True`` option estimates a
variance-stabilizing transformation numerically: it runs a small
preliminary bootstrap (``Bvar`` samples) to estimate how the standard
error varies with the statistic, fits a smoother to that relationship,
and then applies the implied transformation automatically.

.. code-block:: python

    lo, hi = bp.t_interval(dist, correlation, theta_hat,
                           stabilize_variance=True, Bvar=200, Bouter=2000)
    # (0.409, 0.917)   ~4.4 s

The result is (0.41, 0.92) --- close to the BCa interval --- and it
arrives *faster* than the baseline bootstrap-t. Because the outer
bootstrap uses a constant standard error of 1 in the transformed space,
no inner bootstrap is needed; the only inner-bootstrap cost is the
``Bvar=200`` preliminary samples used to estimate the transformation.

Comparison
----------

All intervals below are nominal 90% (:math:`\alpha = 0.05`), computed
with ``B=2000`` (or ``Bouter=2000``) bootstrap samples on the law
school data (:math:`n = 15`):

.. list-table::
   :header-rows: 1
   :widths: 36 12 12 16

   * - Method
     - Lower
     - Upper
     - Wall time
   * - Percentile
     - 0.521
     - 0.949
     - 0.23 s
   * - BCa
     - 0.438
     - 0.927
     - 0.23 s
   * - ABC
     - 0.443
     - 0.921
     - 0.001 s
   * - Bootstrap-t (baseline)
     - 0.012
     - 0.960
     - 6.1 s
   * - Bootstrap-t (Fisher's z)
     - 0.153
     - 0.925
     - 6.1 s
   * - Bootstrap-t (stabilize_variance)
     - 0.409
     - 0.917
     - 4.4 s

For most applications, :func:`~bootstrap_stat.confidence.bcanon_interval`
is the right default: it matches the percentile interval on speed,
surpasses it on accuracy, and requires no special form for the
statistic. When the statistic can be written in resampling form,
:func:`~bootstrap_stat.confidence.abcnon_interval` delivers
second-order accuracy at negligible cost. The bootstrap-t with
``stabilize_variance=True`` is worth considering when a pivotal
interval is specifically required. See [ET93]_ (S14) for a thorough
comparison of interval methods.
