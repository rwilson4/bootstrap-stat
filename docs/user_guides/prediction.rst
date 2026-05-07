Prediction Intervals
====================

A confidence interval and a prediction interval answer related but
fundamentally different questions. A confidence interval covers a
*parameter* — a fixed but unknown number, like the population mean
:math:`\mu`. A prediction interval covers a future *observation* —
a random quantity drawn from the same distribution as the data, like
next year's measurement. The targets of the two intervals are
qualitatively different, and that difference shows up in the width:
a prediction interval is always wider than a confidence interval at
the same nominal level, and the gap grows with sample size rather
than shrinking.

The reason is decomposable. Predicting a new observation
:math:`X_{n+1}` introduces two independent sources of uncertainty.
The first is estimation uncertainty about the population
:math:`F` itself — the same uncertainty a confidence interval
captures. The second is the *inherent variability* of
:math:`X_{n+1}`: even if we knew :math:`F` exactly, a new draw
would still vary. As :math:`n \to \infty` the first source vanishes,
but the second does not. The confidence interval for :math:`\mu`
shrinks to a point in the limit; the prediction interval converges
to a fixed-width band determined by :math:`F`'s spread. The two
intervals are not the same thing, and a prediction interval is
never a "more conservative confidence interval".

We illustrate this on the rainfall data from Table 4.2 of [ET93]_:
:math:`n = 106` years of yearly rainfall measurements, in inches,
recorded in Nevada City, California from 1873 through 1978. The
sample is large enough that the confidence interval for the
population mean is narrow, and the contrast with the prediction
interval is dramatic.

Setting up
----------

The data are univariate — one rainfall measurement per year — and
we treat the years as exchangeable. (Real rainfall has temporal
structure that this assumption ignores; the example is for
illustration of the bootstrap, not for serious climate prediction.)

.. code-block:: python

    import numpy as np
    import bootstrap_stat as bp
    from bootstrap_stat.datasets import rainfall_data

    df = rainfall_data()
    x = df["rainfall"].values        # n = 106
    x_bar = x.mean()                  # 52.66 inches
    s     = x.std(ddof=1)             # 15.43 inches

    dist = bp.EmpiricalDistribution(x)

The mean annual rainfall over the observed century is
:math:`\bar{x} = 52.7` inches with sample standard deviation
:math:`s = 15.4`. Two natural questions follow:

- What is the population mean :math:`\mu = \mathrm{E}[X]`? — call
  for a confidence interval.
- What might next year's rainfall :math:`X_{n+1}` be? — call for a
  prediction interval.

The two are computed from the same data but answer different
questions, and we will see they differ in width by an order of
magnitude.

Why prediction intervals are wider
----------------------------------

Under the textbook normal model :math:`X \sim \mathcal{N}(\mu,
\sigma^2)`, both intervals have closed-form expressions. The 90%
confidence interval for :math:`\mu` is

.. math::

   \bar{x} \pm t_{n-1,\,1-\alpha} \cdot \frac{s}{\sqrt{n}},

with width :math:`\propto 1 / \sqrt{n}`. The 90% prediction interval
for :math:`X_{n+1}` is

.. math::

   \bar{x} \pm t_{n-1,\,1-\alpha} \cdot s \sqrt{1 + \tfrac{1}{n}},

with width :math:`\propto \sqrt{1 + 1/n}`. The ratio of widths is

.. math::

   \frac{\text{PI width}}{\text{CI width}} = \sqrt{n + 1}
   \;\approx\; \sqrt{n}.

For :math:`n = 106` that ratio is 10.3 — and the actual numbers
bear it out:

.. code-block:: python

    from scipy import stats

    n = len(x)
    t_crit = stats.t.ppf(1 - 0.05, df=n - 1)

    ci_lo = x_bar - t_crit * s / np.sqrt(n)
    ci_hi = x_bar + t_crit * s / np.sqrt(n)
    # CI for mu:        (50.17, 55.15)   width 4.97 inches

    pi_lo = x_bar - t_crit * s * np.sqrt(1 + 1/n)
    pi_hi = x_bar + t_crit * s * np.sqrt(1 + 1/n)
    # PI for X_{n+1}:   (26.94, 78.38)   width 51.44 inches

The confidence interval for the population mean is a narrow
five-inch band. The prediction interval for next year is a fifty-inch
band — ten times wider, just as :math:`\sqrt{n + 1}` predicts.

The decomposition makes this concrete. Under the normal model,
:math:`\mathrm{Var}(\bar{x} - X_{n+1}) = \sigma^2 / n + \sigma^2 =
\sigma^2 (1 + 1/n)`, exactly the two terms in the formula. The
:math:`1/n` term is the estimation variance — uncertainty about
:math:`\mu`. The :math:`1` is the inherent variance of a single
draw — uncertainty about :math:`X_{n+1}` itself. Doubling the
sample size halves the first; it leaves the second untouched. As
:math:`n` grows the prediction interval converges to
:math:`\mu \pm z_{1-\alpha} \sigma`, a band whose width is fixed by
:math:`F`'s spread regardless of how much data we collect.

This asymptotic behavior is the cleanest way to see that the two
intervals answer different questions. A confidence interval gets
better with data; a prediction interval gets better only up to the
floor set by the data-generating distribution.

Classical prediction interval
-----------------------------

For reference and comparison, the formula above gives:

.. code-block:: python

    pi_lo, pi_hi = (26.94, 78.38)   # 90% classical PI; < 0.001 s

The classical PI is exact under bivariate normality of the data and
remains a reasonable approximation when the distribution is not too
skewed. For rainfall data the empirical distribution has skewness
:math:`\approx 0.38` and a Shapiro-Wilk test does not reject
normality (:math:`p = 0.16`), so the classical interval is a
defensible baseline here. For data that are clearly non-normal —
heavy-tailed, bounded, or with mass at zero — the classical formula
will mis-state the asymmetry of the interval.

Bootstrap prediction interval
-----------------------------

:func:`~bootstrap_stat.prediction.prediction_interval` constructs a
distribution-free PI using a studentized bootstrap. For each of
:math:`B` replicates it draws a bootstrap sample
:math:`x^* = (X_1^*, \ldots, X_n^*)` and one extra observation
:math:`z^*` from the empirical distribution, then forms the
studentized "prediction error"

.. math::

   t^* = \frac{\bar{x}^* - z^*}{s^*}.

The :math:`\alpha` and :math:`1 - \alpha` quantiles of :math:`t^*`
are inverted to give

.. math::

   \big( \bar{x} - t^*_{1-\alpha} \cdot s,\;
         \bar{x} - t^*_{\alpha}   \cdot s \big).

The construction mirrors the classical *t*-distribution argument
exactly, but reads the quantiles off the empirical bootstrap
distribution rather than the *t*\ :sub:`n-1` table. No normality
assumption is required; the only assumption is that the studentized
prediction error is approximately pivotal — that its distribution
is roughly the same under :math:`\hat{F}` as under :math:`F`.

.. code-block:: python

    pi_lo, pi_hi = bp.prediction_interval(dist, x, B=2000, alpha=0.05)
    # (30.29, 79.22)   width 48.93 inches   ~0.1 s

The bootstrap interval is shifted upward relative to the classical
one — its lower bound is 30.3 against the classical 26.9, while
the upper bounds are nearly identical. This shift reflects the
empirical distribution's right skewness: rainfall is bounded below
by zero and has a longer right tail, so the lower extreme of a
plausible new draw is less extreme than a symmetric normal model
would predict. The bootstrap captures that asymmetry; the classical
formula imposes symmetry around :math:`\bar{x}` whether the data
support it or not.

For data closer to normal the bootstrap and classical intervals
agree to within Monte Carlo noise. The bootstrap doesn't invent
differences when the assumption holds — it just doesn't impose the
assumption when it doesn't. That is the practical case for using it
even when normality looks plausible: nothing is lost, and asymmetry
is captured automatically when present.

The bootstrap PI is computationally cheap. With :math:`B = 2000`
replicates the wall time is well under a second on a single thread,
because each replicate is a single mean and standard deviation. For
larger samples or more demanding statistics multithreading is
available via the ``num_threads`` argument.

A note on prediction error
--------------------------

The same module also exposes
:func:`~bootstrap_stat.prediction.prediction_error_optimism` and
:func:`~bootstrap_stat.prediction.prediction_error_632`, which
solve a related but distinct problem. Where
:func:`~bootstrap_stat.prediction.prediction_interval` quantifies a
*range* for a new observation drawn from a univariate distribution,
the prediction-error functions estimate a *scalar* error rate of a
fitted model on new observations — the quantity used for model
selection. Conceptually:

- **Prediction interval**: how variable is :math:`X_{n+1}`? (range)
- **Prediction error**: how accurate is :math:`\hat{f}(X_{n+1})`?
  (scalar)

The two share the bootstrap idea — resample to estimate behavior on
data the model has not seen — but the quantities returned and the
typical use cases are different. The .632 bootstrap was historically
popular for model selection but has not held up as well as
cross-validation [Koh95]_ [Arl10]_; the implementation in this
library is best treated as illustrative. For prediction *intervals*,
:func:`~bootstrap_stat.prediction.prediction_interval` remains the
straightforward and effective tool.

Comparison
----------

All intervals below are nominal 90% (:math:`\alpha = 0.05`) on the
rainfall data (:math:`n = 106`, :math:`\bar{x} = 52.66`,
:math:`s = 15.43`):

.. list-table::
   :header-rows: 1
   :widths: 32 14 14 12 16

   * - Method
     - Lower
     - Upper
     - Width
     - Wall time
   * - **CI for** :math:`\mu` (classical)
     - 50.17
     - 55.15
     - 4.97
     - < 0.001 s
   * - **CI for** :math:`\mu` (BCa, B=2000)
     - 50.22
     - 55.05
     - 4.83
     - 0.2 s
   * - **PI for** :math:`X_{n+1}` (classical)
     - 26.94
     - 78.38
     - 51.44
     - < 0.001 s
   * - **PI for** :math:`X_{n+1}` (bootstrap, B=2000)
     - 30.29
     - 79.22
     - 48.93
     - 0.1 s

The two confidence intervals (classical and BCa) cover the
population mean within a five-inch band — narrow, because the
sample mean is precisely estimated from :math:`n = 106`
observations. The two prediction intervals cover next year's
rainfall within a band ten times wider, because next year's
rainfall is itself variable. Closing that gap is not a question of
gathering more data; the inherent variability of yearly rainfall is
the band's irreducible floor.

For applied work,
:func:`~bootstrap_stat.prediction.prediction_interval` is the
right tool for prediction intervals when distributional assumptions
are doubtful or the sample is small. It costs slightly more than
the classical formula, makes no parametric assumption, and
correctly captures asymmetry in the data-generating distribution
when it is present. See [ET93]_ (S6.5) for the underlying theory.
