Standard Errors
===============

A standard error answers a simple question: if we repeated the
experiment, how much would the estimate change? It is the building
block for nearly every other piece of statistical inference. A
confidence interval is, in spirit, the estimate plus or minus a few
standard errors. A *t*-statistic is the estimate divided by its
standard error. A reported uncertainty in a scientific paper is, more
often than not, a standard error or its derivative. Get the standard
error right and the rest of the inferential machinery follows.

For the sample mean, the standard error has a clean closed-form
expression: :math:`s / \sqrt{n}`. For almost every other statistic, it
does not. The sample median, a trimmed mean, a ratio of means, the
slope of a robust regression --- each requires either a delta-method
derivation, a parametric assumption, or both. The bootstrap and
jackknife sidestep that work by estimating the standard error directly
from the data: draw resamples, evaluate the statistic on each, and
read the spread off the resulting distribution. What follows
illustrates the three main methods --- the bootstrap, the jackknife,
and the infinitesimal jackknife --- on a small concrete example, with
attention to the cases where each one succeeds or fails.

We use the mouse survival data from Table 2.2 of [ET93]_: survival
times, in days, of :math:`n = 9` mice in the control arm of a
treatment experiment. The example is deliberately small. With only
nine observations the standard error itself has substantial sampling
variability, and the contrast between the sample mean (smooth) and
the sample median (non-smooth) cleanly separates the methods that
work everywhere from the methods that don't.

Setting up
----------

The core workflow mirrors the rest of the library. We wrap the
observed data in an
:class:`~bootstrap_stat.distributions.EmpiricalDistribution`, which
places probability :math:`1/n` on each observation, and define the
statistics as plain Python functions.

.. code-block:: python

    import numpy as np
    import bootstrap_stat as bp
    from bootstrap_stat.datasets import mouse_data

    data = np.array(mouse_data("control"))   # n = 9 survival times

    def mean(x):
        return np.mean(x)

    def median(x):
        return np.median(x)

    dist = bp.EmpiricalDistribution(data)
    theta_mean   = mean(data)     # 56.22
    theta_median = median(data)   # 46.0

We have :math:`n = 9` observations, a sample mean of
:math:`\bar{x} = 56.2` days, and a sample median of 46 days. All
methods below use the same ``data`` and ``dist`` objects.

Classical standard error (sample mean)
--------------------------------------

As a baseline, the textbook formula for the standard error of the
sample mean uses the unbiased sample standard deviation:

.. math::

   \widehat{\mathrm{se}}(\bar{x}) = \frac{s}{\sqrt{n}},
   \qquad
   s^2 = \frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \bar{x})^2.

.. code-block:: python

    se = np.std(data, ddof=1) / np.sqrt(len(data))
    # 14.16   < 0.001 s

The formula requires the statistic to be a sample mean. For the
median, the corresponding asymptotic result is
:math:`\widehat{\mathrm{se}} \approx 1 / \big(2 f(\theta) \sqrt{n}\big)`,
which depends on the unknown density :math:`f` at the median --- a
quantity that itself must be estimated. Already the limitations of the
classical approach are visible: closed-form standard errors exist only
for a handful of textbook statistics, and for everything else the
analytical work is either nontrivial or unavailable.

Bootstrap standard error
------------------------

:func:`~bootstrap_stat.standard_error.standard_error` draws
:math:`B` bootstrap samples from the empirical distribution, evaluates
the statistic on each, and returns the standard deviation of the
resulting :math:`B` values.

.. code-block:: python

    se_mean   = bp.standard_error(dist, mean,   B=2000)
    se_median = bp.standard_error(dist, median, B=2000)
    # 13.41   ~0.04 s    (mean)
    # 12.83   ~0.04 s    (median)

The bootstrap standard error of the mean, 13.41, is close to but
slightly below the classical value of 14.16. The discrepancy is not
sampling noise: it persists as :math:`B \to \infty`. The bootstrap
uses the empirical variance with :math:`1/n` in the denominator,
whereas :math:`s^2` uses :math:`1/(n-1)`. The two differ by a factor
of :math:`\sqrt{(n-1)/n} = \sqrt{8/9} \approx 0.943`, and indeed
:math:`0.943 \times 14.16 = 13.35`, the asymptotic bootstrap value.
For :math:`n = 9` this matters; for :math:`n = 100` it is negligible.

The bootstrap standard error of the median is 12.83. There is no
classical formula to compare it against without estimating the density
at the median, which is itself a nontrivial smoothing problem. The
bootstrap delivers the answer with the same code, no derivation
required.

How many bootstrap samples are needed? Far fewer than for confidence
intervals. The default ``B=200`` is usually sufficient for a standard
error to two significant figures, because estimating a single number
(a standard deviation) is much easier than estimating two quantiles in
the tails of a distribution. ``B=2000`` is conservative; ``B=50`` is
often enough for an exploratory pass.

Jackknife standard error
------------------------

:func:`~bootstrap_stat.standard_error.jackknife_standard_error`
estimates the standard error from the :math:`n` leave-one-out values
of the statistic:

.. math::

   \widehat{\mathrm{se}}_{\mathrm{jack}} =
   \sqrt{ \frac{n - 1}{n} \sum_{i=1}^{n} (\hat{\theta}_{(i)} - \hat{\theta}_{(\cdot)})^2 },

where :math:`\hat{\theta}_{(i)}` is the statistic evaluated on the
sample with the :math:`i`\ th observation removed.

.. code-block:: python

    se_mean   = bp.jackknife_standard_error(data, mean)
    se_median = bp.jackknife_standard_error(data, median)
    # 14.16   < 0.001 s    (mean)
    #  7.34   < 0.001 s    (median)

For the mean, the jackknife reproduces the classical formula
*exactly*: the algebraic identity
:math:`\widehat{\mathrm{se}}_{\mathrm{jack}}(\bar{x}) = s / \sqrt{n}`
is not an approximation but an equality. And it does so in :math:`n`
statistic evaluations rather than :math:`B` --- two orders of
magnitude fewer for the same answer.

For the median, the jackknife reports 7.34 against the bootstrap's
12.83, less than 60% of the bootstrap value. This is not a small-sample
fluctuation. The jackknife is *inconsistent* for the median: even as
:math:`n \to \infty`, the jackknife standard error does not converge
to the true sampling standard deviation. The reason is that the
jackknife implicitly approximates the statistic by its first-order
Taylor expansion in the empirical distribution, and the median is not
smooth in that distribution --- removing a single observation can
leave the median essentially unchanged or shift it by half a gap,
depending on which observation. For :math:`n = 9` and :math:`\hat{\theta}
= 46`, four of the nine leave-one-out medians equal 46 exactly and
the rest cluster nearby, dramatically understating the variability.
See [ET93]_ (S10.6) for the formal statement. The practical rule is
simple: use the jackknife only for smooth statistics. The mean,
correlation, regression coefficients, smooth functionals of moments
--- yes. Quantiles, modes, ranks --- no.

Infinitesimal jackknife
-----------------------

:func:`~bootstrap_stat.standard_error.infinitesimal_jackknife` is the
limit of the jackknife as the leave-one-out perturbation shrinks to
zero. Rather than removing an observation entirely, it places a small
extra weight :math:`\epsilon` on each observation in turn and
differentiates: the *influence component*
:math:`U_i = \lim_{\epsilon \to 0} [\hat{\theta}(\mathbf{p} +
\epsilon \mathbf{e}_i) - \hat{\theta}(\mathbf{p})] / \epsilon`. The
standard error follows from
:math:`\widehat{\mathrm{se}}_{\mathrm{IJ}} = \sqrt{\sum_i U_i^2 / n^2}`.

The catch is that the statistic must be supplied in *resampling form*
--- a function of both the data and a weight vector ``p``, where
``p = np.ones(n) / n`` corresponds to equal weights. For the mean:

.. code-block:: python

    def mean_p(x, p):
        return np.dot(p, x)

    se = bp.infinitesimal_jackknife(data, mean_p)
    # 13.35   < 0.001 s

The result, 13.35, matches the asymptotic limit of the bootstrap
standard error to three significant figures --- which is no
coincidence. The infinitesimal jackknife and the
:math:`B \to \infty` bootstrap agree exactly for any statistic that is
smooth in the resampling weights, with the bootstrap value differing
only by Monte Carlo noise at finite :math:`B`. It computes the same
answer in :math:`n` finite-difference evaluations rather than :math:`B`
bootstrap replicates --- two to three orders of magnitude fewer.

The resampling-form requirement is the same tradeoff that appears in
:func:`~bootstrap_stat.confidence.abcnon_interval`. For smooth
algebraic statistics --- means, weighted regression coefficients,
ratios --- the weighted form is straightforward. For statistics
computed by external libraries, or that involve sorting or ranks (the
median, again), the requirement is infeasible and the bootstrap is
the only option.

Robust standard error
---------------------

The standard deviation is sensitive to the tails of the bootstrap
distribution: a handful of unusually large or small replicates can
inflate :math:`\widehat{\mathrm{se}}`. When the bootstrap distribution
is heavy-tailed --- common for ratios, products, and statistics
involving small denominators --- a percentile-based standard error
provides a robust alternative. Passing
``robustness=`` :math:`\alpha` to ``standard_error`` returns

.. math::

   \widehat{\mathrm{se}}_{\mathrm{rob}} =
   \frac{q_\alpha - q_{1 - \alpha}}{2 z_\alpha},

where :math:`q_\alpha` is the :math:`\alpha`-quantile of the
bootstrap distribution and :math:`z_\alpha = \Phi^{-1}(\alpha)`. For
:math:`\alpha = 0.84` this matches the standard deviation under
normality but down-weights the influence of extreme replicates.

.. code-block:: python

    se = bp.standard_error(dist, mean, B=2000, robustness=0.84)
    # 13.63   ~0.04 s

For the mouse-control mean the robust estimate (13.63) is within Monte
Carlo noise of the non-robust estimate (13.41). The diagnostic value
is in the comparison: if the two differ by more than a few percent,
the bootstrap distribution has heavy tails and the unprotected
standard deviation is overstating typical sampling variation. See
[ET93]_ (S10.3).

Stability of the SE estimate
----------------------------

How precise is the bootstrap standard error itself? With
``jackknife_after_bootstrap=True``, ``standard_error`` returns a
second number quantifying the variability of the SE estimate as a
function of the data:

.. code-block:: python

    se, se_jack = bp.standard_error(dist, mean, B=2000,
                                    jackknife_after_bootstrap=True)
    # se = 13.41,  se_jack = 5.10   ~0.08 s

The interpretation is that, if the experiment were repeated, the
bootstrap standard error would itself vary with standard deviation
roughly 5.1. That number is large compared to the SE of 13.4 ---
because :math:`n = 9` is small. The Monte Carlo contribution from
finite :math:`B` is negligible at :math:`B = 2000`; almost all of
``se_jack`` reflects the genuine sampling variability of
:math:`s / \sqrt{n}` itself. Put differently, with nine mice the data
are the bottleneck, not the bootstrap.

The same diagnostic on a larger sample would return a much smaller
``se_jack`` --- which is the intended use: to confirm that the
reported standard error is not itself dominated by sampling noise.
See [ET93]_ (S19.4).

Comparison
----------

All numbers below are for the mouse control data
(:math:`n = 9`, :math:`\bar{x} = 56.2`, :math:`\hat{\theta}_{\text{med}}
= 46`), with bootstrap methods at :math:`B = 2000`:

.. list-table::
   :header-rows: 1
   :widths: 32 14 14 16

   * - Method
     - SE (mean)
     - SE (median)
     - Wall time
   * - Classical (:math:`s / \sqrt{n}`)
     - 14.16
     - ---
     - < 0.001 s
   * - Bootstrap
     - 13.41
     - 12.83
     - 0.04 s
   * - Jackknife
     - 14.16
     - 7.34 (inconsistent)
     - < 0.001 s
   * - Infinitesimal jackknife
     - 13.35
     - n/a (not smooth)
     - < 0.001 s
   * - Bootstrap (robust, :math:`\alpha = 0.84`)
     - 13.63
     - ---
     - 0.04 s

For most applications,
:func:`~bootstrap_stat.standard_error.standard_error` is the right
default: it works for any statistic, requires no derivation, and
:math:`B = 200` is typically sufficient.
:func:`~bootstrap_stat.standard_error.jackknife_standard_error` is
preferable for smooth statistics where the :math:`n`-evaluation cost
beats :math:`B` bootstrap replicates --- but it should not be applied
to quantiles or other non-smooth functionals.
:func:`~bootstrap_stat.standard_error.infinitesimal_jackknife` is the
fastest of all when the statistic is naturally available in
resampling form. The robust and jackknife-after-bootstrap options are
diagnostics: useful for assessing whether the reported standard error
is trustworthy, less useful as the headline number. See [ET93]_ (S6,
S10) for a fuller treatment.
