Zero-Inflated Bootstrap
=======================

Many real-world distributions are *zero-inflated*: most observations
are exactly zero, and the inferentially interesting variation lives
in a small fraction of non-zero values. Customer revenue per
visitor, insurance claim amounts, ad clicks per impression, daily
purchases per user — the canonical pattern is the same. A few
percent of the units carry the signal; the rest contribute zero,
which is to say nothing.

The naive bootstrap on such data is wasteful. Each bootstrap
replicate draws :math:`n` observations with replacement from
:math:`\hat{F}`, but the vast majority of those draws are zero by
construction and contribute nothing to most statistics. With
:math:`n = 50{,}000` observations and a 2% non-zero rate, a single
bootstrap sample takes about half a millisecond on modern hardware,
of which roughly 99% is spent generating zeros. This guide describes
the *zero-inflated bootstrap*, a clean way to skip that work
entirely. The trick is exact, not approximate, and integrates into
the library through a small
:class:`~bootstrap_stat.distributions.EmpiricalDistribution`
subclass.

The trick
---------

Let :math:`m` denote the number of non-zero values among the
:math:`n` observations and :math:`p = m/n` the non-zero fraction. A
naive bootstrap sample is :math:`X_1^*, \ldots, X_n^*`, drawn iid
from the empirical distribution :math:`\hat{F}`. Decompose each
draw as

.. math::

   X_i^* = \xi_i \cdot Y_i,
   \qquad
   \xi_i \overset{\text{iid}}{\sim} \mathrm{Bernoulli}(p),
   \qquad
   Y_i \overset{\text{iid}}{\sim} \hat{F}_{\text{nonzero}},

where :math:`\hat{F}_{\text{nonzero}}` is the empirical
distribution of the non-zero observations. When :math:`\xi_i = 0`
the draw is zero; when :math:`\xi_i = 1` the draw is one of the
:math:`m` non-zero values, chosen uniformly with replacement. The
two formulations have identical distributions because the
probability of drawing a zero from :math:`\hat{F}` is exactly
:math:`(n - m)/n = 1 - p`.

Now observe that almost every statistic of interest depends on the
:math:`X_i^*` only through their non-zero values. The bootstrap
mean, for example, is

.. math::

   \bar{X}^* = \frac{1}{n} \sum_{i=1}^{n} X_i^*
             = \frac{1}{n} \sum_{i:\, \xi_i = 1} Y_i
             = \frac{S^*}{n},

where :math:`S^* = \sum_{i:\, \xi_i = 1} Y_i` is a sum of
:math:`K = \sum_i \xi_i` iid draws from :math:`\hat{F}_{\text{nonzero}}`,
and :math:`K \sim \mathrm{Binom}(n, p)`. The dependence on the zeros
is purely structural: they show up as a divisor :math:`n`, not as
data we need to materialize.

The optimization is to simulate the structural decomposition
directly. For each replicate:

1. Draw :math:`K \sim \mathrm{Binom}(n, p)`.
2. Draw :math:`Y_1, \ldots, Y_K` iid from
   :math:`\hat{F}_{\text{nonzero}}`.

The compact sample :math:`(Y_1, \ldots, Y_K)` is everything the
statistic needs. Each replicate now does :math:`O(K)` work instead
of :math:`O(n)`, with :math:`\mathrm{E}[K] = np = m`. The asymptotic
speedup is :math:`1/p`.

The example
-----------

We illustrate on a synthetic dataset that mimics e-commerce visitor
revenue: :math:`n = 50{,}000` visitors, of whom about 2% make a
purchase, with order values drawn from a lognormal distribution.

.. code-block:: python

    import numpy as np
    import bootstrap_stat as bp

    np.random.seed(42)
    n = 50_000
    nonzero_mask = np.random.binomial(1, 0.02, n).astype(bool)
    data = np.zeros(n)
    data[nonzero_mask] = np.random.lognormal(mean=4, sigma=1,
                                             size=nonzero_mask.sum())

    # m ≈ 994 (p̂ ≈ 0.0199), sample mean ≈ 1.84

The interest is in the population mean :math:`\mu = \mathrm{E}[X]`
— the average revenue per visitor — and the goal is a standard
error and a confidence interval. The sample mean is
:math:`\hat{\mu} \approx \$1.84`, but the high variance of the
non-zero values (sample sd :math:`\approx 20.8`) means the standard
error is non-negligible relative to that estimate.

The naive bootstrap (baseline)
------------------------------

The naive approach uses
:class:`~bootstrap_stat.distributions.EmpiricalDistribution`
unchanged:

.. code-block:: python

    dist_naive = bp.EmpiricalDistribution(data)
    se = bp.standard_error(dist_naive, np.mean, B=500)
    # se ≈ 0.092   ~0.29 s

It works correctly. With :math:`n = 50{,}000` and :math:`B = 500`,
each replicate samples 50,000 observations and computes their mean.
The wall time is dominated by drawing — and immediately discarding
— roughly :math:`n - m = 49{,}000` zero values per replicate.

The optimized bootstrap
-----------------------

The decomposition above translates into a small subclass:

.. code-block:: python

    class ZeroInflatedDist(bp.EmpiricalDistribution):
        def __init__(self, data):
            super().__init__(data)
            self.nonzero = np.asarray(data)[np.asarray(data) != 0]
            self.m = len(self.nonzero)
            self.p_nonzero = self.m / self.n

        def sample(self, size=None, **kwargs):
            n = self.n if size is None else size
            k = np.random.binomial(n, self.p_nonzero)
            return np.random.choice(self.nonzero, size=k, replace=True)

The override returns the *compact* sample of non-zero values only.
The statistic must accept this representation and account for the
implicit zeros via a fixed denominator:

.. code-block:: python

    def mean_stat(x):
        return x.sum() / n   # n closed over from outer scope

The closure on :math:`n` is the trick. The same function evaluates
correctly whether :math:`x` is the compact bootstrap sample (size
:math:`K`), the full data (size :math:`n`), or a leave-one-out
slice (size :math:`n - 1`): for the zero-inflated mean,
:math:`\mathrm{sum}(x) / n` is the right answer in every case
because the non-included values are all zero anyway. We will see
shortly that this same property makes BCa interval computation work
without further modification.

.. code-block:: python

    dist_opt = ZeroInflatedDist(data)
    se = bp.standard_error(dist_opt, mean_stat, B=500)
    # se ≈ 0.089   ~0.01 s   (29× faster)

The standard error agrees with the naive estimate to within Monte
Carlo noise, as it must — the optimization is an exact
distributional equivalence, not an approximation. The :math:`B = 500`
budget that took 0.29 seconds with the naive bootstrap takes 0.01
seconds with the optimized one. At :math:`B = 5{,}000` the
optimized version takes 0.11 seconds, still faster than the naive
:math:`B = 500` and giving a more precise estimate (:math:`\hat{\mathrm{se}}
\approx 0.094`).

Confidence intervals
--------------------

Percentile and BCa intervals require no further changes for the
percentile case:

.. code-block:: python

    lo, hi = bp.percentile_interval(dist_opt, mean_stat, B=500)
    # (1.701, 1.998)   ~0.01 s   (vs naive 0.32 s)

For the BCa interval, the bootstrap step is sped up the same way,
but the BCa acceleration is computed by *jackknifing* the original
data — :math:`n = 50{,}000` evaluations of the statistic on
leave-one-out slices, which is by far the dominant cost on data
this size. We can sidestep that work too, because for the
zero-inflated mean the jackknife values have a closed form. Letting
:math:`S = \sum_{j} v_j` denote the sum of non-zero values:

.. math::

   \hat{\theta}_{(i)} =
   \begin{cases}
   S / n & \text{if } x_i = 0,\\
   (S - x_i) / n & \text{if } x_i = v_j \neq 0.
   \end{cases}

(Strictly, the natural jackknife uses :math:`n - 1` in the
denominator. The constant factor :math:`(n-1)/n` does not affect
the BCa acceleration, which is invariant under affine
transformations of the jackknife values.)

Pre-computing :math:`\boldsymbol{j}\boldsymbol{v}` and passing it
through the ``jv`` argument bypasses the
:func:`~bootstrap_stat.sampling.jackknife_values` call entirely:

.. code-block:: python

    sum_nz = dist_opt.nonzero.sum()
    jv = np.full(n, sum_nz / n)                       # all zero observations
    nonzero_idx = np.where(data != 0)[0]
    jv[nonzero_idx] = (sum_nz - data[nonzero_idx]) / n

    lo, hi = bp.bcanon_interval(dist_opt, mean_stat, data, B=500, jv=jv)
    # (1.718, 2.026)   ~0.01 s   (vs naive 3.18 s — 318× faster)

The interval matches the naive computation to Monte Carlo noise,
arriving in 0.01 seconds against the naive 3.18.

Speedup as a function of sparsity
---------------------------------

The theoretical speedup is :math:`1/p`. In practice fixed overhead
from the bootstrap loop and the random number generation puts a
ceiling on it, so the speedup saturates at very low :math:`p`:

.. list-table::
   :header-rows: 1
   :widths: 12 18 18 14 14

   * - :math:`p`
     - Naive (B=200)
     - Optimized
     - Speedup
     - Theory (:math:`1/p`)
   * - 0.5
     - 0.12 s
     - 0.06 s
     - 2×
     - 2
   * - 0.1
     - 0.12 s
     - 0.02 s
     - 5×
     - 10
   * - 0.05
     - 0.12 s
     - 0.013 s
     - 10×
     - 20
   * - 0.01
     - 0.11 s
     - 0.004 s
     - 27×
     - 100
   * - 0.005
     - 0.12 s
     - 0.005 s
     - 25×
     - 200

For data with single-digit-percent non-zero rates the optimization
delivers a one-to-two order of magnitude speedup; below
:math:`p \approx 0.005` the absolute time is so small that fixed
overhead dominates and the ratio plateaus around 25×. The naive
bootstrap remains :math:`O(n)` per replicate regardless of
:math:`p`, so the *absolute* savings continue to grow as the data
becomes sparser even when the *ratio* does not.

A general principle
-------------------

The zero-inflated optimization is one instance of a broader pattern.
The bootstrap principle says: replace :math:`F` with :math:`\hat{F}`
and resample. *How* we resample from :math:`\hat{F}` is up to us,
as long as the bootstrap distribution is preserved. When
:math:`\hat{F}` has structural redundancy — repeated values, a
mixture decomposition, a sparsity pattern, a known stratification —
that structure can be exploited to make sampling cheaper without
changing the result. Stratified bootstrap (sample within strata in
proportion to their original weights), the Bayesian bootstrap (a
re-weighting view of the same idea), and the block bootstrap for
time series are all examples of the same move: factor :math:`\hat{F}`
into pieces, sample each piece efficiently, and recombine. The
zero-inflated trick factors :math:`\hat{F}` as

.. math::

   \hat{F} = (1 - p) \cdot \delta_0 + p \cdot \hat{F}_{\text{nonzero}}

and notes that simulating from a point mass at zero requires no
work at all.

Caveats
-------

The optimization is exact for any statistic that depends on the
data only through sums or related linear functionals — means, sums,
weighted means, second moments built on top of sums and sums of
squares, regression coefficients written as such. For order
statistics (medians, quantiles, extremes) the structure is more
delicate: the median of a zero-inflated sample is zero whenever
:math:`p < 0.5`, and the bootstrap has to track which side of that
threshold each replicate falls on. The compact representation
suffices in principle but the statistic must be written to
interpret it correctly.

The optimization assumes the statistic *requires* knowledge of
:math:`n` — the total sample size — which is why we pass it via
closure rather than via ``len(x)``. Statistics computed by external
libraries that hard-code ``len(x)`` as the sample size cannot be
plugged in directly; a wrapper is needed.

Finally, the speedup is meaningful only when sample-generation cost
dominates statistic-computation cost. For statistics whose
evaluation on a sample of size :math:`n` is itself :math:`O(n)` and
cheap (means, variances), the optimization is effectively pure
profit. For statistics whose per-replicate cost is dominated by the
fitting of a complex model, the bootstrap loop's outer cost
:math:`O(B \cdot n)` is shadowed by :math:`O(B \cdot \text{model
cost})` and the relative gain shrinks accordingly.

For data with many zeros and statistics built on sums, however, the
optimization is essentially free in implementation cost and large
in payoff. A 30× to 300× wall-time reduction on routine analyses is
the kind of improvement that turns "let it run overnight" into "let
it run during a coffee break".
