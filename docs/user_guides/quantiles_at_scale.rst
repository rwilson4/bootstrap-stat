Bootstrap Quantiles at Scale
============================

Quantiles are the natural target for inference on heavy-tailed data:
the 95th-percentile request latency, the median order value, the
99th-percentile rendering time. The naive bootstrap recipe applies —
draw :math:`B` samples of size :math:`n` with replacement, compute
the quantile on each, read off the percentiles of the resulting
distribution — and the library supports it directly. But it scales
as :math:`O(B \cdot n)` per replicate (or :math:`O(B \cdot n \log n)`
once sorting is included), and on web-scale data with :math:`n` in
the millions the cost becomes prohibitive: a single 95% confidence
interval on a 10-million-observation sample takes minutes when the
sort is repeated for each replicate.

Schultzberg and Ankargren [SA22]_ — engineers at Spotify — observed
that for sample quantiles, the bootstrap loop has redundant
structure that can be eliminated. The Poisson-bootstrap distribution
of the index of the order statistic that ends up *being* the q-th
quantile of a bootstrap replicate is, for large :math:`n`,
approximately :math:`\mathrm{Bin}(n+1, q)`. So one can sort the
original sample once, sample :math:`B` indices from a binomial, and
look up the corresponding order statistics. No resampling, no
per-replicate sort, no per-replicate quantile evaluation.

The technique is small enough — a few lines of NumPy — that the
library does not ship a dedicated function for it. The bootstrap
machinery already accepts a pre-computed array of bootstrap
statistics through the ``theta_star`` argument of
:func:`~bootstrap_stat.confidence.percentile_interval` and
:func:`~bootstrap_stat.confidence.bcanon_interval`, so the trick
composes cleanly with the existing confidence-interval API. This
guide walks through the construction.

The trick
---------

Let :math:`X_1, \ldots, X_n` be the original sample and write
:math:`X_{(1)} \le \cdots \le X_{(n)}` for the order statistics. A
*Poisson bootstrap* sample assigns each :math:`X_i` an independent
:math:`\mathrm{Poisson}(1)` count :math:`N_i` and constructs the
replicate by including :math:`X_i` exactly :math:`N_i` times. Total
sample size is :math:`M = \sum_i N_i \sim \mathrm{Poisson}(n)`,
random rather than fixed at :math:`n` — this is the price the
Poisson bootstrap pays for the independence between the
:math:`N_i`. For large :math:`n` the Poisson and multinomial
bootstraps agree to leading order; both target the same sampling
distribution.

Now consider the q-th quantile of the bootstrap replicate. Sort the
distinct values; the bootstrap quantile is :math:`X_{(I)}` for some
random index :math:`I`. The index :math:`I` is determined by where
the cumulative bootstrap weight first crosses :math:`q M`. With iid
Poisson counts, [SA22]_ show that for large :math:`n`,

.. math::

   I \;\approx\; \mathrm{Bin}(n + 1,\, q).

The :math:`n+1` (rather than :math:`n`) is a small-sample correction
that aligns the discrete approximation with the continuous Beta
distribution of the corresponding population quantile under
sampling from :math:`\hat{F}`; the approximation is excellent for
:math:`n` even in the low hundreds. The implication is operational:
the entire bootstrap distribution of :math:`\hat{q}^*` lives in the
sorted original sample, indexed by a binomial random variable.

The example
-----------

We illustrate on synthetic latency data: :math:`n = 1{,}000{,}000`
draws from a lognormal distribution with location 4 and scale 1
(median :math:`\approx 55`, p95 :math:`\approx 283`).

.. code-block:: python

    import numpy as np
    import bootstrap_stat as bp

    rng = np.random.default_rng(0)
    n = 1_000_000
    data = rng.lognormal(mean=4.0, sigma=1.0, size=n)

The target is a 90% confidence interval on the 95th percentile.

The naive bootstrap (baseline)
------------------------------

The library supports this directly:

.. code-block:: python

    dist = bp.EmpiricalDistribution(data)

    def p95(x):
        return float(np.quantile(x, 0.95))

    lo, hi = bp.percentile_interval(dist, p95, B=500)
    # CI ≈ (282.55, 284.50)   ~9.7 s

Each of the 500 replicates draws :math:`n = 10^6` observations with
replacement and computes their 95th percentile. NumPy's
``np.quantile`` is :math:`O(n)` (via ``np.partition``), but the
draw-with-replacement step itself is the dominant cost, and it is
repeated :math:`B` times. The wall time is several seconds, and it
scales linearly in :math:`n`.

The optimized recipe
--------------------

The Schultzberg–Ankargren construction reduces the bootstrap step to
a sort plus :math:`B` binomial draws plus :math:`B` lookups:

.. code-block:: python

    def fast_quantile_bootstrap(data, q, B, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        n = len(data)
        sorted_data = np.sort(data)
        idx = rng.binomial(n + 1, q, size=B)
        idx = np.clip(idx, 0, n - 1)        # guard the boundary
        return sorted_data[idx]

The function returns the bootstrap distribution of :math:`\hat{q}^*`
directly. To turn it into a confidence interval, hand it to the
library's existing percentile-interval routine via the
``theta_star`` argument:

.. code-block:: python

    theta_star = fast_quantile_bootstrap(data, q=0.95, B=500, rng=rng)
    lo, hi = bp.percentile_interval(dist, p95, theta_star=theta_star)
    # CI ≈ (282.55, 284.52)   ~0.010 s   (~970× faster)

When ``theta_star`` is supplied, ``percentile_interval`` skips the
resampling step entirely and reads the percentiles off the array
that was passed in. The ``dist`` and ``stat`` arguments are still
required for API symmetry but go unused on this path; ``stat`` is
never called.

The interval matches the naive computation to Monte Carlo noise. The
agreement is not coincidental — both target the same bootstrap
distribution; the difference is between sampling from it the long
way and sampling from a known closed-form approximation to its
quantile-position.

Difference in quantiles
-----------------------

The Spotify use case is the difference of a quantile across two
arms of an A/B test. The construction generalizes immediately: sort
each arm once, sample independent binomial indices for each, and
take the difference of the looked-up order statistics.

.. code-block:: python

    n = 200_000
    ctrl = rng.lognormal(mean=4.00, sigma=1.0, size=n)
    trt  = rng.lognormal(mean=4.05, sigma=1.0, size=n)

    sc = np.sort(ctrl)
    st = np.sort(trt)
    B = 500

    idx_c = np.clip(rng.binomial(n + 1, 0.95, size=B), 0, n - 1)
    idx_t = np.clip(rng.binomial(n + 1, 0.95, size=B), 0, n - 1)
    theta_star = st[idx_t] - sc[idx_c]

    lo, hi = np.quantile(theta_star, [0.05, 0.95])
    # CI ≈ (11.17, 16.87)   ~0.003 s   (vs naive ~3.5 s, ~1130× faster)

The two arms are independent under the null and under the
alternative, so independent binomial draws across arms is correct.
For paired or stratified designs the index distributions are no
longer independent; the technique still applies but the joint
distribution must be specified accordingly, which is no longer a
one-liner.

Scaling
-------

Wall times for ``B = 500`` on a single-arm 95th-percentile
confidence interval, lognormal data:

.. list-table::
   :header-rows: 1
   :widths: 15 18 18 14

   * - :math:`n`
     - Naive
     - Fast
     - Speedup
   * - 10,000
     - 0.18 s
     - 0.0004 s
     - 485×
   * - 100,000
     - 1.16 s
     - 0.0009 s
     - 1340×
   * - 1,000,000
     - 9.74 s
     - 0.010 s
     - 980×
   * - 10,000,000
     - ≳ 100 s (estimated)
     - 0.49 s
     - ≳ 200×

The naive method is :math:`O(B \cdot n)` and tracks linearly in
:math:`n`; the fast method is dominated by the one-time sort
(:math:`O(n \log n)`) plus :math:`O(B)` lookups, and so the speedup
*ratio* shrinks at very large :math:`n` while the *absolute* time
stays small enough to be negligible in any analysis pipeline. At
:math:`n = 10^7` a single-quantile confidence interval comes back in
under a second, against the better part of two minutes for the
naive bootstrap.

BCa intervals at scale
----------------------

The percentile interval has first-order coverage error and ignores
both the median bias and the skewness of the bootstrap distribution
of :math:`\hat{q}^*`. The BCa interval (see :doc:`confidence_intervals`)
corrects both, achieving second-order accuracy, and is the
recommended default for most applications. But BCa as
:func:`~bootstrap_stat.confidence.bcanon_interval` implements it
involves two passes over the data: the bootstrap, and a jackknife
of the original sample, used to estimate the acceleration constant.
The bootstrap step is sped up by the index trick above, but the
jackknife step is :math:`O(n)` evaluations of the statistic and at
large :math:`n` becomes the binding cost.

That step also has structure we can exploit. The argument parallels
the closed-form jackknife for the zero-inflated mean (see
:doc:`zero_inflated`) — different mechanism, same payoff.

Closed-form jackknife for sample quantiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`\hat{q} = X_{(k)}` with :math:`k = \lceil n q \rceil`
denote the order-statistic quantile estimator, and write
:math:`k' = \lceil (n-1) q \rceil` for the analogous index in a
leave-one-out sample of size :math:`n - 1`. Removing one observation
shifts the ranks above it down by one position and leaves the ranks
below it unchanged, so the leave-one-out :math:`k'`-th order
statistic is determined by where the removed observation sat in the
original sort:

.. math::

   \hat{q}_{(i)} =
   \begin{cases}
   X_{(k')}     & \text{if } \mathrm{rank}(X_i) > k' \quad
                  (n - k' \text{ observations}),\\
   X_{(k' + 1)} & \text{if } \mathrm{rank}(X_i) \le k' \quad
                  (k' \text{ observations}).
   \end{cases}

The jackknife array takes *at most two distinct values*. Both come
from the sort the bootstrap step needed anyway. Concretely:

.. code-block:: python

    def quantile_jackknife(data, q):
        """Closed-form jackknife values for the q-th order-statistic
        quantile. Returns an array of length n."""
        n = len(data)
        sorted_data = np.sort(data)
        kp = int(np.ceil((n - 1) * q))
        # rank of each observation in the sorted array (1-indexed)
        order = np.argsort(data, kind="stable")
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n + 1)
        jv = np.where(ranks > kp, sorted_data[kp - 1], sorted_data[kp])
        return jv

Closed form for the acceleration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Plugging the two-valued :math:`\hat{q}_{(i)}` array into the BCa
acceleration formula

.. math::

   \hat{a} = \frac{\sum_{i=1}^n (\hat{q}_{(\cdot)} - \hat{q}_{(i)})^3}
                  {6 \left[\sum_{i=1}^n (\hat{q}_{(\cdot)} - \hat{q}_{(i)})^2\right]^{3/2}}

and letting :math:`\delta = X_{(k'+1)} - X_{(k')}`, the squared- and
cubed-deviation sums work out to :math:`\delta^2 \, k'(n-k') / n`
and :math:`\delta^3 \, k'(n - k')(2k' - n) / n^2`. The
:math:`\delta^3 / \delta^3` cancels — :math:`\hat{a}` is
affine-invariant — and we are left with a closed form that depends
only on :math:`n` and :math:`q` through :math:`k'`:

.. math::

   \hat{a} \;=\; \frac{2k' - n}{6\sqrt{n \cdot k'(n - k')}}.

A few features fall out:

* For the median (:math:`q = 0.5`, :math:`n` even), :math:`k' = n/2`
  and :math:`\hat{a} = 0`. BCa reduces to the bias-corrected (BC)
  interval, which is the known result for symmetric quantile
  estimators.
* For tail quantiles (:math:`q` close to :math:`0` or :math:`1`),
  :math:`|\hat{a}|` grows, capturing the skewness of the bootstrap
  distribution that the percentile interval ignores.
* :math:`\hat{a} = O(1/\sqrt{n})`. The acceleration vanishes as
  :math:`n` grows, and BCa converges to the percentile interval —
  a quantitative version of the intuition that at very large
  :math:`n` the bootstrap distribution of :math:`\hat{q}^*` is
  approximately normal and the BCa adjustment is small.

Putting it together
^^^^^^^^^^^^^^^^^^^

The library accepts both ``theta_star`` and ``jv`` on
:func:`~bootstrap_stat.confidence.bcanon_interval`. Pass both, and
the entire BCa pipeline runs without a single bootstrap resample
*and* without a single jackknife evaluation:

.. code-block:: python

    theta_star = fast_quantile_bootstrap(data, q=0.95, B=500, rng=rng)
    jv         = quantile_jackknife(data, q=0.95)

    lo, hi = bp.bcanon_interval(
        dist, p95, data,
        B=500,
        theta_star=theta_star,
        jv=jv,
    )

Validation against the library's default jackknife on lognormal
data, :math:`q = 0.95`:

.. list-table::
   :header-rows: 1
   :widths: 15 22 22

   * - :math:`n`
     - :math:`\hat{a}` (closed form)
     - :math:`\hat{a}` (jackknife)
   * - 200
     - 0.048666
     - 0.048666
   * - 1,000
     - 0.021764
     - 0.021764
   * - 10,000
     - 0.006882
     - 0.006882
   * - 100,000
     - 0.002176
     - 0.002176

Agreement is exact (to floating-point precision) — the two
expressions are algebraically identical.

End-to-end wall time matches the algebra: at :math:`n = 200{,}000`
and :math:`B = 500`, the full BCa interval via the library default
(internal jackknife of ``np.quantile``) takes about 5 minutes, more
than 99% of which is the jackknife pass. Supplying the closed-form
``theta_star`` and ``jv`` together drops the same call to about 35
milliseconds — a roughly :math:`10^4`-fold speedup, with the
interval matching to Monte Carlo noise. The percentile speedup
shown earlier was three orders of magnitude; eliminating the
jackknife adds the fourth.

A general principle
^^^^^^^^^^^^^^^^^^^

The two-hook BCa pattern — separate the bootstrap step from the
jackknife step, give each its own structure-aware shortcut — recurs
whenever the statistic has exploitable structure on both axes. The
zero-inflated mean has it because the statistic is linear in the
data: bootstraps avoid materializing zeros, and jackknife values
have the closed form :math:`(S - x_i)/n` (or :math:`S/n` for zero
observations). The sample quantile has it because the statistic is
defined by ranks: bootstraps reduce to a binomial draw on
:math:`\mathrm{Bin}(n+1, q)`, and jackknife values take two values
determined by the rank of the removed observation. Different
mechanism, same recipe — the library's separation of ``theta_star``
and ``jv`` arguments is what lets both cases use the same machinery
to achieve order-of-magnitude speedups end-to-end.

Why this is not a library function
-----------------------------------

The optimization is roughly ten lines of NumPy. The corresponding
generic library function would have to expose a quantile-only API
and parameterize over the small set of decisions a caller might
plausibly want to vary — single quantile vs. difference-in-quantile
vs. multiple quantiles, percentile vs. BCa interval, optional RNG,
optional pre-sorted input. The result would be considerably larger
than the recipe itself, and would obscure the fact that the trick is
exact-modulo-Poisson-approximation rather than something the
library is doing on the user's behalf.

The library's design choice is to expose the composable hooks —
``theta_star`` and ``jv`` on the confidence-interval functions, plus
the underlying :class:`~bootstrap_stat.distributions.EmpiricalDistribution`
and :func:`~bootstrap_stat.sampling.bootstrap_samples` for cases
where standard resampling is wanted — and let domain-specific
recipes like this one live in user code or in a guide.

Caveats
-------

The Bin(:math:`n+1`, :math:`q`) result is an approximation to the
Poisson-bootstrap index distribution, and the Poisson bootstrap is
in turn slightly different from the multinomial bootstrap that
``EmpiricalDistribution.sample`` implements. Both approximations
have error :math:`O(1/n)` and are negligible by :math:`n` in the
low thousands; for the small samples typical of textbook bootstrap
examples the approximation is unnecessary because the naive
bootstrap is already fast.

The technique applies to *plain* sample quantiles — the q-th order
statistic, or one of the closely related index-based definitions
that NumPy's ``np.quantile`` selects between via the ``method``
argument when the index falls between two order statistics. For
quantile estimators that interpolate (the default ``linear``
method), the order-statistic identity is a slight idealization; on
samples where the interpolation moves the estimate by more than a
single observation the discrepancy matters, but for the large
:math:`n` this technique is built for, the gap between adjacent
order statistics is small enough that the difference is below Monte
Carlo noise.

The closed-form acceleration developed in the BCa section assumes
the same order-statistic convention used in the bootstrap step. If
the statistic is computed under an interpolated convention (default
``np.quantile``) but the closed-form ``jv`` is built under the
order-statistic convention, the two are still close at large
:math:`n` — they differ only when the leave-one-out interpolation
moves the estimate strictly between :math:`X_{(k')}` and
:math:`X_{(k'+1)}` — but they are no longer algebraically
identical. For strict consistency, evaluate the bootstrap statistic
with ``method='inverted_cdf'`` (or one of the closely related
non-interpolating methods) so that the bootstrap step, the observed
:math:`\hat{q}`, and the jackknife all reference the same estimator.

Finally, the technique gives bootstrap intervals on the *quantile
estimator*. It does not fix any of the well-known issues with
bootstrapping quantiles on small samples (the bootstrap distribution
of a sample quantile is discrete, supported on at most :math:`n`
distinct values; coverage of nominal-level intervals can be poor).
On the data sizes that motivate this technique — tens of thousands
of observations and up — those concerns are not the binding
constraint, but they remain worth checking for any particular
application.
