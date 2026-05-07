Significance Tests
==================

Confidence intervals and significance tests are two views of the same
inferential machinery. A 100(1 − α)% confidence interval excludes
the null value θ\ :sub:`0` if and only if the corresponding test of
H\ :sub:`0`: θ = θ\ :sub:`0` rejects at level α. Conversely, the
*p*-value attached to a particular θ\ :sub:`0` is the smallest α at
which a one-sided CI just barely covers it. This duality is not an
accident of normal-theory: it falls out of the definitions, and any
inference procedure worth the name ought to honor it. The bootstrap
does. The same bootstrap distribution that produces an interval also
produces a *p*-value, and the two answers are forced by the
construction to be consistent with each other.

We illustrate this on the same setup as the
:doc:`confidence_intervals` guide: the Pearson correlation between
LSAT scores and undergraduate GPA at :math:`n = 15` American law
schools, where :math:`\hat{\theta} = 0.776`. The CI guide concluded
with two 90% intervals,

- percentile: :math:`(0.521,\ 0.949)`,
- BCa: :math:`(0.438,\ 0.927)`.

These imply *p*-values for any null value :math:`\theta_0` we care
to test, and we will see the implications spelled out concretely
below — including a case where the percentile test rejects at
:math:`\alpha = 0.05` and the BCa test does not.

Setting up
----------

The setup is identical to the CI guide. We wrap the data in an
:class:`~bootstrap_stat.distributions.EmpiricalDistribution` and
define the correlation as a plain function.

.. code-block:: python

    import numpy as np
    import bootstrap_stat as bp
    from bootstrap_stat.datasets import law_data

    data = law_data()

    def correlation(d):
        return np.corrcoef(d["LSAT"], d["GPA"])[0, 1]

    dist = bp.EmpiricalDistribution(data)
    theta_hat = correlation(data)   # 0.7764

The natural null hypothesis for a correlation is
H\ :sub:`0`: :math:`\rho = 0`, but a more interesting test on these
data is whether the population correlation is at least moderately
strong, H\ :sub:`0`: :math:`\rho = 0.5`. Both will appear below.

.. note::

   The library reports one-sided *p*-values by default; pass
   ``two_sided=True`` for the two-sided variant (still experimental
   for ``bcanon_asl``).

Classical test (Fisher's z)
---------------------------

As a baseline, Fisher's :math:`z`-transform delivers a closed-form
test under the assumption of bivariate normality. With :math:`z =
\mathrm{arctanh}(r)` having approximate variance :math:`1 / (n - 3)`,
the standardized statistic
:math:`(z - \mathrm{arctanh}(\theta_0)) \sqrt{n - 3}` is approximately
standard normal under H\ :sub:`0`.

.. code-block:: python

    from scipy import stats

    z = np.arctanh(theta_hat)
    z0 = np.arctanh(0.0)
    p_classical = 1 - stats.norm.cdf((z - z0) * np.sqrt(len(data) - 3))
    # 0.00017   < 0.001 s    (one-sided, H0: rho = 0)

The classical *p*-value is :math:`1.7 \times 10^{-4}`, comfortably
below any conventional threshold. As with the Fisher interval, the
limitations are the bivariate normality assumption and the
:math:`n - 3` approximation, both strained at :math:`n = 15`. The
bootstrap methods below make no distributional assumptions.

Percentile ASL
--------------

:func:`~bootstrap_stat.significance.percentile_asl` is the test
counterpart of
:func:`~bootstrap_stat.confidence.percentile_interval`. It draws
bootstrap replicates :math:`\hat{\theta}^*_b`, evaluates the
statistic on each, and counts the fraction that fall on the *opposite
side* of :math:`\theta_0` from :math:`\hat{\theta}`:

.. math::

   \widehat{\mathrm{ASL}} =
   \frac{1}{B} \#\{ b : \hat{\theta}^*_b < \theta_0 \},
   \qquad \text{(when } \hat{\theta} > \theta_0\text{)}.

.. code-block:: python

    p0  = bp.percentile_asl(dist, correlation, data, theta_0=0.0, B=2000)
    p05 = bp.percentile_asl(dist, correlation, data, theta_0=0.5, B=2000)
    # 0.0000   ~0.7 s   (H0: rho = 0)
    # 0.0395   ~0.7 s   (H0: rho = 0.5)

The duality with the percentile interval is direct. The 90%
percentile CI ran from 0.521 to 0.949; the lower endpoint
corresponds to the one-sided :math:`p = 0.05` cutoff. Any
:math:`\theta_0 < 0.521` is rejected at :math:`\alpha = 0.05`. The
test of H\ :sub:`0`: :math:`\rho = 0.5` reports :math:`p = 0.040`,
just under the threshold — which matches the CI lower bound just
above 0.5.

The reported *p*-value of :math:`0.0000` for H\ :sub:`0`:
:math:`\rho = 0` is not literally zero; it is the resolution limit
of :math:`B = 2000` replicates. With no bootstrap sample falling
below 0, the best the empirical estimate can say is
:math:`p < 1/B = 0.0005`. Resolving smaller *p*-values requires
larger :math:`B`, which is feasible but rarely necessary — at any
useful significance threshold, a *p*-value below the bootstrap
resolution is an unambiguous rejection.

The percentile ASL inherits the percentile interval's
:math:`O(n^{-1/2})` accuracy. For statistics whose bootstrap
distribution is skewed or biased — including the correlation when
the true value is near :math:`\pm 1` — the *p*-value can be off by
enough to flip a borderline conclusion. The next method addresses
this.

BCa ASL
-------

:func:`~bootstrap_stat.significance.bcanon_asl` applies the same
bias-correction :math:`\hat{z}_0` and acceleration :math:`\hat{a}`
to the percentile *p*-value that
:func:`~bootstrap_stat.confidence.bcanon_interval` applies to the
percentile endpoints. The result is :math:`O(n^{-1})` accurate, an
order of magnitude better than the percentile method, with no
additional sampling cost beyond the jackknife required for the
acceleration.

.. code-block:: python

    p0  = bp.bcanon_asl(dist, correlation, data, theta_0=0.0, B=2000)
    p05 = bp.bcanon_asl(dist, correlation, data, theta_0=0.5, B=2000)
    # 0.0000   ~0.6 s   (H0: rho = 0)
    # 0.0869   ~0.6 s   (H0: rho = 0.5)

For H\ :sub:`0`: :math:`\rho = 0` the answer agrees with the
percentile test to within bootstrap resolution: an unambiguous
rejection. For H\ :sub:`0`: :math:`\rho = 0.5` the two methods
disagree:

- Percentile: :math:`p = 0.040` — reject at :math:`\alpha = 0.05`.
- BCa: :math:`p = 0.087` — do not reject at :math:`\alpha = 0.05`.

This is the same disagreement that appears in the confidence
intervals. The 90% percentile interval, :math:`(0.521,\ 0.949)`,
excludes 0.5 in the lower tail; the 90% BCa interval,
:math:`(0.438,\ 0.927)`, includes it. The duality forces the test
results: *whichever bound the CI uses, the test must agree*. The
percentile method overstates how far below :math:`\hat{\theta}` the
plausible values extend, because it is insensitive to the left
skewness of the correlation's sampling distribution near 1 — and
the *p*-value picks up the same error. The BCa adjustment, derived
to capture exactly this skewness, gives the more accurate answer in
both cases. For applied work the BCa test is preferred for the same
reasons the BCa interval is preferred.

General bootstrap ASL
---------------------

The percentile and BCa tests both invert the bootstrap distribution
of :math:`\hat{\theta}^*` against a single null value
:math:`\theta_0`. Some hypotheses don't fit that mold. Suppose the
question is not "is :math:`\rho = 0.5`?" but rather "are LSAT and
GPA *independent*?" — a structural hypothesis about the joint
distribution rather than a numerical one about a single parameter.
Under independence the population correlation is zero, but the
converse is false: zero correlation does not imply independence in
general. The right test samples directly from a null distribution in
which independence holds.

:func:`~bootstrap_stat.significance.bootstrap_asl` accepts an
explicit null distribution :math:`\hat{F}_0` and reports the
fraction of bootstrap statistics from :math:`\hat{F}_0` that exceed
:math:`\hat{\theta}`. Constructing :math:`\hat{F}_0` is the user's
responsibility; for an independence null the natural choice is
independent resampling of the marginals:

.. code-block:: python

    import pandas as pd

    class IndependentDist(bp.EmpiricalDistribution):
        """Empirical distribution with LSAT and GPA resampled independently."""
        def sample(self, size=None, **kwargs):
            m = self.n if size is None else size
            lsat = np.random.choice(self.data["LSAT"].values, size=m, replace=True)
            gpa  = np.random.choice(self.data["GPA"].values,  size=m, replace=True)
            return pd.DataFrame({"LSAT": lsat, "GPA": gpa})

    null_dist = IndependentDist(data)
    p = bp.bootstrap_asl(null_dist, correlation, data,
                         theta_hat=theta_hat, B=10000)
    # 0.0003   ~4 s   (H0: LSAT and GPA independent)

Because the null distribution is constructed to satisfy
H\ :sub:`0` exactly, the *p*-value is just a frequency: out of
:math:`B` correlations between independent LSAT and GPA pairs, what
fraction are at least as large as :math:`\hat{\theta} = 0.776`? With
:math:`B = 10000` the answer is three. The result coincides with
the classical Fisher-:math:`z` *p*-value (:math:`1.7 \times 10^{-4}`)
to within Monte Carlo noise, despite making no normality assumption.

The general method is the right tool for two-sample tests
(H\ :sub:`0`: :math:`\mu_1 = \mu_2`, sampled by pooling), goodness
of fit, and any null hypothesis that constrains the data-generating
process rather than a single parameter. It is also the one to reach
for when the percentile/BCa duality is awkward — for example when
the natural test statistic is not the parameter being constrained
(such as a likelihood ratio against a null distribution). The cost
is the work of specifying :math:`\hat{F}_0`. See [ET93]_ (S16) for
a thorough treatment.

Power analysis
--------------

A *p*-value answers the question "would I be surprised under
H\ :sub:`0`?". Power answers the dual question "would I detect a
real effect if there is one?". For a fixed test at level
:math:`\alpha` and a specified alternative distribution, the power
is the probability of rejection. Closed-form power calculations
exist for a handful of standard tests. The bootstrap version simulates:
draw :math:`P` samples from the alternative distribution, compute a
*p*-value on each, and report the fraction below :math:`\alpha`.

:func:`~bootstrap_stat.significance.bootstrap_power` automates this.
It accepts an *instance* of an
:class:`~bootstrap_stat.distributions.EmpiricalDistribution` for the
alternative (used to draw samples) and the *class* for the null
(used to construct a null distribution from each simulated sample).

.. code-block:: python

    alt_dist = bp.EmpiricalDistribution(data)   # observed rho ~ 0.776
    pwr = bp.bootstrap_power(alt_dist, IndependentDist, correlation,
                             alpha=0.05, P=200, B=400)
    # 0.98   ~30 s

Against the alternative implied by the observed data — a population
that resamples LSAT and GPA jointly with their observed
correlation — the test of independence rejects at :math:`\alpha =
0.05` in 98% of simulated replicates. Even with :math:`n = 15`, a
correlation of 0.78 is a strong enough signal that the test almost
always finds it.

Power calculations are nested bootstraps and accordingly expensive:
:math:`P` outer simulations, each running an inner bootstrap of size
:math:`B`. Defaults of :math:`P = 100` and :math:`B = 400` are
adequate for power estimates good to about 5 percentage points,
which is usually all that is needed for sample-size planning.

Comparison
----------

All bootstrap ASLs below are computed with :math:`B = 2000` (or
:math:`B = 10000` where noted) on the law school data, against the
specified null:

.. list-table::
   :header-rows: 1
   :widths: 28 24 14 14 12

   * - Method
     - Null
     - *p*-value
     - Reject at α=0.05?
     - Wall time
   * - Classical (Fisher z)
     - ρ = 0
     - 0.00017
     - yes
     - < 0.001 s
   * - Percentile ASL
     - ρ = 0
     - < 0.0005
     - yes
     - 0.7 s
   * - Percentile ASL
     - ρ = 0.5
     - 0.040
     - yes
     - 0.7 s
   * - BCa ASL
     - ρ = 0
     - < 0.0005
     - yes
     - 0.6 s
   * - BCa ASL
     - ρ = 0.5
     - 0.087
     - **no**
     - 0.6 s
   * - Bootstrap ASL (general)
     - independence
     - 0.0003 (B=10000)
     - yes
     - 4 s

For most applications where the test concerns a single parameter,
:func:`~bootstrap_stat.significance.bcanon_asl` is the right
default — for the same reasons
:func:`~bootstrap_stat.confidence.bcanon_interval` is the recommended
default for confidence intervals. The duality with the BCa interval
is exact: each tells you the same thing in a different language, and
the BCa adjustment captures skewness in the bootstrap distribution
that the percentile method misses.
:func:`~bootstrap_stat.significance.percentile_asl` is the simpler
quantile-based companion, useful for exposition but vulnerable to
the same skewness errors that affect percentile intervals.
:func:`~bootstrap_stat.significance.bootstrap_asl` is the
general-purpose tool when the null hypothesis cannot be expressed
as a single value of the test statistic — most notably for
two-sample tests and tests of independence — at the cost of
constructing the null distribution explicitly. See [ET93]_ (S15–16)
for the broader theory.
