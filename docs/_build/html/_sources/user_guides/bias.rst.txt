Bias
====

The bias of an estimator is the systematic gap between its expected
value and the parameter it is trying to estimate. If the bias is
small relative to the standard error, it can usually be ignored. If
it is large, it distorts every downstream conclusion: confidence
intervals are off-center, *p*-values are miscalibrated, and reported
point estimates are nudged in a fixed direction. Knowing how big the
bias is — even approximately — is essential before reporting it.

For a few canonical estimators, bias has a closed-form expression.
The plug-in sample variance has bias :math:`-\sigma^2 / n`. The
sample correlation has a bias of order :math:`1/n` whose leading term
is known. Beyond a small set of textbook cases, no formula is
available. The bootstrap fills the gap by estimating bias directly
from the data — but at first glance this looks impossible. Bias is
a statement about how the estimator behaves under the *unknown*
population distribution :math:`F`. The data we have is one sample
from :math:`F`. How can we possibly say what the average behavior of
:math:`\hat{\theta}` is across the samples we *didn't* draw?

The bootstrap principle
-----------------------

Bias is defined as

.. math::

   \mathrm{bias}_F(\hat{\theta}) = \mathrm{E}_F[\hat{\theta}] - \theta(F),

where :math:`\hat{\theta} = s(X_1, \ldots, X_n)` is the estimator,
:math:`\theta(F)` is the parameter of interest as a functional of the
true distribution :math:`F`, and the expectation is over hypothetical
repeated samples of size :math:`n` drawn from :math:`F`. We do not
know :math:`F`, so neither term is directly computable.

The bootstrap principle is the substitution :math:`F \to \hat{F}`,
where :math:`\hat{F}` is the empirical distribution of the observed
sample. Both terms now become tractable:

.. math::

   \mathrm{bias}_{\hat{F}}(\hat{\theta}) =
       \mathrm{E}_{\hat{F}}[\hat{\theta}^*] - \theta(\hat{F}).

The expectation :math:`\mathrm{E}_{\hat{F}}[\hat{\theta}^*]` is the
average of the estimator over samples of size :math:`n` drawn *with
replacement* from the observed data — a quantity we can approximate
to arbitrary precision by Monte Carlo: draw :math:`B` bootstrap
samples, compute :math:`\hat{\theta}^*_b` on each, and average. The
plug-in value :math:`\theta(\hat{F})` is whatever the parameter
functional :math:`\theta(\cdot)` evaluates to when fed the empirical
distribution. For most estimators the two coincide:
:math:`\theta(\hat{F}) = \hat{\theta}`. (For some, like a trimmed
mean used to estimate the population mean, they don't, which is why
``bias`` accepts both ``stat`` and ``t`` arguments — see [ET93]_
S10.2.)

What does this estimate? It is the bias of :math:`\hat{\theta}` *if
the empirical distribution were the truth*. That is well-defined and
fully computable, but it is not the quantity we ultimately want,
which is :math:`\mathrm{bias}_F(\hat{\theta})`. The two will
generally differ — which raises the question of why the substitution
is useful at all.

The answer rests on two observations. First, :math:`\hat{F}`
converges to :math:`F` as :math:`n \to \infty`, so for parameters
that depend continuously on the distribution, bias under
:math:`\hat{F}` converges to bias under :math:`F`. The bootstrap bias
estimate is *consistent*. Second, and more importantly for finite
:math:`n`: the bias of a smooth estimator typically depends on
:math:`F` through a few moments — variance, third cumulant — that
:math:`\hat{F}` already captures well even for modest sample sizes.
Concretely, if the leading bias term is :math:`c(F)/n` for some
smooth functional :math:`c`, then the bootstrap reports
:math:`c(\hat{F})/n`, which differs from the truth only at order
:math:`1/n^{3/2}`. The bootstrap bias estimate thus inherits the
estimator's own smoothness in :math:`F`.

A sanity check: the sample variance
-----------------------------------

The plug-in sample variance,

.. math::

   \hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2,

is the textbook example of a biased estimator. Its expectation is
:math:`(n - 1)\sigma^2 / n`, so
:math:`\mathrm{bias}_F(\hat{\sigma}^2) = -\sigma^2 / n`. On the
mouse-control survival data (:math:`n = 9`,
:math:`\hat{\sigma}^2 = 1603.7`), the analytical bias estimate is

.. math::

   -\hat{\sigma}^2 / n = -178.2.

The bootstrap should recover this without invoking the formula:

.. code-block:: python

    import numpy as np
    import bootstrap_stat as bp
    from bootstrap_stat.datasets import mouse_data

    data = np.array(mouse_data("control"))

    def variance_p(x, p):
        m = np.dot(p, x)
        return np.dot(p, (x - m) ** 2)

    bias_hat = bp.better_bootstrap_bias(data, variance_p, B=10000)
    # -177.0   (analytical: -178.2)

The agreement to three significant figures is not a numerical
accident: for the plug-in variance, one can show by direct algebra
that :math:`\mathrm{E}_{\hat{F}}[\hat{\sigma}^{2*}] = (n-1)
\hat{\sigma}^2 / n`, so the bootstrap bias estimate is exactly
:math:`-\hat{\sigma}^2 / n` in expectation, with Monte Carlo error
shrinking like :math:`1/\sqrt{B}`. The bootstrap recovers the
classical result by the same mechanism that produces it analytically
— except that it does so for any estimator, smooth or not, with no
algebra required.

The example
-----------

For a problem with no closed-form bias, we use the patch data from
Table 10.1 of [ET93]_. Eight subjects each wore three medical patches
designed to deliver a hormone: a placebo, an "old" patch from an
established manufacturing plant, and a "new" patch from a newly
opened plant. The FDA criterion for declaring the two plants
bioequivalent is that the average effect of switching from the old
plant to the new is at most 20% of the average effect of switching
from placebo to the old plant. With

.. math::

   y_i = \text{newpatch}_i - \text{oldpatch}_i,
   \qquad
   z_i = \text{oldpatch}_i - \text{placebo}_i,

bioequivalence requires :math:`|\theta| \le 0.20`, where
:math:`\theta = \mathrm{E}[Y] / \mathrm{E}[Z]`. The natural plug-in
estimator is

.. math::

   \hat{\theta} = \bar{y} / \bar{z}.

This is a ratio of means: even though :math:`\bar{y}` and
:math:`\bar{z}` are individually unbiased estimators of
:math:`\mathrm{E}[Y]` and :math:`\mathrm{E}[Z]`, the ratio is biased
at order :math:`1/n` because the denominator's variability
contributes asymmetrically. With :math:`n = 8` we are well into the
regime where this matters.

Setting up
----------

.. code-block:: python

    import numpy as np
    import bootstrap_stat as bp
    from bootstrap_stat.datasets import patch_data

    data = patch_data()                  # n = 8

    def ratio(d):
        return d["y"].mean() / d["z"].mean()

    def ratio_p(d, p):
        return np.dot(p, d["y"].values) / np.dot(p, d["z"].values)

    dist = bp.EmpiricalDistribution(data)
    theta_hat = ratio(data)              # -0.0713

The point estimate is :math:`\hat{\theta} = -0.071`, well within the
bioequivalence threshold of 0.20. The remaining question is whether
that point estimate is biased enough to matter.

Bootstrap bias
--------------

:func:`~bootstrap_stat.bias.bias` implements the direct estimate

.. math::

   \widehat{\mathrm{bias}} = \frac{1}{B} \sum_{b=1}^{B} \hat{\theta}^*_b
                           - \theta(\hat{F}).

It takes both ``stat`` (the estimator, applied to a sample) and ``t``
(the parameter functional, applied to a distribution). For a plug-in
statistic the two are the same function:

.. code-block:: python

    bias_hat = bp.bias(dist, ratio, ratio, B=2000)
    # 0.00946   ~0.5 s

The estimated bias is :math:`+0.009`, about 13% of the magnitude of
:math:`\hat{\theta} = -0.071`. The sign is informative: the ratio
estimate is biased toward zero, so the true bioequivalence ratio is
likely *more* negative than :math:`\hat{\theta}` suggests.

Bias is a noisy quantity in the bootstrap. The mean of
:math:`\hat{\theta}^*` has Monte Carlo standard error roughly
:math:`\widehat{\mathrm{se}}(\hat{\theta}) / \sqrt{B}`, and we are
subtracting a non-noisy quantity from it. To resolve a bias of
:math:`+0.009` against an SE of :math:`0.106`, we need
:math:`B = 2000` to reduce the Monte Carlo SE to
:math:`\sim 0.002`. This is far more bootstrap samples than are
needed for a standard error.

Better bootstrap bias
---------------------

:func:`~bootstrap_stat.bias.better_bootstrap_bias` is a
variance-reduction technique for plug-in statistics that converges
much faster than the direct estimator. It exploits the identity

.. math::

   \mathrm{E}_{\hat{F}}[\hat{\theta}^*] - \theta(\hat{F}) =
   \mathrm{E}_{\hat{F}}\left[ s(\mathbf{x}, \mathbf{p}^*) -
                              s(\mathbf{x}, \bar{\mathbf{p}}^*) \right],

where :math:`\mathbf{p}^*_b` is the resampling-frequency vector for
the :math:`b`\ th bootstrap sample and :math:`\bar{\mathbf{p}}^* =
B^{-1} \sum_b \mathbf{p}^*_b`. Both terms are evaluated at the *same*
realized resampling vectors, removing the dominant source of Monte
Carlo variance. The catch is the resampling-form requirement: the
statistic must take ``(x, p)`` rather than just ``x``.

.. code-block:: python

    bias_hat = bp.better_bootstrap_bias(data, ratio_p, B=400)
    # 0.00953   ~0.03 s

With :math:`B = 400` the better estimator is already as accurate as
the direct estimator at :math:`B = 2000`, an order-of-magnitude
speedup at equal precision.

Jackknife bias
--------------

:func:`~bootstrap_stat.bias.jackknife_bias` estimates the bias from
the :math:`n` leave-one-out values:

.. math::

   \widehat{\mathrm{bias}}_{\mathrm{jack}}
       = (n - 1) \big( \overline{\hat{\theta}_{(\cdot)}} - \hat{\theta} \big),

where :math:`\hat{\theta}_{(i)}` is the statistic computed with
observation :math:`i` removed and
:math:`\overline{\hat{\theta}_{(\cdot)}}` is their mean.

.. code-block:: python

    bias_hat = bp.jackknife_bias(data, ratio)
    # 0.00800   < 0.01 s

The jackknife is fast and deterministic — :math:`n` evaluations
total — and reports bias :math:`+0.008` on the patch data, agreeing
with the bootstrap to within Monte Carlo noise. As with the standard
error, the catch is smoothness: the jackknife is consistent only for
plug-in statistics whose parameter functional is smooth at
:math:`\hat{F}`. It is *not* applicable to the median or other
quantile-based statistics. See [ET93]_ (S10.5).

Bias correction
---------------

Once the bias is estimated, the natural move is to subtract it:

.. math::

   \tilde{\theta} = \hat{\theta} - \widehat{\mathrm{bias}}.

:func:`~bootstrap_stat.bias.bias_corrected` computes this directly,
defaulting to ``better_bootstrap_bias`` for the underlying estimate:

.. code-block:: python

    theta_tilde = bp.bias_corrected(data, ratio_p, B=400)
    # -0.0808   ~0.03 s

The corrected estimate :math:`\tilde{\theta} = -0.081` is slightly
more negative than the naive :math:`\hat{\theta} = -0.071` —
consistent with the bias being toward zero — and remains comfortably
within the 0.20 bioequivalence threshold.

**Bias correction has a cost.** The corrected estimator
:math:`\tilde{\theta}` is the difference of two random quantities,
:math:`\hat{\theta}` and :math:`\widehat{\mathrm{bias}}`, and its
variance is therefore the sum of their variances minus twice their
covariance. In practice this means
:math:`\widehat{\mathrm{se}}(\tilde{\theta})` is typically larger
than :math:`\widehat{\mathrm{se}}(\hat{\theta})` — sometimes
substantially. For statistics where the bias is small relative to
the SE, the increase in variance can outweigh the reduction in bias,
producing a corrected estimator with worse mean-squared error than
the original. The standard rule of thumb from [ET93]_ (S10.6) is:
correct only if :math:`|\widehat{\mathrm{bias}}| /
\widehat{\mathrm{se}} > 0.25`. For the patch ratio, that ratio is
:math:`0.009 / 0.106 \approx 0.09`, well below the threshold; the
right choice is to report :math:`\hat{\theta}` and leave it
uncorrected, while noting that the bias is small and toward zero.

Comparison
----------

All bias estimates are for the ratio :math:`\hat{\theta} =
\bar{y} / \bar{z}` on the patch data (:math:`n = 8`,
:math:`\hat{\theta} = -0.0713`):

.. list-table::
   :header-rows: 1
   :widths: 32 12 12 16

   * - Method
     - Bias
     - Corrected :math:`\tilde{\theta}`
     - Wall time
   * - Bootstrap (:math:`B = 2000`)
     - +0.0095
     - −0.081
     - 0.5 s
   * - Better bootstrap (:math:`B = 400`)
     - +0.0095
     - −0.081
     - 0.03 s
   * - Better bootstrap (:math:`B = 2000`)
     - +0.0083
     - −0.080
     - 0.13 s
   * - Jackknife
     - +0.0080
     - −0.079
     - < 0.01 s

For most applications,
:func:`~bootstrap_stat.bias.better_bootstrap_bias` is the right
default for plug-in statistics: it converges roughly an order of
magnitude faster than the direct estimator and is similarly easy to
use. :func:`~bootstrap_stat.bias.jackknife_bias` is competitive at
:math:`O(n)` evaluations for smooth statistics on small samples, but
should not be applied to non-smooth functionals.
:func:`~bootstrap_stat.bias.bias` is the general-purpose fallback
when the statistic is not a plug-in (e.g., a robust estimator used
for a non-robust parameter), at the cost of needing :math:`B \approx
2000` for stable answers. Whichever method is used, the decision to
*correct* should follow the bias-to-SE rule: correction is worth its
variance penalty only when the bias is a substantial fraction of the
standard error. See [ET93]_ (S10) for a full treatment.
