# pyre-unsafe
"""Standard error estimation methods."""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from bootstrap_stat._utils import (
    ArrayLike,
    JackknifeValues,
    Statistic,
    _influence_components,
)
from bootstrap_stat.distributions import EmpiricalDistribution
from bootstrap_stat.sampling import bootstrap_samples, jackknife_values


def jackknife_standard_error(
    x: ArrayLike | tuple[ArrayLike, ...],
    stat: Statistic,
    return_samples: bool = False,
    jv: JackknifeValues | None = None,
    num_threads: int = 1,
) -> float | tuple[float, JackknifeValues]:
    r"""Jackknife estimate of standard error.

    Parameters
    ----------
     x : array_like or pandas DataFrame or tuple
        The data. If tuple. interpretation is as a MultiSample
        distribution, like in A/B testing.
     stat : function
        The statistic.
     return_samples : boolean, optional
        If True, return the jackknife values. Defaults to False.
     jv : array_like, optional
        Jackknife values. Can be passed if they have already been
        calculated, which will speed this up considerably.
     num_threads : int, optional
        Number of threads to use for multicore processing. Defaults to
        1, meaning all calculations will be done in a single
        thread. Set to -1 to use all available cores.

    Returns
    -------
     se : float
        The standard error.
     jv : ndarray
        Jackknife values. Only returned if `return_samples` is True.

    Notes
    -----
    The jackknife estimate of standard error is only applicable when
    `stat` is a plug-in statistic, that is, having the form
    :math:`t(\hat{F})`, where :math:`\hat{F}` is the empirical
    distribution. Moreover, it is only applicable when t is a smooth
    function. Notable exceptions include the median. The jackknife
    cannot be used to estimate the standard error of non-smooth
    estimators. See [EF93, S10.6]

    """
    if isinstance(x, tuple):
        # Multisample jackknife
        if jv is None:
            jv = [
                jackknife_values(x, stat, num_threads=num_threads, sample=i)
                for i in range(len(x))
            ]

        var = 0
        for jv_s in jv:
            var += (len(jv_s) - 1) * np.var(jv_s, ddof=0)
        se = np.sqrt(var)
    else:
        if jv is None:
            jv = jackknife_values(x, stat, num_threads=num_threads)
        n = len(jv)
        se = np.sqrt((n - 1) * np.var(jv, ddof=0))

    if return_samples:
        return se, jv
    else:
        return se


def standard_error(
    dist: EmpiricalDistribution,
    stat: Statistic,
    robustness: float | None = None,
    B: int = 200,
    size: int | tuple[int, ...] | None = None,
    jackknife_after_bootstrap: bool = False,
    return_samples: bool = False,
    theta_star: npt.NDArray[np.float64] | None = None,
    num_threads: int = 1,
) -> (
    float
    | tuple[float, float]
    | tuple[float, npt.NDArray[np.float64]]
    | tuple[float, float, npt.NDArray[np.float64]]
):
    """Standard error

    Parameters
    ----------
     dist : EmpiricalDistribution
        The empirical distribution.
     stat : function
        The statistic for which we wish to calculate the standard
        error.
     robustness : float or None, optional
        Controls whether to use a robust estimate of standard
        error. If specified, should be a float in (0.5, 1.0), with
        lower values corresponding to greater bias but increased
        robustness. If None (default), uses the non-robust estimate of
        standard error.
     B : int, optional
        Number of bootstrap samples. Defaults to 200.
     size : int or tuple of ints, optional
        Size to pass for generating samples from the distribution.
        Defaults to None, indicating the samples will be the same size
        as the original dataset.
     jackknife_after_bootstrap : boolean, optional
        If True, will estimate the variability of our estimate of
        standard error. See [ET93, S19.4] for details.
     return_samples : boolean, optional
        If True, return the bootstrapped statistic values. Defaults to False.
     theta_star : array_like, optional
        Bootstrapped statistic values. Can be passed if they have
        already been calculated, which will speed this up
        considerably.
     num_threads : int, optional
        Number of threads to use for multicore processing. Defaults to
        1, meaning all calculations will be done in a single
        thread. Set to -1 to use all available cores.

    Returns
    --------
     se : float
        The standard error.
     se_jack : float
        Jackknife-after-bootstrap estimate of the standard error of se
        (i.e. an estimate of the variability of se itself). Only
        returned if `jackknife_after_bootstrap` is True.
     theta_star : ndarray
        Array of bootstrapped statistic values. Only returned if
        `return_samples` is True.

    """
    import scipy.stats as ss

    if theta_star is None or jackknife_after_bootstrap:
        if jackknife_after_bootstrap:
            theta_star, Cj = bootstrap_samples(
                dist,
                stat,
                B,
                size=size,
                jackknife=True,
                num_threads=num_threads,
            )
        else:
            theta_star = bootstrap_samples(
                dist, stat, B, size=size, num_threads=num_threads
            )

    if robustness is None:
        se = np.std(theta_star, ddof=1)
        if jackknife_after_bootstrap:
            qB = [np.std(C, ddof=0) for C in Cj]
            n = len(qB)
            se_jack = np.sqrt((n - 1) * np.var(qB, ddof=0))
    else:
        if robustness <= 0.5 or robustness >= 1:
            raise ValueError(f"Invalid robustness: {robustness}")

        z_alpha = ss.norm.ppf(robustness)
        p = np.percentile(theta_star, [100 * robustness, 100 * (1 - robustness)])
        se = p[0] - p[1]
        se /= 2 * z_alpha

        if jackknife_after_bootstrap:
            raise NotImplementedError("Not implemented")

    if return_samples and jackknife_after_bootstrap:
        return se, se_jack, theta_star
    elif return_samples:
        return se, theta_star
    elif jackknife_after_bootstrap:
        return se, se_jack
    else:
        return se


def infinitesimal_jackknife(
    x: ArrayLike | tuple[ArrayLike, ...],
    stat: Callable[[Any, npt.NDArray[np.float64]], float],
    eps: float = 1e-3,
    influence_components: npt.NDArray[np.float64] | None = None,
    return_influence_components: bool = False,
    num_threads: int = 1,
) -> float | tuple[float, npt.NDArray[np.float64]]:
    """Infinitesimal Jackknife

    Parameters
    ----------
     x : array_like or pandas DataFrame or tuple
        The data. If tuple. interpretation is as a MultiSample
        distribution, like in A/B testing.
     stat : function
        The statistic. Should take `x` and a resampling vector as
        input. See Notes.
     eps : float, optional
        Epsilon for limit calculation. Defaults to 1e-3.
     influence_components : array_like, optional
        Influence components. See Notes.
     return_influence_components : boolean, optional
        Specifies whether to return the influence components. Defaults
        to False.
     num_threads : int, optional
        Number of threads to use for multicore processing. Defaults to
        1, meaning all calculations will be done in a single
        thread. Set to -1 to use all available cores.

    Returns
    -------
     se : float
        The standard error.
     influence_components : array_like
        The influence components. Only returned if
        `return_influence_components` is True.

    Notes
    -----
    The infinitesimal jackknife requires the statistic to be expressed
    in "resampling form". See [ET93, S21] for details. The ith
    influence component is a type of derivative of the statistic with
    respect to the ith observation. This is computed by a finite
    difference method: we simply evaluate the statistic putting a
    little extra weight (`eps`) on the ith observation, minus the
    statistic evaluated on the original dataset, divided by `eps`.

    In some cases, there is an analytical formula for the influence
    components. In these cases, it would be better for the caller to
    compute the influence components elsewhere and simply pass them to
    this function.

    """
    n = len(x)
    eps /= n
    if influence_components is None:
        influence_components = _influence_components(
            x, stat, eps=eps, num_threads=num_threads
        )

    se = np.sqrt(influence_components.dot(influence_components) / (n * n))
    if return_influence_components:
        return se, influence_components
    else:
        return se
