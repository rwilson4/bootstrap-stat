# pyre-unsafe
"""Achieved significance level and power analysis methods."""

from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.stats as ss

from bootstrap_stat._utils import (
    ArrayLike,
    JackknifeValues,
    Statistic,
    _bca_acceleration,
)
from bootstrap_stat.distributions import EmpiricalDistribution
from bootstrap_stat.sampling import bootstrap_samples, jackknife_values


def bootstrap_asl(
    dist: EmpiricalDistribution,
    stat: Statistic,
    x: ArrayLike | tuple[ArrayLike, ...] | None,
    B: int = 1000,
    size: int | tuple[int, ...] | None = None,
    return_samples: bool = False,
    theta_star: npt.NDArray[np.float64] | None = None,
    theta_hat: float | None = None,
    two_sided: bool = False,
    num_threads: int = 1,
) -> float | tuple[float, npt.NDArray[np.float64]]:
    """Achieved Significance Level, general bootstrap method

    Parameters
    ----------
     dist : EmpiricalDistribution
        The empirical distribution.
     stat : function
        The test statistic.
     x : array_like or pandas DataFrame or tuple
        The data. It isn't used for anything, so the caller can pass
        None if the data are not available, but it is here for
        consistency across asl routines.
     B : int, optional
        Number of bootstrap samples. Defaults to 1000.
     size : int or tuple of ints, optional
        Size to pass for generating samples from the alternative
        distribution. Defaults to None.
     return_samples : boolean, optional
        If True, return the bootstrapped statistic or jackknife
        values. Defaults to False.
     theta_star : array_like, optional
        Bootstrapped statistic values. Can be passed if they have
        already been calculated, which will speed this up
        considerably.
     theta_hat : float, optional
        Observed statistic. Can be passed if it has already been
        calculated, which will speed this up slightly.
     two_sided : boolean, optional
        If True, computes a two-sided significance value. If False
        (default), only a one-sided value is returned. Support for
        two-sided tests is *experimental*. Use with caution!
     num_threads : int, optional
        Number of threads to use for multicore processing. Defaults to
        1, meaning all calculations will be done in a single
        thread. Set to -1 to use all available cores.

    Returns
    -------
     asl : float
       Achieved significance level, the probability of an outcome at
       least as extreme as that actually observed under the null
       hypothesis; aka the p-value.
     theta_star : ndarray
        Array of bootstrapped statistic values. Only returned if
        `return_samples` is True.

    """
    asl = 0

    if theta_hat is None:
        theta_hat = stat(x)

    if theta_star is None:
        theta_star = bootstrap_samples(
            dist, stat, B, size=size, num_threads=num_threads
        )

    if two_sided:
        asl = (abs(theta_star) >= abs(theta_hat)).sum()
    else:
        asl = (theta_star >= theta_hat).sum()

    asl /= len(theta_star)
    if return_samples:
        return asl, theta_star
    else:
        return asl


def percentile_asl(
    dist: EmpiricalDistribution,
    stat: Statistic,
    x: ArrayLike | tuple[ArrayLike, ...],
    theta_0: float = 0,
    B: int = 1000,
    size: int | tuple[int, ...] | None = None,
    return_samples: bool = False,
    theta_star: npt.NDArray[np.float64] | None = None,
    theta_hat: float | None = None,
    two_sided: bool = False,
    num_threads: int = 1,
) -> float | tuple[float, npt.NDArray[np.float64]]:
    """Achieved Significance Level, percentile method

    Parameters
    ----------
     dist : EmpiricalDistribution
        The empirical distribution.
     stat : function
        The test statistic.
     x : array_like or pandas DataFrame or tuple
        The data, used to calculate the observed value of the
        statistic if `theta_hat` is not passed.
     theta_0 : float, optional
        The mean of the test statistic under the null
        hypothesis. Defaults to 0.
     B : int, optional
        Number of bootstrap samples.
     size : int or tuple of ints, optional
        Size to pass for generating samples from the alternative
        distribution. Defaults to None.
     return_samples : boolean, optional
        If True, return the bootstrapped statistic or jackknife
        values. Defaults to False.
     theta_star : array_like, optional
        Bootstrapped statistic values. Can be passed if they have
        already been calculated, which will speed this up
        considerably.
     theta_hat : float, optional
        Observed statistic. Can be passed if it has already been
        calculated, which will speed this up slightly.
     two_sided : boolean, optional
        If True, computes a two-sided significance value. If False
        (default), only a one-sided value is returned. Support for
        two-sided tests is *experimental*. Use with caution!
     num_threads : int, optional
        Number of threads to use for multicore processing. Defaults to
        1, meaning all calculations will be done in a single
        thread. Set to -1 to use all available cores.

    Returns
    -------
     asl : float
       Achieved significance level, the probability of an outcome at
       least as extreme as that actually observed under the null
       hypothesis; aka the p-value.
     theta_star : ndarray
        Array of bootstrapped statistic values. Only returned if
        `return_samples` is True.

    Notes
    -----
    Under the null hypothesis, the value of the statistic is
    theta_0. Suppose theta_hat > theta_0. Let theta_lo, theta_hi be
    the endpoints of a 100(1-alpha)% confidence interval on theta.
    Suppose alpha is such that theta_lo = theta_0. Then alpha is the
    achieved significance level.

    For the percentile interval, this is simply the fraction of
    bootstrap samples that are "on the other side" of theta_0 from
    theta_hat.

    """
    if theta_hat is None:
        theta_hat = stat(x)

    if theta_hat == theta_0:
        return 1.0

    if theta_star is None:
        theta_star = bootstrap_samples(
            dist, stat, B, size=size, num_threads=num_threads
        )

    if theta_hat > theta_0:
        b = (theta_star < theta_0).sum()
    else:
        b = (theta_star > theta_0).sum()

    asl = b / len(theta_star)
    if two_sided:
        asl *= 2

    if return_samples:
        return asl, theta_star
    else:
        return asl


def bcanon_asl(
    dist: EmpiricalDistribution,
    stat: Statistic,
    x: ArrayLike | tuple[ArrayLike, ...],
    theta_0: float = 0,
    B: int = 1000,
    size: int | tuple[int, ...] | None = None,
    return_samples: bool = False,
    theta_star: npt.NDArray[np.float64] | None = None,
    theta_hat: float | None = None,
    jv: JackknifeValues | None = None,
    two_sided: bool = False,
    num_threads: int = 1,
) -> float | tuple[float, npt.NDArray[np.float64], JackknifeValues | None]:
    """Achieved Significance Level, bcanon method

    Parameters
    ----------
     dist : EmpiricalDistribution
        The empirical distribution.
     stat : function
        The test statistic.
     x : array_like or pandas DataFrame or tuple
        The data, used to evaluate the observed statistic and compute
        jackknife values.
     theta_0 : float, optional
        The mean of the test statistic under the null
        hypothesis. Defaults to 0.
     B : int, optional
        Number of bootstrap samples.
     size : int or tuple of ints, optional
        Size to pass for generating samples from the alternative
        distribution. Defaults to None.
     return_samples : boolean, optional
        If True, return the bootstrapped statistic or jackknife
        values. Defaults to False.
     theta_star : array_like, optional
        Bootstrapped statistic values. Can be passed if they have
        already been calculated, which will speed this up
        considerably.
     theta_hat : float, optional
        Observed statistic. Can be passed if it has already been
        calculated, which will speed this up slightly.
     jv : array_like, optional
        Jackknife values. Can be passed if they have already been
        calculated, which will speed this up considerably.
     two_sided : boolean, optional
        If True, computes a two-sided significance value. If False
        (default), only a one-sided value is returned. Support for
        two-sided tests is *experimental*. Use with caution!
     num_threads : int, optional
        Number of threads to use for multicore processing. Defaults to
        1, meaning all calculations will be done in a single
        thread. Set to -1 to use all available cores.

    Returns
    -------
     asl : float
       Achieved significance level, the probability of an outcome at
       least as extreme as that actually observed under the null
       hypothesis; aka the p-value.
     theta_star : ndarray
        Array of bootstrapped statistic values. Only returned if
        `return_samples` is True.
     jv : ndarray
        Jackknife values. Only returned if `return_samples` is True.

    """
    if theta_hat is None:
        theta_hat = stat(x)

    a0, theta_star = percentile_asl(
        dist,
        stat,
        x,
        theta_0=theta_0,
        B=B,
        size=size,
        return_samples=True,
        theta_star=theta_star,
        theta_hat=theta_hat,
        two_sided=False,
        num_threads=num_threads,
    )
    if a0 == 0 or a0 == 1:
        if return_samples:
            return a0, theta_star, None
        else:
            return a0

    if theta_hat > theta_0:
        zb = (theta_star < theta_hat).sum()
    else:
        zb = (theta_star > theta_hat).sum()

    z0_hat = ss.norm.ppf(zb / len(theta_star))

    if jv is None:
        if dist.is_multi_sample:
            jv = [
                jackknife_values(x, stat, num_threads=num_threads, sample=i)
                for i in range(len(x))
            ]
            jv = (*jv,)
        else:
            jv = jackknife_values(x, stat, num_threads=num_threads)

    a_hat = _bca_acceleration(jv)
    w0 = ss.norm.ppf(a0)
    t = (w0 - z0_hat) / (1 + a_hat * (w0 - z0_hat)) - z0_hat
    asl = ss.norm.cdf(t)

    if two_sided:
        # To-do: generalize as follows.
        # What we really want is stat with the samples
        # interchanged. In many cases of practically interest, that is
        # simply -stat, as shown below. But that's not always the case
        # and we can be more general by allowing the user to pass a
        # transform mapping stat to interchange stat. Need to think
        # this through more.
        other_asl = bcanon_asl(
            dist,
            lambda z: -stat(z),
            x,
            theta_hat=-theta_hat,
            theta_0=-theta_0,
            theta_star=-theta_star,
            two_sided=False,
            num_threads=num_threads,
        )
        asl += other_asl

    if return_samples:
        return asl, theta_star, jv
    else:
        return asl


def bootstrap_power(
    alt_dist: EmpiricalDistribution,
    null_dist: type[EmpiricalDistribution],
    stat: Statistic,
    asl: Statistic = bootstrap_asl,
    alpha: float = 0.05,
    size: int | tuple[int, ...] | None = None,
    P: int = 100,
    **kwargs: Any,
) -> float:
    """Bootstrap Power

    Parameters
    ----------
     alt_dist : EmpiricalDistribution
        Distribution under the alternative hypothesis.
     null_dist : class
        Class corresponding to the null distribution. See Notes.
     stat : function
        Function that computes the test statistic.
     asl : function, optional
        Function that computes an achieved significance
        level. Defaults to bootstrap_asl.
     alpha : float, optional
        Desired Type-I error rate. Defaults to 0.05.
     size : int or tuple of ints, optional
        Size to pass for generating samples from the alternative
        distribution. Defaults to None.
     P : int, optional
        Number of Monte Carlo simulations to run for the purposes of
        calculating power. Defaults to 100.
     kwargs : optional
        Other keyword arguments to pass to `asl`, such as the number
        of bootstrap samples to use.

    Returns
    -------
     pwr : float
        The fraction of Monte Carlo simulations in which the null
        hypothesis was rejected.

    Notes
    -----
    Perhaps the most confusing aspect of this function is that there
    are two distribution passed as input, and they are of a different
    form. The `alt_dist` should be passed as an instance of an
    EmpiricalDistribution or a subclass thereof. We use this parameter
    to generate samples from that distribution. We then need to
    generate an EmpiricalDistribution from that sample, for which we
    need the *class* corresponding to the null distribution, not an
    instance thereof. I recognize this is confusing!

    """
    rejections = 0
    for i in range(P):
        sample = alt_dist.sample(size=size)
        sim_dist = null_dist(sample)
        a = asl(sim_dist, stat, sample, **kwargs)
        if a <= alpha:
            rejections += 1
    return rejections / P
