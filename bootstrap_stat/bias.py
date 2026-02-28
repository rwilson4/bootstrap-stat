# pyre-unsafe
"""Bias estimation and correction methods."""

import multiprocessing as mp
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pathos.multiprocessing import ProcessPool as Pool

from bootstrap_stat._utils import ArrayLike, Statistic, _resampling_vector
from bootstrap_stat.distributions import EmpiricalDistribution
from bootstrap_stat.sampling import bootstrap_samples, jackknife_values


def bias(
    dist: EmpiricalDistribution,
    stat: Statistic,
    t: Statistic,
    B: int = 200,
    return_samples: bool = False,
    theta_star: npt.NDArray[np.float64] | None = None,
    num_threads: int = 1,
) -> float | tuple[float, npt.NDArray[np.float64]]:
    """Estimate of bias

    Parameters
    ----------
     dist : EmpiricalDistribution
        The empirical distribution.
     stat : function
        The statistic for which we wish to calculate the bias.
     t : function
        Function to be applied to empirical distribution function.
     B : int, optional
        Number of bootstrap samples. Defaults to 200.
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
    -------
     bias : float
        Estimate of bias.
     theta_star : ndarray
        Array of bootstrapped statistic values. Only returned if
        `return_samples` is True.

    """
    if theta_star is None:
        theta_star = bootstrap_samples(dist, stat, B, num_threads=num_threads)

    tF_hat = dist.calculate_parameter(t)
    bias_est = np.mean(theta_star) - tF_hat

    if return_samples:
        return bias_est, theta_star
    else:
        return bias_est


def better_bootstrap_bias(
    x: ArrayLike,
    stat: Callable[[Any, npt.NDArray[np.float64]], float],
    B: int = 400,
    return_samples: bool = False,
    num_threads: int = 1,
) -> float | tuple[float, npt.NDArray[np.float64]]:
    r"""Better bootstrap bias.

    Parameters
    ----------
     x : array_like or pandas DataFrame
        The data.
     stat : function
        The statistic. Should take `x` and a resampling vector as
        input. See Notes.
     B : int, optional
        Number of bootstrap samples. Defaults to 400.
     return_samples : boolean, optional
        If True, return the bootstrapped statistic values. Defaults to False.
     num_threads : int, optional
        Number of threads to use for multicore processing. Defaults to
        1, meaning all calculations will be done in a single
        thread. Set to -1 to use all available cores.

    Returns
    -------
     bias : float
        Estimate of bias.
     theta_star : ndarray
        Array of bootstrapped statistic values. Only returned if
        `return_samples` is True.

    Notes
    -----
    The "better" bootstrap bias estimate is only applicable when
    `stat` is a plug-in statistic for the parameter being estimated,
    that is, having the form :math:`t(\hat{F})`, where :math:`\hat{F}`
    is the empirical distribution. Notable situations where this
    assumption does not hold include robust statistics like using the
    alpha-trimmed mean, since that is not the plug-in statistic for
    the mean. In cases like that, just use the "worse" `bias`
    function. The advantage of the "better" bootstrap bias estimate is
    faster convergence.  Whereas using `B` = 400 is typically adequate
    here, it can take thousands of bootstrap samples to give an
    accurate estimate in the "worse" `bias` function.

    """
    if num_threads == -1:
        num_threads = mp.cpu_count()

    def _bootstrap_sim(x, stat, batch_size, seed):
        if seed is not None:
            np.random.seed(seed)

        n = len(x)
        sum_p = np.zeros((n,))
        theta_star = np.zeros((batch_size,))
        for i in range(batch_size):
            p = _resampling_vector(n)
            sum_p += p
            theta_star[i] = stat(x, p)

        return theta_star, sum_p

    if num_threads == 1:
        theta_star, sum_p = _bootstrap_sim(x, stat, B, None)
    else:
        pool = Pool(num_threads)
        try:
            pool.restart()
        except AssertionError:
            pass

        results = []
        batch_size = B // num_threads
        extra = B % num_threads
        batch_sizes = [batch_size] * num_threads
        for i in range(extra):
            batch_sizes[i] += 1

        seeds = np.random.randint(0, 2**32 - 1, num_threads)
        for i, seed in enumerate(seeds):
            r = pool.apipe(_bootstrap_sim, x, stat, batch_sizes[i], seed)
            results.append(r)

        theta_star = []
        sum_p = np.zeros((len(x),))
        for res in results:
            theta_star_i, sum_p_i = res.get()
            theta_star.append(theta_star_i)
            sum_p += sum_p_i

        theta_star = np.hstack([t for t in theta_star])
        pool.close()
        pool.join()

    bias_est = np.mean(theta_star) - stat(x, sum_p / len(theta_star))

    if return_samples:
        return bias_est, theta_star
    else:
        return bias_est


def jackknife_bias(
    x: ArrayLike,
    stat: Statistic,
    return_samples: bool = False,
    jv: npt.NDArray[np.float64] | None = None,
    num_threads: int = 1,
) -> float | tuple[float, npt.NDArray[np.float64]]:
    r"""Jackknife estimate of bias.

    Parameters
    ----------
     x : array_like or pandas DataFrame
        The data.
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
     bias : float
        Estimate of bias.
     jv : ndarray
        Jackknife values. Only returned if `return_samples` is True.

    Notes
    -----
    The jackknife estimate of bias is only applicable when `stat` is a
    plug-in statistic, that is, having the form :math:`t(\hat{F})`,
    where :math:`\hat{F}` is the empirical distribution. Moreover, it
    is only applicable when `t` is a smooth function. Notable
    exceptions include the median. The jackknife cannot be used to
    estimate the bias of non-smooth estimators. See [EF93, S10.5] for
    details.

    """
    if jv is None:
        jv = jackknife_values(x, stat, num_threads=num_threads)

    n = len(jv)
    bias_est = (n - 1) * (np.mean(jv) - stat(x))
    if return_samples:
        return bias_est, jv
    else:
        return bias_est


def bias_corrected(
    x: ArrayLike,
    stat: Statistic | Callable[[Any, npt.NDArray[np.float64]], float],
    method: Literal[
        "better_bootstrap_bias", "bias", "jackknife"
    ] = "better_bootstrap_bias",
    dist: EmpiricalDistribution | None = None,
    t: Statistic | None = None,
    B: int | None = None,
    return_samples: bool = False,
    theta_star: npt.NDArray[np.float64] | None = None,
    jv: npt.NDArray[np.float64] | None = None,
    num_threads: int = 1,
) -> float | tuple[float, npt.NDArray[np.float64]]:
    """Bias-corrected estimator.

    Parameters
    ----------
     x : array_like or pandas DataFrame
        The data.
     stat : function
        The statistic. For use with "bias" and "jackknife" methods,
        should take a dataset as the input. For use with the
        "better_bootstrap_bias" method, it should take the dataset and
        a resampling vector as input. See the documentation in the
        better_bootstrap_bias function for details.
     method : ["better_bootstrap_bias", "bias", "jackknife"]
        The method by which we correct for bias. Defaults to
        "better_bootstrap_bias".
     dist : EmpiricalDistribution
        The empirical distribution. Required when method == "bias".
     t : function
        Function to be applied to empirical distribution
        function. Required when method == "bias".
     B : int, optional
        Number of bootstrap samples. Required when method == "bias" or
        "better_bootstrap_bias". Defaults to 400 when method ==
        "better_bootstrap_bias" or 4000 when method == "bias".
     return_samples : boolean, optional
        If True, return the bootstrapped statistic or jackknife
        values. Defaults to False.
     theta_star : array_like, optional
        Bootstrapped statistic values. Can be passed if they have
        already been calculated, which will speed this up
        considerably. Only used when method == "bias".
     jv : array_like, optional
        Jackknife values. Can be passed if they have already been
        calculated, which will speed this up considerably. Only used
        when method == "jackknife".
     num_threads : int, optional
        Number of threads to use for multicore processing. Defaults to
        1, meaning all calculations will be done in a single
        thread. Set to -1 to use all available cores.


    Returns
    -------
     theta_bar : float
        The bias-corrected estimator.
     theta_star : ndarray
        Array of bootstrapped statistic values. Only returned if
        `return_samples` is True and method is either "bias" or
        "better_bootstrap_bias".
     jv : ndarray
        Jackknife values. Only returned if `return_samples` is True
        and method == "jackknife".

    Notes
    -----
    Per [ET93, S10.6], bias-corrected estimators tend to have much
    higher variance than the non-corrected version. This should be
    assessed, for example, using the bootstrap to directly estimate
    the standard error of the corrected and uncorrected estimators.

    """
    if method == "better_bootstrap_bias":
        if B is None:
            B = 400
        n = len(x)
        p0 = np.ones((n,)) / n
        b, theta_star = better_bootstrap_bias(
            x, stat, B=B, return_samples=True, num_threads=num_threads
        )
        corrected = stat(x, p0) - b
        if return_samples:
            return corrected, theta_star
        else:
            return corrected
    elif method == "bias":
        if B is None:
            B = 4000

        b, theta_star = bias(
            dist,
            stat,
            t,
            B=B,
            theta_star=theta_star,
            return_samples=True,
            num_threads=num_threads,
        )
        corrected = stat(x) - b
        if return_samples:
            return corrected, theta_star
        else:
            return corrected
    elif method == "jackknife":
        b, jv = jackknife_bias(
            x, stat, jv=jv, return_samples=True, num_threads=num_threads
        )
        corrected = stat(x) - b
        if return_samples:
            return corrected, jv
        else:
            return corrected
