# pyre-unsafe
"""Private utility functions and type aliases for bootstrap methods."""

import multiprocessing as mp
import warnings
from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as ss
from pathos.multiprocessing import ProcessPool as Pool

# Type aliases for common types
ArrayLike: TypeAlias = npt.NDArray[np.float64] | list[float] | pd.Series | pd.DataFrame
Statistic: TypeAlias = Callable[..., float]
JackknifeValues: TypeAlias = (
    npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], ...]
)


def _bca_acceleration(jv: JackknifeValues) -> float:
    """Compute the BCa acceleration.

    Parameters
    ----------
     jv : array_like or tuple
        Jackknife values. If tuple, interpretation is that each
        element of the tuple is a set of jackknife values for a
        multi-sample problem.

    Returns
    -------
     a_hat : float
        Acceleration for use with the BCa algorithm.

    """
    if isinstance(jv, tuple):
        num = 0
        den = 0
        for jv_s in jv:
            n = len(jv_s)
            theta_dot = np.mean(jv_s)
            U = (n - 1) * (theta_dot - jv_s)

            n2 = n * n
            U2 = U * U
            num += np.sum(U2 * U) / (n2 * n)
            den += np.sum(U2) / n2
        den = 6 * (den**1.5)
    else:
        theta_dot = np.mean(jv)
        U = theta_dot - jv
        U2 = U * U
        num = np.sum(U2 * U)
        den = 6 * ((np.sum(U2)) ** 1.5)

    a_hat = num / den
    return a_hat


def _adjust_percentiles(
    alpha: float, a_hat: float, z0_hat: float
) -> tuple[float, float]:
    """Adjusted percentiles

    Parameters
    ----------
     alpha : float
        Number controlling the size of the confidence interval.
     a_hat, z0_hat : float
        BCa parameters

    Returns
    -------
     alpha1, alpha2 : float
        Adjusted percentiles.

    Notes
    -----
    Computes the adjusted percentiles for use with the BCa
    algorithm. Taken from Eq. 14.10 in [ET93].

    """
    z_alpha = ss.norm.ppf(alpha)
    z_one_m_alpha = ss.norm.ppf(1 - alpha)
    alpha1 = ss.norm.cdf(z0_hat + (z0_hat + z_alpha) / (1 - a_hat * (z0_hat + z_alpha)))
    alpha2 = ss.norm.cdf(
        z0_hat + (z0_hat + z_one_m_alpha) / (1 - a_hat * (z0_hat + z_one_m_alpha))
    )

    return alpha1, alpha2


def _percentile(
    z: npt.NDArray[np.float64], p: float | list[float], full_sort: bool = True
) -> npt.NDArray[np.float64]:
    """Percentiles of an array.

    Parameters
    ----------
     z : array_like
        Data. Not assumed to be sorted.
     p : float or array of floats
        Number in (0, 0.5), specifing the percentiles.
     full_sort : boolean, optional
        Whether to fully sort z (see Notes). Defaults to False.

    Returns
    -------
     p : [low, high]
        The alpha and 1-alpha percentiles of z.

    Notes
    -----
    Uses the methodology recommended in S12.5 of [ET93]. When z is
    very large, sorting is inefficient and unnecessary. However, for
    modest B, doing a partial sort isn't actually any faster.

    """
    B = len(z)
    if not isinstance(p, list):
        p = [p]

    if full_sort:
        sorted_z = sorted(z)

    percentiles = np.zeros((len(p),))
    for i, pi in enumerate(p):
        if pi <= 0.5:
            alpha = pi
            Balpha = B * alpha
        else:
            Balpha = B - B * pi
            alpha = 1 - pi

        if int(Balpha) == Balpha:
            k = int(Balpha)
            if pi > 0.5:
                k = B - k
        else:
            k = int(np.floor(Balpha + alpha))
            if pi > 0.5:
                k = B + 1 - k

        if k <= 0 or k >= B:
            warnings.warn("Index outside of bounds. Try more bootstrap samples.")
            if k <= 0:
                k = 0
            elif k >= B:
                k = B

        if full_sort:
            percentiles[i] = sorted_z[k - 1]
        else:
            percentiles[i] = np.partition(z, k - 1)[k - 1]

    return percentiles


def loess(
    z0: float,
    z: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    alpha: float,
    sided: Literal["both", "trailing", "leading"] = "both",
) -> float:
    """Locally estimated scatterplot smoothing

    Parameters
    ----------
     z0 : float
        Test point
     z : array_like
        Endogenous variables.
     y : array_like
        Exogenous variables.
     alpha : float
        Smoothing parameter, governing how many points to include in
        the local fit. Should be between 0 and 1, with higher
        values corresponding to more smoothing.
     sided : ["both", "trailing", "leading"], optional
        Dictates what side(s) of z0 can be used to create the
        smooth. With the default behavior ("both"), points both before
        and after z0 can be used. If "trailing" is specified, only
        points less than or equal to z0 may be used to perform the
        smooth. If "leading" is specified, only points greater than or
        equal to z0 may be used. This is intended to support time
        series methods where we only want to do trailing averages.

    Returns
    -------
     y_smoothed : float
        The smoothed estimate of the exogenous variable evaluated at
        z0.

    """
    N = len(z)
    n = int(np.floor(alpha * N))

    if sided == "trailing":
        ii = np.argwhere(z <= z0).flatten()
        if len(ii) == 0:
            return np.nan
        y = y[ii]
        z = z[ii]
    elif sided == "leading":
        ii = np.argwhere(z >= z0).flatten()
        if len(ii) == 0:
            return np.nan
        y = y[ii]
        z = z[ii]

    ii = np.argsort(np.abs(z - z0))
    if n == 0:
        return y[ii[0]]

    Nz = z[ii[0:n]]
    Ny = y[ii[0:n]]
    if len(Nz) == 1:
        return Ny[0]

    u = np.abs(Nz - z0) / np.abs(Nz[-1] - z0)
    w = 1 - u * u * u
    w = w * w * w
    wsqrt = np.sqrt(w)

    slope, intercept, _, _, _ = ss.linregress(Nz * wsqrt, Ny * wsqrt)
    return intercept + slope * z0


def _resampling_vector(n: int) -> npt.NDArray[np.float64]:
    """Resampling vector.

    Parameters
    ----------
     n : int
        Number of observations in dataset.

    Returns
    -------
     p : array of `n` floats
        Resampling vector.

    Notes
    -----
    Suppose we have a dataset x_1, ..., x_n. A bootstrap sample is a
    collection of n observations drawn with replacement from
    {x_i}. Let p_j be the number of bootstrap samples equal to the jth
    sample from the original dataset, divided by n. Then the vector
    with components p_j is the resampling vector. Each entry has the
    interpretation of being a proportion.

    """
    x = range(1, n + 1)
    xStar = np.random.choice(x, n, replace=True)
    p = np.array([np.count_nonzero(xStar == i) for i in x], np.float64)
    return p / n


def _influence_components(
    x: ArrayLike | tuple[ArrayLike, ...],
    stat: Callable[[Any, npt.NDArray[np.float64]], float],
    order: Literal[1, 2] = 1,
    eps: float = 1e-3,
    num_threads: int = 1,
) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Influence components

    Parameters
    ----------
     x : array_like or pandas DataFrame or tuple
        The data. If tuple. interpretation is as a MultiSample
        distribution, like in A/B testing.
     stat : function
        The statistic. Should take `x` and a resampling vector as
        input. See Notes.
     order : [1, 2], optional
        How many derivatives to compute. Defaults to 1.
     eps : float, optional
        Epsilon for limit calculation. Deafults to 1e-6.
     num_threads : int, optional
        Number of threads to use for multicore processing. Defaults to
        1, meaning all calculations will be done in a single
        thread. Set to -1 to use all available cores.

    Returns
    -------
     influence_components : array_like
        Vector of influence components.
     second_derivatives : array_like
        Vector of second derivatives. Only returned if
        `order` == 2.

    Notes
    -----
    The ith influence component is the derivative of `stat` with
    respect to the ith data point. We use a central difference method
    for computing this.

    The ith second derivative is (obviously) the second derivative of
    `stat` with respect to the ith data point.

    """
    if num_threads == -1:
        num_threads = mp.cpu_count()

    n = len(x)
    p0 = np.ones((n,)) / n
    t0 = stat(x, p0)

    def _jackknife_sim(x, stat, eps, t0, start, end):
        n = len(x)
        p0 = np.ones((n,)) / n
        batch_size = end - start
        t_dot = np.zeros((batch_size,))
        t_ddot = np.zeros((batch_size,))
        for i in range(start, end):
            di = np.zeros((n,))
            di[i] = 1
            di = di - p0
            tp = stat(x, p0 + eps * di)
            tm = stat(x, p0 - eps * di)
            t_dot[i - start] = (tp - tm) / (2 * eps)
            if order > 1:
                t_ddot[i - start] = (tp - 2 * t0 + tm) / (eps * eps)

        return t_dot, t_ddot

    if num_threads == 1:
        t_dot, t_ddot = _jackknife_sim(x, stat, eps, t0, 0, n)
    else:
        pool = Pool(num_threads)
        try:
            pool.restart()
        except AssertionError:
            pass

        results = []
        batch_size = n // num_threads
        extra = n % num_threads
        batch_sizes = [batch_size] * num_threads
        for i in range(extra):
            batch_sizes[i] += 1

        start = 0
        for i in range(num_threads):
            end = start + batch_sizes[i]
            r = pool.apipe(_jackknife_sim, x, stat, eps, t0, start, end)
            results.append(r)
            start = end

        t_dot = []
        t_ddot = []
        for res in results:
            t_dot_i, t_ddot_i = res.get()
            t_dot.append(t_dot_i)
            t_ddot.append(t_ddot_i)

        t_dot = np.hstack(t_dot)
        t_ddot = np.hstack(t_ddot)

        pool.close()
        pool.join()

    if order == 1:
        return t_dot
    else:
        return t_dot, t_ddot
