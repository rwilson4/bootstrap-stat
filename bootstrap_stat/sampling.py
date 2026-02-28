# pyre-unsafe
"""Core bootstrap sampling functions."""

import multiprocessing as mp
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pathos.multiprocessing import ProcessPool as Pool

from bootstrap_stat._utils import ArrayLike, Statistic
from bootstrap_stat.distributions import EmpiricalDistribution


def jackknife_values(
    x: ArrayLike | tuple[ArrayLike, ...],
    stat: Statistic,
    sample: int | None = None,
    num_threads: int = 1,
) -> npt.NDArray[np.float64]:
    """Compute jackknife values.

    Parameters
    ----------
     x : array_like or pandas DataFrame or tuple of arrays/DataFrames.
        The data.
     stat : function
        The statistic.
     sample : int, optional
        When Jackknifing a multi-sample distribution, like for an A/B
        test, we generate one set of jackknife values for each
        sample. The caller should specify which sample for which
        jackknife values should be generated, calling this function
        once for each sample.
     num_threads : int, optional
        Number of threads to use for multicore processing. Defaults to
        1, meaning all calculations will be done in a single
        thread. Set to -1 to use all available cores.

    Returns
    -------
     jv : ndarray
        The jackknife values.

    Notes
    -----
    The jackknife values consist of the statistic applied to a
    collection of datasets derived from the original by holding out
    each observation in turn. For example, let x1 be the dataset
    corresponding to x, but with the first datapoint removed. The
    first jackknife value is simply stat(x1).

    """
    if num_threads == -1:
        num_threads = mp.cpu_count()

    if sample is not None and isinstance(x, tuple):
        # Multi-sample jackknife. Create a new statistic that is
        # simply a wrapper around the desired `stat`. Only perform the
        # hold-out logic on the specified sample.
        x = list(x)
        x_b = x[0:sample]
        x_s = x[sample]
        x_a = x[(sample + 1) :]

        def statistic(zz):
            xx = x_b + [zz] + x_a
            return stat((*xx,))

        return jackknife_values(x_s, statistic, num_threads=num_threads)

    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        is_dataframe = True
        n = len(x.index)
    else:
        is_dataframe = False
        x = np.array(x)
        n = len(x)

    def _jackknife_sim(x, stat, is_dataframe, start, end):
        n = end - start
        theta_i = np.empty((n,))
        for i in range(start, end):
            if is_dataframe:
                xi = x.drop(x.index[i])
            else:
                xi = np.delete(x, i)

            theta_i[i - start] = stat(xi)

        return theta_i

    if num_threads == 1:
        theta_i = _jackknife_sim(x, stat, is_dataframe, 0, n)
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
            r = pool.apipe(_jackknife_sim, x, stat, is_dataframe, start, end)
            results.append(r)
            start = end

        theta_i = np.hstack([res.get() for res in results])

        pool.close()
        pool.join()

    return theta_i


def multithreaded_bootstrap_samples(
    dist: EmpiricalDistribution,
    stat: Statistic | dict[str, Statistic],
    B: int,
    size: int | tuple[int, ...] | None = None,
    jackknife: bool = False,
    num_threads: int = -1,
) -> npt.NDArray[np.float64] | dict[str, npt.NDArray[np.float64]]:
    """Generate bootstrap samples in parallel.

    Parameters
    ----------
     dist : EmpiricalDistribution
        The empirical distribution.
     stat : function or dict_like
        The statistic or statistics for which we wish to calculate the
        standard error. If you want to compute different statistics on
        the same bootstrap samples, specify as a dictionary having
        functions as values.
     B : int
        Number of bootstrap samples.
     size : int or tuple of ints, optional
        Size to pass for generating samples. Defaults to None.
     num_threads : int, optional
        Number of threads to use. Defaults to the number of available
        CPUs.

    Returns
    -------
     theta_star : ndarray or dictionary
        Array of bootstrapped statistic values. When multiple
        statistics are being calculated on the same bootstrap samples,
        the return value will be a dictionary having keys the same as
        `stat` and values an ndarray for each statistic.

    Notes
    -----
    Jackknifing is not currently supported.

    """
    if num_threads == -1:
        num_threads = mp.cpu_count()

    def _bootstrap_sim(dist, stat, size, B, seed):
        if isinstance(stat, dict):
            theta_star = {k: np.empty((B,)) for k in stat}
        else:
            theta_star = np.empty((B,))

        np.random.seed(seed)
        for i in range(B):
            x_star = dist.sample(size=size)
            if isinstance(stat, dict):
                for k, v in stat.items():
                    theta_star[k][i] = v(x_star)
            else:
                theta_star[i] = stat(x_star)

        return theta_star

    pool = Pool(num_threads)
    try:
        # If we have used a pool before, we need to restart it.
        pool.restart()
    except AssertionError:
        # If have never used a pool before, no need to do anything.
        pass

    results = []
    batch_size = B // num_threads
    extra = B % num_threads
    batch_sizes = [batch_size] * num_threads
    for i in range(extra):
        batch_sizes[i] += 1

    seeds = np.random.randint(0, 2**32 - 1, num_threads)
    for i, seed in enumerate(seeds):
        r = pool.apipe(_bootstrap_sim, dist, stat, size, batch_sizes[i], seed)
        results.append(r)

    if isinstance(stat, dict):
        theta_star = {k: [] for k in stat}
        for res in results:
            t = res.get()
            for k in stat:
                theta_star[k].extend(t[k])
        theta_star = {k: np.array(v) for k, v in theta_star.items()}
    else:
        theta_star = np.hstack([res.get() for res in results])

    pool.close()
    pool.join()
    return theta_star


def bootstrap_samples(
    dist: EmpiricalDistribution,
    stat: Statistic | dict[str, Statistic],
    B: int,
    size: int | tuple[int, ...] | None = None,
    jackknife: bool = False,
    num_threads: int = 1,
) -> (
    npt.NDArray[np.float64]
    | dict[str, npt.NDArray[np.float64]]
    | tuple[npt.NDArray[np.float64] | dict[str, npt.NDArray[np.float64]], list[Any]]
):
    """Generate bootstrap samples.

    Parameters
    ----------
     dist : EmpiricalDistribution
        The empirical distribution.
     stat : function or dict_like
        The statistic or statistics for which we wish to calculate the
        standard error. If you want to compute different statistics on
        the same bootstrap samples, specify as a dictionary having
        functions as values.
     B : int
        Number of bootstrap samples.
     size : int or tuple of ints, optional
        Size to pass for generating samples. Defaults to None.
     jackknife : boolean, optional
        If True, returns an array of jackknife bootstrap statistics
        (see the `jackknife_array` return value). Defaults to False.
     num_threads : int, optional
        Number of threads to use for multicore processing. Defaults to
        1, meaning all calculations will be done in a single
        thread. Set to -1 to use all available cores.

    Returns
    -------
     theta_star : ndarray or dictionary
        Array of bootstrapped statistic values. When multiple
        statistics are being calculated on the same bootstrap samples,
        the return value will be a dictionary having keys the same as
        `stat` and values an ndarray for each statistic.
     jackknife_array : ndarray
        Array of n arrays or dicts, where n is the number of elements
        of the original dataset upon which the empirical distribution
        is based. Each element of this array is in turn either an
        array or dict according to `stat`. If `stat` is just a single
        function, it will be an array, otherwise a dict. The ith
        element of the inner array will be those bootstrap statistics
        corresponding to samples not including the ith value from the
        original dataset. (For jackknifing, only samples not including
        the ith datapoint can be used for inferences involving the ith
        point.) See [ET93, S19.4] for details. Only returned if
        `jackknife` is True.

    """
    if num_threads == -1:
        num_threads = mp.cpu_count()

    if jackknife and num_threads > 1:
        warnings.warn(
            "Multicore support for jackknifing not yet supported. "
            "Continuing with 1 core"
        )
        num_threads = 1

    if num_threads > 1:
        return multithreaded_bootstrap_samples(
            dist, stat, B, size=size, num_threads=num_threads
        )

    if isinstance(stat, dict):
        theta_star = {k: np.empty((B,)) for k in stat}
    else:
        theta_star = np.empty((B,))

    if jackknife:
        jackknife_array = None

    for i in range(B):
        if jackknife:
            x_star, ind = dist.sample(size=size, return_indices=True)
        else:
            x_star = dist.sample(size=size)

        if jackknife and jackknife_array is None:
            n = dist.n
            if isinstance(stat, dict):
                jackknife_array = [{k: []} for _ in range(n) for k in stat]
            else:
                jackknife_array = [[] for _ in range(n)]

        if isinstance(stat, dict):
            for k, v in stat.items():
                theta_star[k][i] = v(x_star)
        else:
            theta_star[i] = stat(x_star)

        if jackknife:
            for j in range(len(jackknife_array)):
                if j not in ind:
                    if isinstance(stat, dict):
                        for k in stat:
                            jackknife_array[j][k] = theta_star[k][i]
                    else:
                        jackknife_array[j].append(theta_star[i])

    if jackknife:
        return theta_star, jackknife_array
    else:
        return theta_star
