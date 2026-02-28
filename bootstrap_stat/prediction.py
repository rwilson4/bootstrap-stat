# pyre-unsafe
"""Prediction error and prediction interval methods."""

import multiprocessing as mp
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from pathos.multiprocessing import ProcessPool as Pool

from bootstrap_stat._utils import ArrayLike, _percentile
from bootstrap_stat.distributions import EmpiricalDistribution
from bootstrap_stat.sampling import bootstrap_samples


def prediction_error_optimism(
    dist: EmpiricalDistribution,
    data: ArrayLike,
    train: Callable[[ArrayLike], Any],
    predict: Callable[[Any, ArrayLike], Any],
    error: Callable[[Any, ArrayLike], npt.NDArray[np.float64]],
    B: int = 200,
    apparent_error: float | None = None,
    num_threads: int = 1,
) -> float:
    """Prediction Error, Optimism Method

    Parameters
    ----------
     dist : EmpiricalDistribution
       Empirical distribution.
     data : array_like or pandas DataFrame
        The data.
     train : function
       Function which takes as input a dataset sampled from the
       empirical distribution and returns a fitted model.
     predict : function
       Function which takes as input a fitted model and a dataset, and
       returns the predicted labels for that dataset
     error : function
       Function which takes as input a fitted model and a dataset, and
       returns the mean prediction error on that dataset.
     B : int, optional
        Number of bootstrap samples. Defaults to 200.
     apparent_error : float, optional
        The prediction error of the model on the dataset used to train
        the model, also known as the training error. If omitted, will
        be calculated. Can be passed to this function to save time,
        for example if the model had already been fit elsewhere.
     num_threads : int, optional
        Number of threads to use for multicore processing. Defaults to
        1, meaning all calculations will be done in a single
        thread. Set to -1 to use all available cores.

    Returns
    -------
     pe : float
        Prediction error.

    Notes
    -----
    The bootstrap estimate of prediction error can be used for model
    selection. It is similar to cross validation. It adds a bias
    correction term to the apparent error (the accuracy of the
    predictor applied to the same dataset used to train the
    predictor). This bias correction term is called the optimism. See
    [ET93, S17] for details.

    """
    if apparent_error is None:
        mdl = train(data)
        pred = predict(mdl, data)
        apparent_error = np.mean(error(pred, data))

    def stat(x):
        mdl_boot = train(x)

        pred_orig = predict(mdl_boot, data)  # Predictions for original dataset
        pred_boot = predict(mdl_boot, x)  # Predictions for bootstrap dataset

        err_orig = np.mean(error(pred_orig, data))  # Error for original dataset
        err_boot = np.mean(error(pred_boot, x))  # Error for bootstrap dataset

        optimism = err_orig - err_boot
        return optimism

    optimism = bootstrap_samples(dist, stat, B, num_threads=num_threads)
    pe = apparent_error + np.mean(optimism)
    return pe


def prediction_error_632(
    dist: EmpiricalDistribution,
    data: ArrayLike,
    train: Callable[[ArrayLike], Any],
    predict: Callable[[Any, ArrayLike], Any],
    error: Callable[[Any, ArrayLike], npt.NDArray[np.float64]],
    B: int = 200,
    apparent_error: float | None = None,
    use_632_plus: bool = False,
    gamma: float | None = None,
    no_inf_err_rate: Callable[[Any, ArrayLike], float] | None = None,
    num_threads: int = 1,
) -> float:
    """.632 Bootstrap

    Parameters
    ----------
     dist : EmpiricalDistribution
       Empirical distribution.
     data : array_like or pandas DataFrame
        The data.
     train : function
       Function which takes as input a dataset sampled from the
       empirical distribution and returns a fitted model.
     predict : function
       Function which takes as input a fitted model and a dataset, and
       returns the predicted labels for that dataset
     error : function
       Function which takes as input a fitted model and a dataset, and
       returns the prediction error *for each observation* in that
       dataset.
     B : int, optional
        Number of bootstrap samples. Defaults to 200.
     apparent_error : float, optional
        The prediction error of the model on the dataset used to train
        the model, also known as the training error. If omitted, will
        be calculated. Can be passed to this function to save time,
        for example if the model had already been fit elsewhere.
     use_632_plus : boolean, optional
        If True, uses the .632+ bootstrap. See Notes.
     num_threads : int, optional
        Number of threads to use for multicore processing. Defaults to
        1, meaning all calculations will be done in a single
        thread. Set to -1 to use all available cores.

    Returns
    -------
     pe : float
        Prediction error.

    Notes
    -----
    The .632 bootstrap estimate of prediction error is the weighted
    average of the apparent error and another term called eps0. The
    latter term is kind of like cross validation: we generate a
    bootstrap sample, train a model on it, and then make predictions
    using that model on the original dataset. But we only care about
    the predictions on observations that are *not* part of the
    bootstrap sample. We then average those prediction errors across
    all bootstrap samples. See [ET93, S17.7] for details.

    The method is so-called because the estimated prediction error is
    .368 times the apparent error plus .632 times eps0. [ET93] reports
    that this method performed better than leave-one-out cross
    validation in their simulations, having lower variance, but they
    themselves admit they had not thoroughly evaluated it. [Koh95]
    reported that the .632 bootstrap performed quite poorly when
    overfitting is present. This led Efron and Tibshirani to propose
    the .632+ bootstrap in [ET97]. Finally, [Arl10] surveys various
    approaches to model selection, recommending 10-fold cross
    validation as the preferred method of model selection.

    Because the .632 bootstrap has apparently not withstood the test
    of time, I have made no attempt to implement it efficiently,
    instead preferring the easy-to-follow approach below. This
    function should really only serve pedagogical purposes: it is not
    recommended for serious applications!

    The no information error rate looks at the prediction at point j,
    and computes the error *for every label y_i*. It averages the
    errors over all i and j. This is because, if we assume the
    features offer no insight into the labels, any observation is as
    good as any other at predicting any particular label.

    """
    if num_threads == -1:
        num_threads = mp.cpu_count()

    if apparent_error is None or (use_632_plus and gamma is None):
        mdl = train(data)
        pred = predict(mdl, data)
        if apparent_error is None:
            apparent_error = np.mean(error(pred, data))

        if use_632_plus and gamma is None:
            if no_inf_err_rate is None:
                raise ValueError(
                    "Please specify either the no information error rate,"
                    "gamma, or a function to calculate it."
                )
            gamma = no_inf_err_rate(pred, data)

    def _bootstrap_sim(dist, data, train, predict, error, B, seed):
        if seed is not None:
            np.random.seed(seed)

        n = len(data)
        # Total prediction error for each observation, over those
        # bootstrap samples that do not include that observation.
        Qi = np.zeros((n,))
        # Number of times the ith observation appeared in a bootstrap
        # sample.
        Bi = np.zeros((n,))
        for j in range(B):
            x_star, s_star = dist.sample(return_indices=True)
            mdl_boot = train(x_star)
            pred_orig = predict(mdl_boot, data)
            q = error(pred_orig, data)

            for i in range(n):
                if i not in s_star:
                    Bi[i] += 1
                    Qi[i] += q[i]

        return Bi, Qi

    if num_threads == 1:
        Bi, Qi = _bootstrap_sim(dist, data, train, predict, error, B, None)
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
            r = pool.apipe(
                _bootstrap_sim,
                dist,
                data,
                train,
                predict,
                error,
                batch_sizes[i],
                seed,
            )
            results.append(r)

        n = len(data)
        Bi = np.zeros((n,))
        Qi = np.zeros((n,))
        for res in results:
            Bii, Qii = res.get()
            Bi += Bii
            Qi += Qii

        pool.close()
        pool.join()

    eps0 = np.mean(Qi / Bi)
    err_632 = 0.368 * apparent_error + 0.632 * eps0

    if not use_632_plus:
        return err_632
    else:
        R = (eps0 - apparent_error) / (gamma - apparent_error)

        if eps0 <= gamma:
            eps0p = eps0
        else:
            eps0p = gamma

        if eps0p > apparent_error:
            Rp = R
        else:
            Rp = 0

        err_632p = err_632 + (
            (eps0p - apparent_error) * (0.368 * 0.632 * Rp) / (1 - 0.368 * Rp)
        )

        return err_632p


def prediction_interval(
    dist: EmpiricalDistribution,
    x: ArrayLike,
    mean: Callable[[ArrayLike], float] | None = None,
    std: Callable[[ArrayLike], float] | None = None,
    B: int = 1000,
    alpha: float = 0.05,
    t_star: npt.NDArray[np.float64] | None = None,
    return_t_star: bool = False,
    num_threads: int = -1,
) -> tuple[float, float] | tuple[float, float, npt.NDArray[np.float64]]:
    r"""Prediction interval

    Parameters
    ----------
     dist : EmpiricalDistribution
        The empirical distribution.
     x : array_like or pandas DataFrame
        The data.
     mean : function, optional
        A function returning the mean of a bootstrap sample. Defaults
        to np.mean, but this only works for arrays, not DataFrames. To
        emphasize, this function must return a float!
     std : function, optional
        A function returning the standard deviation of a bootstrap
        sample. Defaults to np.std using ddof=1. As with `mean`,
        specify something different for DataFrames!
     B : int, optional
        Number of bootstrap samples. Defaults to 1000.
     alpha : float, optional
        Number controlling the size of the interval. That is, this
        function will return a 100(1-2 * `alpha`)% prediction
        interval. Defaults to 0.05.
     t_star : array_like or None
        Array of studentized values, used to calculate the interval.
        Can be passed to this function to speed it up, for example
        when calculating multiple intervals.
     return_t_star : boolean, optional
        If True, return the studentized values. (Sometimes it is
        helpful to plot these.)
     num_threads : int, optional
        Number of threads to use for multicore processing. Defaults to
        1, meaning all calculations will be done in a single
        thread. Set to -1 to use all available cores.

    Returns
    -------
     pred_low, pred_high : float
        A 100(1 - 2 * `alpha`)% prediction interval on a point sampled
        from F.
     t_star : array
        Array of studentized values. Returned only if `return_t_star`
        is True.

    Notes
    -----
    Suppose we observe :math:`X_1, X_2, \ldots, X_n` sampled IID from
    a distribution :math:`F`. We wish to calculate a range of
    plausible values for a new point drawn from the same
    distribution. This function returns such a *prediction interval*.

    """
    if num_threads == -1:
        num_threads = mp.cpu_count()

    if mean is None:

        def mean(x):
            return np.mean(x)

    if std is None:

        def std(x):
            return np.std(x, ddof=1)

    x_bar = mean(x)
    s = std(x)

    def _bootstrap_sim(dist, mean, std, batch_size, seed):
        if seed is not None:
            np.random.seed(seed)

        t_star = np.empty((batch_size,))
        for i in range(batch_size):
            x_star = dist.sample()
            z_star = dist.sample(size=1)
            s_star = std(x_star)
            t_star[i] = (mean(x_star) - mean(z_star)) / s_star

        return t_star

    if t_star is None:
        if num_threads == 1:
            t_star = _bootstrap_sim(dist, mean, std, B, None)
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
                r = pool.apipe(_bootstrap_sim, dist, mean, std, batch_sizes[i], seed)
                results.append(r)

            t_star = np.hstack([res.get() for res in results])
            pool.close()
            pool.join()

    t_alpha = _percentile(t_star, [alpha, 1 - alpha])
    pred_low = x_bar - t_alpha[1] * s
    pred_high = x_bar - t_alpha[0] * s
    if return_t_star:
        return pred_low, pred_high, t_star
    else:
        return pred_low, pred_high
