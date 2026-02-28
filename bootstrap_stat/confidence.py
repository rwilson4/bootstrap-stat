# pyre-unsafe
"""Confidence interval methods."""

import multiprocessing as mp
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.optimize as optimize
import scipy.stats as ss
from pathos.multiprocessing import ProcessPool as Pool

from bootstrap_stat._utils import (
    ArrayLike,
    JackknifeValues,
    Statistic,
    _adjust_percentiles,
    _bca_acceleration,
    _influence_components,
    _percentile,
    loess,
)
from bootstrap_stat.distributions import EmpiricalDistribution
from bootstrap_stat.sampling import bootstrap_samples, jackknife_values
from bootstrap_stat.standard_error import standard_error


def t_interval(
    dist: EmpiricalDistribution,
    stat: Statistic,
    theta_hat: float,
    stabilize_variance: bool = False,
    se_hat: float | None = None,
    fast_std_err: Callable[[ArrayLike], float] | None = None,
    alpha: float = 0.05,
    Binner: int = 25,
    Bouter: int = 1000,
    Bvar: int = 100,
    size: int | tuple[int, ...] | None = None,
    empirical_distribution: type[EmpiricalDistribution] = EmpiricalDistribution,
    return_samples: bool = False,
    theta_star: npt.NDArray[np.float64] | None = None,
    se_star: npt.NDArray[np.float64] | None = None,
    z_star: npt.NDArray[np.float64] | None = None,
    num_threads: int = 1,
) -> (
    tuple[float, float]
    | tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]
    | tuple[
        float,
        float,
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]
):
    """Bootstrap-t Intervals

    Parameters
    ----------
     dist : EmpiricalDistribution
        The empirical distribution.
     stat : function
        The statistic for which we wish to calculate a confidence
        interval.
     theta_hat : float
        The observed statistic.
     stabilize_variance : boolean, optional
        If True, use the variance stabilization technique. Defaults to
        False.
     se_hat : float or None, optional
        The standard error of the observed data. If None (default),
        will be calculated using the nonrobust bootstrap estimate of
        standard error, using the default number of iterations. The
        user may wish to use a nondefault number of bootstrap
        iterations to calculate this, or a robust variant. If so, the
        user should calculate this externally and pass it to this
        function.
     fast_std_err : function or None, optional
        To speed this up, the user may specify a fast function for
        computing the standard error of a bootstrap sample. If not
        specified, we will use the nonrobust bootstrap estimate of
        standard error, using the default number of iterations. This
        can also be used to specify a nondefault bootstrap
        methodology, such as a robust version. See Examples for some
        examples.
     alpha : float, optional
        Number controlling the size of the interval. That is, this
        function will return a 100(1-2*`alpha`)% confidence
        interval. Defaults to 0.05, corresponding to a 90% confidence
        interval.
     Binner : int, optional
        Number of bootstrap samples for calculating standard
        error. Defaults to 25.
     Bouter : int, optional
        Number of bootstrap samples for calculating
        percentiles. Defaults to 1000.
     Bvar : int, optional
        Number of bootstrap samples used to estimate the relationship
        between the statistic and the standard error for use with
        variance stabilization. Defaults to 100.
     size : int or tuple of ints, optional
        Size to pass for generating samples from the distribution.
        Defaults to None, indicating the samples will be the same size
        as the original dataset.
     empirical_distribution : class, optional
        Class to be used to generate an empirical distribution.
        Defaults to the regular EmpiricalDistribution, but any class
        that implements a sample method will work. For example, the
        MultiSampleEmpiricalDistribution can be used. This can be used
        to accommodate more exotic applications of the Bootstrap.
     return_samples : boolean, optional
        If True, return the bootstrapped statistic values. Defaults to False.
     theta_star : array_like, optional
        Bootstrapped statistic values. Can be passed if they have
        already been calculated, which will speed this up
        considerably.
     se_star : array_like, optional
        Bootstrapped statistic standard errors. Can be passed if they
        have already been calculated, which will speed this up
        considerably.
     z_star : array_like, optional
        Bootstrapped pivot values. Can be passed if they have already
        been calculated, which will speed this up considerably.
     num_threads : int, optional
        Number of threads to use for multicore processing. Defaults to
        1, meaning all calculations will be done in a single
        thread. Set to -1 to use all available cores.

    Returns
    -------
     ci_low, ci_high : float
        Lower and upper bounds on a 100(1-2*`alpha`)% confidence
        interval on theta.
     theta_star : ndarray
        Array of bootstrapped statistic values. Only returned if
        `return_samples` is True.
     se_star : ndarray
        Array of bootstrapped statistic standard errors. Only returned
        if `return_samples` is True.
     z_star : ndarray
        Array of bootstrapped pivot values. Only returned if
        `return_samples` is True and `stabilize_variance` is False.

    Examples
    --------
    >>> x = np.random.randn(100)
    >>> dist = EmpiricalDistribution(x)
    >>> def statistic(x): return np.mean(x)
    >>> theta_hat = statistic(x)
    >>> ci_low, ci_high = t_interval(dist, statistic, theta_hat)

    >>> se_hat = standard_error(dist, statistic, robustness=0.95, B=2000)
    >>> ci_low, ci_high = t_interval(dist, statistic, theta_hat, se_hat=se_hat)

    >>> def fast_std_err(x): return np.sqrt(np.var(x, ddof=1) / len(x))
    >>> ci_low, ci_high = t_interval(dist, statistic, theta_hat,
    ...                              fast_std_err=fast_std_err)

    >>> def fast_std_err(x):
    ...    dist = EmpiricalDistribution(x)
    ...    return standard_error(dist, statistic, robustness=0.95, B=2000)
    >>> ci_low, ci_high = t_interval(dist, statistic, theta_hat,
    ...                              fast_std_err=fast_std_err)

    """
    if se_hat is None and not stabilize_variance:
        se_hat = standard_error(dist, stat, size=size, num_threads=num_threads)

    if stabilize_variance:
        B = Bvar
    else:
        B = Bouter

    # This function runs in two modes: with or without variance
    # stabilization. In addition, the user has the option of passing
    # theta_star and se_star to this function, in which case these are
    # always interpreted as the basis for calculating the
    # interval. The variance-stabilizing transformation is always
    # calculated separately. Doing so enables us to calculate one set
    # of bootstrap statistics and recycle them for percentile
    # intervals, standardized intervals, BCa intervals, t-intervals
    # without stabilization, and t-intervals with stabilization. Of
    # course, if we aren't comparing various techniques this ability
    # to recycle the same bootstrap statistics is irrelevant.
    if stabilize_variance or (
        z_star is None and (theta_star is None or se_star is None)
    ):
        statistics = {"theta_star": stat}
        if fast_std_err is not None:
            statistics["se_star"] = fast_std_err
        else:
            statistics["se_star"] = lambda x: standard_error(
                empirical_distribution(x), stat, B=Binner
            )

        boot_stats = bootstrap_samples(
            dist, statistics, B, size=size, num_threads=num_threads
        )

        if stabilize_variance:
            # These values are used to estimate the
            # variance-stabilizing transformation, not to calculate
            # the confidence interval.
            var_theta_star = boot_stats["theta_star"]
            var_se_star = boot_stats["se_star"]
        else:
            theta_star = boot_stats["theta_star"]
            se_star = boot_stats["se_star"]

    if not stabilize_variance and z_star is None:
        z_star = np.empty((len(theta_star),))
        z_star = (theta_star - theta_hat) / se_star

    if not stabilize_variance:
        p = _percentile(z_star, [alpha, 1 - alpha])

        ci_low = theta_hat - p[1] * se_hat
        ci_high = theta_hat - p[0] * se_hat
    else:

        def one_over_s(u):
            """1 / s(u)

            s(u) is the estimated standard error of the statistic as a
            function of its mean value. We use a smoother (loess) to
            estimate it.

            """
            alpha = 0.3
            return 1.0 / loess(u, var_theta_star, var_se_star, alpha)

        theta_star_min = min(var_theta_star)
        theta_star_max = max(var_theta_star)
        theta_star_mid = 0.5 * (theta_star_min + theta_star_max)
        dx = (theta_star_max - theta_star_min) / 50.0

        def g(x, a=theta_star_mid):
            """Integral of 1/s

            Uses the trapezoid rule to compute the definite integral
            of 1 / s(u) from `a` to `x`, with `a` defaulting to the
            midrange of the observed statistic values. Using this
            default led to considerable speedups since when `x` is
            close to `a`, we need fewer evaluations of s.

            """
            if a == x:
                return 0.0
            elif a > x:
                return -g(a, x)

            # Hereafter we assume a < x
            integral = 0
            one_over_s_a = one_over_s(a)
            while a < x:
                if a + dx < x:
                    b = a + dx
                else:
                    b = x

                one_over_s_b = one_over_s(b)
                integral += (b - a) * (one_over_s_a + one_over_s_b)
                a = b
                one_over_s_a = one_over_s_b
            return 0.5 * integral

        def new_stat(x):
            """Reexpress the statistic in the transformed space"""
            return g(stat(x))

        # If theta_star was passed to this function, transform it to
        # the variance-stablized scale and pass it along.
        g_theta_hat = g(theta_hat)
        if theta_star is not None:
            # We should be able to speed this up using multicore, but
            # when I try to pickle g I get a maximum recursion depth
            # error.
            z_star = np.array([g(t_i) - g_theta_hat for t_i in theta_star])
        else:
            # Any z_star passed to this function would be meaningless.
            z_star = None

        # For some reason, multithreading doesn't work in this
        # call. And, since we are using 1 as the fast_std_err, this is
        # really fast anyway, so nothing to speed up.
        gci_low, gci_high = t_interval(
            dist,
            new_stat,
            g_theta_hat,
            stabilize_variance=False,
            se_hat=1,
            fast_std_err=lambda x: 1,
            alpha=alpha,
            Binner=Binner,
            Bouter=Bouter,
            size=size,
            empirical_distribution=empirical_distribution,
            z_star=z_star,
            num_threads=1,
        )

        try:
            ci_low = optimize.root_scalar(
                lambda x: g(x) - gci_low,
                method="bisect",
                bracket=(theta_star_min, theta_star_max),
            ).root

            ci_high = optimize.root_scalar(
                lambda x: g(x) - gci_high,
                method="bisect",
                bracket=(theta_star_min, theta_star_max),
            ).root
        except ValueError:
            warnings.warn(
                "Confidence limit outside values used to estimate variance-"
                "stabilizing transformation. Try specifying a higher `Bvar` "
                "when calling this function."
            )
            raise

    if return_samples:
        if stabilize_variance:
            return ci_low, ci_high, theta_star, se_star
        else:
            return ci_low, ci_high, theta_star, se_star, z_star
    else:
        return ci_low, ci_high


def percentile_interval(
    dist: EmpiricalDistribution,
    stat: Statistic,
    alpha: float = 0.05,
    B: int = 1000,
    size: int | tuple[int, ...] | None = None,
    return_samples: bool = False,
    theta_star: npt.NDArray[np.float64] | None = None,
    num_threads: int = 1,
) -> tuple[float, float] | tuple[float, float, npt.NDArray[np.float64]]:
    """Percentile Intervals

    Parameters
    ----------
     dist : EmpiricalDistribution
        The empirical distribution.
     stat : function
        The statistic.
     alpha : float, optional
        Number controlling the size of the interval. That is, this
        function will return a 100(1 - 2 * `alpha`)% confidence
        interval. Defaults to 0.05.
     B : int, optional
        Number of bootstrap samples. Defaults to 1000.
     size : int or tuple of ints, optional
        Size to pass for generating samples from the distribution.
        Defaults to None, indicating the samples will be the same size
        as the original dataset.
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
     ci_low, ci_high : float
        Lower and upper bounds on a 100(1-2*`alpha`)% confidence
        interval on theta.
     theta_star : ndarray
        Array of bootstrapped statistic values. Only returned if
        `return_samples` is True.

    """
    if theta_star is None:
        theta_star = bootstrap_samples(
            dist, stat, B, size=size, num_threads=num_threads
        )

    p = _percentile(theta_star, [alpha, 1 - alpha])
    if return_samples:
        return p[0], p[1], theta_star
    else:
        return p[0], p[1]


def bcanon_interval(
    dist: EmpiricalDistribution,
    stat: Statistic,
    x: ArrayLike | tuple[ArrayLike, ...],
    alpha: float = 0.05,
    B: int = 1000,
    size: int | tuple[int, ...] | None = None,
    return_samples: bool = False,
    theta_star: npt.NDArray[np.float64] | None = None,
    theta_hat: float | None = None,
    jv: JackknifeValues | None = None,
    num_threads: int = 1,
) -> (
    tuple[float, float] | tuple[float, float, npt.NDArray[np.float64], JackknifeValues]
):
    """BCa Confidence Intervals

    Parameters
    ----------
     dist : EmpiricalDistribution
        The empirical distribution.
     stat : function
        The statistic.
     x : array_like or pandas DataFrame or tuple
        The data, used to evaluate the observed statistic and compute
        jackknife values.
     alpha : float, optional
        Number controlling the size of the interval. That is, this
        function will return a 100(1-2*`alpha`)% confidence
        interval. Defaults to 0.05.
     B : int, optional
        Number of bootstrap samples. Defaults to 1000.
     size : int or tuple of ints, optional
        Size to pass for generating samples from the distribution.
        Defaults to None, indicating the samples will be the same size
        as the original dataset.
     return_samples : boolean, optional
        If True, return the bootstrapped statistic values. Defaults to False.
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
     num_threads : int, optional
        Number of threads to use for multicore processing. Defaults to
        1, meaning all calculations will be done in a single
        thread. Set to -1 to use all available cores.

    Returns
    -------
     ci_low, ci_high : float
        Lower and upper bounds on a 100(1-2*`alpha`)% confidence
        interval on theta.
     theta_star : ndarray
        Array of bootstrapped statistic values. Only returned if
        `return_samples` is True.
     jv : ndarray
        Jackknife values. Only returned if `return_samples` is True.

    """
    # The observed value of the statistic.
    if theta_hat is None:
        theta_hat = stat(x)

    if theta_star is None:
        theta_star = bootstrap_samples(
            dist, stat, B, size=size, num_threads=num_threads
        )

    zb = (theta_star < theta_hat).sum()
    z0_hat = ss.norm.ppf(zb / len(theta_star))

    if jv is None:
        if dist.is_multi_sample:
            jv = [
                jackknife_values(x, stat, sample=i, num_threads=num_threads)
                for i in range(len(x))
            ]
            jv = (*jv,)
        else:
            jv = jackknife_values(x, stat, num_threads=num_threads)

    a_hat = _bca_acceleration(jv)
    alpha1, alpha2 = _adjust_percentiles(alpha, a_hat, z0_hat)

    p = _percentile(theta_star, [alpha1, alpha2])
    if return_samples:
        return p[0], p[1], theta_star, jv
    else:
        return p[0], p[1]


def abcnon_interval(
    x: ArrayLike,
    stat: Callable[[Any, npt.NDArray[np.float64]], float],
    alpha: float | list[float] = 0.05,
    eps: float = 0.001,
    influence_components: npt.NDArray[np.float64] | None = None,
    second_derivatives: npt.NDArray[np.float64] | None = None,
    return_influence_components: bool = False,
    num_threads: int = 1,
) -> tuple[float, ...]:
    """ABC Confidence Intervals

    Parameters
    ----------
     x : array_like or pandas DataFrame
        The data, used to evaluate the observed statistic and compute
        influence components.
     stat : function
        The statistic. Should take `x` and a resampling vector as
        input. See Notes.
     alpha : float or array of floats, optional
        Number controlling the size of the interval. That is, this
        function will return a 100(1 - 2 * `alpha`)% confidence
        interval. Defaults to 0.05. Alternatively, the user can pass
        an array of floats in (0, 1). In that case, the return value
        will be a tuple of confidence points. See Notes.
     influence_components : array_like, optional
        Influence components. See Notes.
     second_derivatives : array_like, optional
        Vector of second derivatives. See Notes.
     return_influence_components : boolean, optional
        Specifies whether to return the influence components and
        second derivatives. Defaults to False.
     num_threads : int, optional
        Number of threads to use for multicore processing. Defaults to
        1, meaning all calculations will be done in a single
        thread. Set to -1 to use all available cores.

    Returns
    -------
     ci_low, ci_high : float
        Lower and upper bounds on a 100(1 - 2 * `alpha`)% confidence
        interval on theta.
     influence_components : array_like
        Vector of influence components. Only returned if
        `return_influence_components` is True.
     second_derivatives : array_like
        Vector of second_derivatives. Only returned if
        `return_influence_components` is True.

    Notes
    -----
    Approximate Bootstrap Confidence (ABC) intervals require the
    statistic to be expressed in resampling form. See [ET93, S14.4]
    for details. It only applies to statistics which are smooth
    functions of the data. A notable example where ABC does not apply
    is the sample median.

    ABC works in terms of the derivatives and second derivatives of
    the statistic with respect to the data. In some cases, analytical
    forms are possible. In that case, the caller may wish to calculate
    them externally to this function and pass them in. The default
    behavior is to calculate them using a finite difference method.

    When we want to compute multiple confidence points, we can reuse
    many of the calculations. For example, if we want to compute a 90%
    confidence interval *and* a 99% confidence interval, we would
    specify `alpha` = [0.005, 0.05, 0.95, 0.995] and read off the
    appropriate return values. This would be much faster than multiple
    calls to this function. (The recurring cost is 1 call to `stat`
    for each additional point specified.) This in turn facilitates
    using these intervals for achieved significance levels,
    effectively by inverting the interval having endpoint 0. See the
    ASL functions for how this might be done.

    """
    n = len(x)
    n2 = n * n
    eps /= n
    p0 = np.ones((n,)) / n
    t0 = stat(x, p0)

    if influence_components is None or second_derivatives is None:
        influence_components, second_derivatives = _influence_components(
            x, stat, order=2, eps=eps, num_threads=num_threads
        )

    sum_inf_squared = np.sum(influence_components**2)
    sigma_hat = np.sqrt(sum_inf_squared) / n
    a_hat = np.sum(influence_components**3) / (6 * sum_inf_squared**1.5)

    delta_hat = influence_components / (n2 * sigma_hat)
    c_q = (stat(x, p0 + eps * delta_hat) - 2 * t0 + stat(x, p0 - eps * delta_hat)) / (
        2 * sigma_hat * eps * eps
    )

    b_hat = np.sum(second_derivatives) / (2 * n * n)
    gamma_hat = b_hat / sigma_hat - c_q
    z0_hat = ss.norm.ppf(2 * ss.norm.cdf(a_hat) * ss.norm.cdf(-gamma_hat))

    try:
        for a in alpha:
            break
    except TypeError:
        alpha = [alpha, 1 - alpha]

    conf_points = []
    for a in alpha:
        w = z0_hat + ss.norm.ppf(a)
        lmbda = w / ((1 - a_hat * w) ** 2)
        conf_points.append(stat(x, p0 + lmbda * delta_hat))

    if return_influence_components:
        conf_points.extend([influence_components, second_derivatives])

    return (*conf_points,)


def calibrate_interval(
    dist: EmpiricalDistribution,
    stat: Callable[[Any, npt.NDArray[np.float64]], float],
    x: ArrayLike | tuple[ArrayLike, ...],
    theta_hat: float,
    alpha: float = 0.05,
    B: int = 1000,
    return_confidence_points: bool = False,
    num_threads: int = 1,
) -> tuple[float, float] | tuple[float, float, float, float]:
    """Calibrated confidence interval

    Parameters
    ----------
     dist : EmpiricalDistribution
        The empirical distribution.
     stat : function
        The statistic.
     x : array_like or pandas DataFrame or tuple
        The data.
     theta_hat : float
        Observed statistic.
     alpha : float, optional
        Number controlling the size of the interval. That is, this
        function will return a 100(1-2*`alpha`)% confidence
        interval. Defaults to 0.05.
     B : int, optional
        Number of bootstrap samples. Defaults to 1000.
     return_confidence_points : boolean, optional
        If True, returns the estimated confidence points have the
        desired coverage. Defaults to False.
     num_threads : int, optional
        Number of threads to use for multicore processing. Defaults to
        1, meaning all calculations will be done in a single
        thread. Set to -1 to use all available cores.

    Returns
    -------
     ci_low, ci_high : float
        Lower and upper bounds on a 100(1-2*`alpha`)% confidence
        interval on theta.
     lmbda_low, lmbda_high : float
        The estimated confidence points having the desired
        coverage. For example, if the interval has the nominal
        coverage, then lmbda_low would be `alpha` and lmbda_high would
        be 1 - `alpha`.

    Notes
    -----
    While we can in principle calibrate any type of confidence
    interval, in most instances that results in a "double
    bootstrap". For this reason, only ABC intervals are currently
    supported. Moreover, from my limited experience calibration seems
    pretty finicky. I would consider this function to be illustrative
    but experimental. See [ET93, S18] for details.

    We compute the observed coverage for a range of points around the
    nominal `alpha` and 1-`alpha`. Then we fit a smoother (loess) to
    these data and invert to find the confidence point, lmbda, having
    the desired coverage.

    """

    def logit(p):
        return np.log(p / (1 - p))

    def inv_logit(p):
        return 1.0 / (1.0 + np.exp(-p))

    logit_alpha = logit(alpha)
    logit_lmbdas = np.linspace(logit_alpha - 4, -1.0, num=25)
    logit_lmbdas = np.concatenate((logit_lmbdas, np.flip(-logit_lmbdas)))
    lmbdas = inv_logit(logit_lmbdas)

    def _calc_p_hat(dist, stat, batch_size, lmbdas, theta_hat, seed):
        if seed is not None:
            np.random.seed(seed)

        p_hat = np.zeros((len(lmbdas),))
        for i in range(batch_size):
            x_star = dist.sample()
            theta_lambda = abcnon_interval(x_star, stat, alpha=lmbdas)
            for j, theta_lambda_b in enumerate(theta_lambda):
                if theta_hat <= theta_lambda_b:
                    p_hat[j] += 1

        return p_hat

    if num_threads == 1:
        p_hat = _calc_p_hat(dist, stat, B, lmbdas, theta_hat, None)
    else:
        if num_threads == -1:
            num_threads = mp.cpu_count()

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
                _calc_p_hat,
                dist,
                stat,
                batch_sizes[i],
                lmbdas,
                theta_hat,
                seed,
            )
            results.append(r)

        p_hat = np.zeros((len(lmbdas),))
        for res in results:
            p_hat += res.get()

        pool.close()
        pool.join()

    p_hat /= B

    def g(lmbda):
        zz = loess(logit(lmbda), logit_lmbdas, logit(p_hat), 0.3)
        return inv_logit(zz)

    lmbda_low = optimize.root_scalar(
        lambda x: g(x) - alpha, method="bisect", bracket=(1e-6, 0.5)
    ).root

    lmbda_high = optimize.root_scalar(
        lambda x: g(x) - (1 - alpha), method="bisect", bracket=(0.5, 1 - 1e-6)
    ).root

    ci_low, ci_high = abcnon_interval(x, stat, alpha=[lmbda_low, lmbda_high])

    if return_confidence_points:
        return ci_low, ci_high, lmbda_low, lmbda_high
    else:
        return ci_low, ci_high
