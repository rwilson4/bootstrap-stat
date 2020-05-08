import warnings
import multiprocessing as mp

import numpy as np
import scipy.stats as ss
import scipy.optimize as optimize
import pandas as pd
from pathos.multiprocessing import ProcessPool as Pool

"""Methods relating to the Bootstrap.

Estimates of standard errors, bias, confidence intervals, prediction
errors, and more!

References
----------
[ET93]:  Bradley Efron and Robert J. Tibshirani, "An Introduction to the
         Bootstrap". Chapman & Hall, 1993.
[Koh95]: Ron Kohavi, "A Study of Cross-Validation and Bootstrap for
         Accuracy Estimation and Model Selection".  International
         Joint Conference on Artificial Intelligence, 1995.
[ET97]:  Bradley Efron and Robert Tibshirani. "Improvements on
         Cross-Validation: The .632+ Bootstrap Method". Journal of the
         American Statistical Association, Vol. 92, No. 438.  June
         1997, pp. 548--560.
[Arl10]: Sylvain Arlot, "A Survey of Cross-Validation Procedures for
         Model Selection". Statistics Surveys, Vol. 4, 2010.

"""


class EmpiricalDistribution:
    r"""Empirical Distribution

    The Empirical Distribution puts probability 1/n on each of n
    observations.


    Parameters
    ----------
     data : array_like or pandas DataFrame
        The data.

    """

    def __init__(self, data):
        """Empirical Distribution

        Parameters
        ----------
         data : array_like or pandas DataFrame
            The data.

        """
        self.data = data
        self.n = len(data)
        self.is_multi_sample = False

    def sample(self, size=None, return_indices=False, reset_index=True):
        """Sample from the empirical distribution

        Parameters
        ----------
         size : int or tuple of ints, optional
            Output shape. If None (default), samples the same number
            of points as the original dataset.
         return_indices : boolean, optional
            If True, return the indices of the data points
            sampled. Defaults to False.
         reset_index : boolean, optional
            If True (default), reset the index. Applies only to data
            frames. This is usually what we would want to do, except
            for debugging perhaps.

        Returns
        -------
         samples : ndarray or pandas DataFrame
            IID samples from the empirical distribution.
         ind : ndarray
            Indices of samples chosen. Only returned if
            `return_indices` is True.

        """
        if size is None:
            s = self.n
        else:
            s = size

        if return_indices:
            ind = np.random.choice(range(self.n), size=s, replace=True)
            try:
                samples = self.data[ind]
            except KeyError:
                samples = self.data.iloc[ind]
                if reset_index:
                    samples.reset_index(drop=True, inplace=True)
            except TypeError:
                d = np.array(self.data)
                samples = d[ind]
            return samples, ind
        else:
            try:
                samples = np.random.choice(self.data, size=s, replace=True)
            except ValueError:
                samples = self.data.sample(s, replace=True)
                if reset_index:
                    samples.reset_index(drop=True, inplace=True)
            return samples

    def calculate_parameter(self, t):
        """Calculate a parameter of the distribution.

        Parameters
        ----------
         t : function
            Function to be applied to dataset. If using an n-Sample
            Distribution, t should take as input a tuple of data sets
            of the appropriate size.

        Returns
        -------
         tF : float
            Parameter of distribution.

        """
        return t(self.data)


class MultiSampleEmpiricalDistribution(EmpiricalDistribution):
    r"""Multi-Sample Empirical Distribution

    Parameters
    ----------
     datasets : tuple of arrays or pandas DataFrames.
        Observed data sets.

    Notes
    -----
    Suppose we observe

    .. math::

       x_i \sim F, i=1,...,m

       y_j \sim G, j=1,...,n

    Then :math:`P = (\hat{F}, \hat{G})` is the probabilistic mechanism
    consisting of the two empirical distributions :math:`F` and
    :math:`G`. Sampling from :math:`P` amounts to sampling :math:`m`
    points IID from :math:`\hat{F}`, and :math:`n` points IID from
    :math:`\hat{G}`. This is called the Two Sample Empirical
    Distribution, and comes up a lot in A/B testing. The
    generalization to more than two samples is obvious.

    Because of some of the implementation details of the bootstrap
    methods, we use tuples for everything. So when initializing, be
    sure to wrap the different datasets in a tuple. Samples will
    themselves be tuples, etc. See the function documentation below
    for details and examples.

    This is a relatively thin wrapper around the regular
    EmpiricalDistribution: we just create a distinct
    EmpiricalDistribution for each dataset, and use that for sampling.

    Examples
    --------
    >>> data = [1, 2, 3]
    >>> dist = EmpiricalDistribution(data)
    >>> dist.sample()
    [1, 2, 1]
    >>> data_a = [1, 2, 3]
    >>> data_b = [4, 5, 6]
    >>> data = (data_a, data_b)  # Note tuple
    >>> dist = MultiSampleEmpiricalDistribution(data)
    >>> a, b = dist.sample()  # Can de-tuple directly
    >>> a
    [2, 2, 3]
    >>> b
    [4, 6, 4]
    >>> ab = dist.sample()  # Or indirectly, which is often more useful
    >>> ab
    [array([2, 2, 3]), array([4, 6, 4])]

    """

    def __init__(self, datasets):
        """Constructor.

        Parameters
        ----------
         datasets : tuple of arrays or pandas DataFrames.
            Observed data sets.

        Examples
        --------
        >>> data = [1, 2, 3]
        >>> dist = EmpiricalDistribution(data)
        >>> dist.sample()
        [1, 2, 1]
        >>> data_a = [1, 2, 3]
        >>> data_b = [4, 5, 6]
        >>> data = (data_a, data_b)  # Note tuple
        >>> dist = MultiSampleEmpiricalDistribution(data)
        >>> a, b = dist.sample()  # Can de-tuple directly
        >>> a
        [2, 2, 3]
        >>> b
        [4, 6, 4]
        >>> ab = dist.sample()  # Or indirectly, which is often more useful
        >>> ab
        [array([2, 2, 3]), array([4, 6, 4])]

        """
        self.data = datasets
        self.dists = [EmpiricalDistribution(d) for d in datasets]
        self.n = [len(d) for d in datasets]
        self.is_multi_sample = True

    def sample(self, size=None):
        """Sample from the empirical distribution

        Parameters
        ----------
         size : tuple of ints, optional
            Number of samples to be drawn from each
            EmpiricalDistribution. If None (default), samples the same
            numbers of points as the original datasets.

        Returns
        -------
         samples : tuple of ndarray or pandas DataFrame
            IID samples from the empirical distributions.

        """
        if size is None:
            s = self.n
        else:
            s = size

        samples = [d.sample(size=si) for d, si in zip(self.dists, s)]
        return (*samples,)

    def calculate_parameter(self, t):
        r"""Calculate a parameter of the distribution.

        Parameters
        ----------
         t : function
            Function to be applied to dataset. Should take as input a
            tuple of data sets of the appropriate size.

        Returns
        -------
         tF : float
            Parameter of distribution.

        Examples
        --------
        Suppose we are in the Two-Sample case and have two empirical
        distributions, :math:`\hat{F}` and :math:`\hat{G}`, and we
        want to calculate the difference in means of these
        distributions. We might do something like:

        >>> data_a = [1, 2, 3]
        >>> data_b = [4, 5, 6]
        >>> data = (data_a, data_b)  # Note tuple
        >>> dist = MultiSampleEmpiricalDistribution(data)
        >>> def parameter(ab):
        ...     a, b = ab  # Note de-tupling
        ...     return np.mean(b) - np.mean(a)
        >>> dist.calculate_parameter(parameter)
        3.0
        """
        return t(self.data)


def jackknife_standard_error(
    x, stat, return_samples=False, jv=None, num_threads=1
):
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
    if type(x) is tuple:
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
    dist,
    stat,
    robustness=None,
    B=200,
    size=None,
    jackknife_after_bootstrap=False,
    return_samples=False,
    theta_star=None,
    num_threads=1,
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
        p = np.percentile(
            theta_star, [100 * robustness, 100 * (1 - robustness)]
        )
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
    x,
    stat,
    eps=1e-3,
    influence_components=None,
    return_influence_components=False,
    num_threads=1,
):
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


def t_interval(
    dist,
    stat,
    theta_hat,
    stabilize_variance=False,
    se_hat=None,
    fast_std_err=None,
    alpha=0.05,
    Binner=25,
    Bouter=1000,
    Bvar=100,
    size=None,
    empirical_distribution=EmpiricalDistribution,
    return_samples=False,
    theta_star=None,
    se_star=None,
    z_star=None,
    num_threads=1,
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
    dist,
    stat,
    alpha=0.05,
    B=1000,
    size=None,
    return_samples=False,
    theta_star=None,
    num_threads=1,
):
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
    dist,
    stat,
    x,
    alpha=0.05,
    B=1000,
    size=None,
    return_samples=False,
    theta_star=None,
    theta_hat=None,
    jv=None,
    num_threads=1,
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
    x,
    stat,
    alpha=0.05,
    eps=0.001,
    influence_components=None,
    second_derivatives=None,
    return_influence_components=False,
    num_threads=1,
):
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

    sum_inf_squared = np.sum(influence_components ** 2)
    sigma_hat = np.sqrt(sum_inf_squared) / n
    a_hat = np.sum(influence_components ** 3) / (6 * sum_inf_squared ** 1.5)

    delta_hat = influence_components / (n2 * sigma_hat)
    c_q = (
        stat(x, p0 + eps * delta_hat) - 2 * t0 + stat(x, p0 - eps * delta_hat)
    ) / (2 * sigma_hat * eps * eps)

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
    dist,
    stat,
    x,
    theta_hat,
    alpha=0.05,
    B=1000,
    return_confidence_points=False,
    num_threads=1,
):
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

        seeds = np.random.randint(0, 2 ** 32 - 1, num_threads)
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


def jackknife_values(x, stat, sample=None, num_threads=1):
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

    if sample is not None and type(x) is tuple:
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


def bias(
    dist, stat, t, B=200, return_samples=False, theta_star=None, num_threads=1
):
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
    bias = np.mean(theta_star) - tF_hat

    if return_samples:
        return bias, theta_star
    else:
        return bias


def better_bootstrap_bias(x, stat, B=400, return_samples=False, num_threads=1):
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

        seeds = np.random.randint(0, 2 ** 32 - 1, num_threads)
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

    bias = np.mean(theta_star) - stat(x, sum_p / len(theta_star))

    if return_samples:
        return bias, theta_star
    else:
        return bias


def jackknife_bias(x, stat, return_samples=False, jv=None, num_threads=1):
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
    bias = (n - 1) * (np.mean(jv) - stat(x))
    if return_samples:
        return bias, jv
    else:
        return bias


def bias_corrected(
    x,
    stat,
    method="better_bootstrap_bias",
    dist=None,
    t=None,
    B=None,
    return_samples=False,
    theta_star=None,
    jv=None,
    num_threads=1,
):
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


def bootstrap_asl(
    dist,
    stat,
    x,
    B=1000,
    size=None,
    return_samples=False,
    theta_star=None,
    theta_hat=None,
    two_sided=False,
    num_threads=1,
):
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
    dist,
    stat,
    x,
    theta_0=0,
    B=1000,
    size=None,
    return_samples=False,
    theta_star=None,
    theta_hat=None,
    two_sided=False,
    num_threads=1,
):
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
    dist,
    stat,
    x,
    theta_0=0,
    B=1000,
    size=None,
    return_samples=False,
    theta_star=None,
    theta_hat=None,
    jv=None,
    two_sided=False,
    num_threads=1,
):
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
    alt_dist,
    null_dist,
    stat,
    asl=bootstrap_asl,
    alpha=0.05,
    size=None,
    P=100,
    **kwargs,
):
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


def prediction_error_optimism(
    dist,
    data,
    train,
    predict,
    error,
    B=200,
    apparent_error=None,
    num_threads=1,
):
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

        err_orig = np.mean(
            error(pred_orig, data)
        )  # Error for original dataset
        err_boot = np.mean(error(pred_boot, x))  # Error for bootstrap dataset

        optimism = err_orig - err_boot
        return optimism

    optimism = bootstrap_samples(dist, stat, B, num_threads=num_threads)
    pe = apparent_error + np.mean(optimism)
    return pe


def prediction_error_632(
    dist,
    data,
    train,
    predict,
    error,
    B=200,
    apparent_error=None,
    use_632_plus=False,
    gamma=None,
    no_inf_err_rate=None,
    num_threads=1,
):
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

        seeds = np.random.randint(0, 2 ** 32 - 1, num_threads)
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
    dist,
    x,
    mean=None,
    std=None,
    B=1000,
    alpha=0.05,
    t_star=None,
    return_t_star=False,
    num_threads=-1,
):
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

            seeds = np.random.randint(0, 2 ** 32 - 1, num_threads)
            for i, seed in enumerate(seeds):
                r = pool.apipe(
                    _bootstrap_sim, dist, mean, std, batch_sizes[i], seed
                )
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


def multithreaded_bootstrap_samples(
    dist, stat, B, size=None, jackknife=False, num_threads=-1
):
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

    seeds = np.random.randint(0, 2 ** 32 - 1, num_threads)
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
    dist, stat, B, size=None, jackknife=False, num_threads=1
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


def _bca_acceleration(jv):
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

    if type(jv) is tuple:
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
        den = 6 * (den ** 1.5)
    else:
        theta_dot = np.mean(jv)
        U = theta_dot - jv
        U2 = U * U
        num = np.sum(U2 * U)
        den = 6 * ((np.sum(U2)) ** 1.5)

    a_hat = num / den
    return a_hat


def _adjust_percentiles(alpha, a_hat, z0_hat):
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
    alpha1 = ss.norm.cdf(
        z0_hat + (z0_hat + z_alpha) / (1 - a_hat * (z0_hat + z_alpha))
    )
    alpha2 = ss.norm.cdf(
        z0_hat
        + (z0_hat + z_one_m_alpha) / (1 - a_hat * (z0_hat + z_one_m_alpha))
    )

    return alpha1, alpha2


def _percentile(z, p, full_sort=True):
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
            warnings.warn(
                "Index outside of bounds. Try more bootstrap samples."
            )
            if k <= 0:
                k = 0
            elif k >= B:
                k = B

        if full_sort:
            percentiles[i] = sorted_z[k - 1]
        else:
            percentiles[i] = np.partition(z, k - 1)[k - 1]

    return percentiles


def loess(z0, z, y, alpha, sided="both"):
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


def _resampling_vector(n):
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


def _influence_components(x, stat, order=1, eps=1e-3, num_threads=1):
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
