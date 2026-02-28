# pyre-unsafe
"""Empirical distribution classes for bootstrap methods."""

from typing import Literal, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from bootstrap_stat._utils import ArrayLike, Statistic


class EmpiricalDistribution:
    r"""Empirical Distribution

    The Empirical Distribution puts probability 1/n on each of n
    observations.


    Parameters
    ----------
     data : array_like or pandas DataFrame
        The data.

    """

    data: ArrayLike
    n: int
    is_multi_sample: bool

    def __init__(self, data: ArrayLike) -> None:
        """Empirical Distribution

        Parameters
        ----------
         data : array_like or pandas DataFrame
            The data.

        """
        self.data = data
        self.n = len(data)
        self.is_multi_sample = False

    @overload
    def sample(
        self,
        size: int | None = None,
        return_indices: Literal[False] = False,
        reset_index: bool = True,
    ) -> npt.NDArray[np.float64] | pd.DataFrame: ...

    @overload
    def sample(
        self,
        size: int | None,
        return_indices: Literal[True],
        reset_index: bool = True,
    ) -> tuple[npt.NDArray[np.float64] | pd.DataFrame, npt.NDArray[np.intp]]: ...

    def sample(
        self,
        size: int | None = None,
        return_indices: bool = False,
        reset_index: bool = True,
    ) -> (
        npt.NDArray[np.float64]
        | pd.DataFrame
        | tuple[npt.NDArray[np.float64] | pd.DataFrame, npt.NDArray[np.intp]]
    ):
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

    def calculate_parameter(self, t: Statistic) -> float:
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

    dists: list[EmpiricalDistribution]
    n: list[int]  # type: ignore[assignment]

    def __init__(self, datasets: tuple[ArrayLike, ...]) -> None:
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

    def sample(  # type: ignore[override]
        self, size: tuple[int, ...] | list[int] | None = None
    ) -> tuple[npt.NDArray[np.float64] | pd.DataFrame, ...]:
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

    def calculate_parameter(self, t: Statistic) -> float:
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
