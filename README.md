# bootstrap-stat

![](https://github.com/rwilson4/bootstrap-stat/workflows/build/badge.svg)

Methods relating to the Bootstrap.

Estimates of standard errors, bias, confidence intervals, prediction
errors, and more!

## Getting Started
Bootstrap-Stat is hosted on PyPI. Install as you would any other
library, e.g.: `poetry add bootstrap-stat`.

Documentation is available at
[Convex Analytics](https://www.convexanalytics.com/bootstrap-stat/index.html)
but I also recommend reviewing the rest of this README.

## Brief Overview

Quoting [ET93], "The bootstrap is a data-based simulation method for
statistical inference...The use of the term bootstrap derives from the
phrase *to pull oneself up by one's bootstrap*..."

The bootstrap is a collection of methodologies for estimating errors
and uncertainties associated with estimates and predictions. For
example, it can be used to compute the bias and standard error of a
particular estimator. It can be used to estimate the prediction error
of a particular ML model (as a competitor to Cross Validation). And it
can be used to compute confidence intervals. Most of the techniques
described in [ET93] have now been implemented. (Notable exceptions
include importance resampling, likelihood-based methods, and better
support for the parametric bootstrap.)

### Basic Terminology

To apply bootstrap techniques, it is important to understand the
following terminology.

* Distribution: an entity assigning probabilities to observations or
  sets of observations.
* Population: the complete set of data, typically only partially observed.
* Sample: the observed data, assumed to be drawn with replacement from
  the population. In real life, samples are usually drawn *without*
  replacement from the population, but provided the sample is a small
  fraction of the population, this is a negligible concern.
* Empirical Distribution: the distribution assigning probability 1/n
  to each observation in a sample of size n.
* Parameter: some function of a distribution. See `statistic`.
* Statistic: some function of a collection of observations. We will
  assume all statistics are real-valued scalars, such as mean, median,
  or variance. Parameters and statistics are similar: it makes sense
  to talk about the mean of a distribution, which is a parameter of
  that distribution, and it also makes sense to talk about the mean of
  a collection of numbers, which is a statistic. For this reason, it
  is important to keep it straight whether we are talking about a
  parameter or a statistic! For example, it makes sense to talk about
  the bias of a statistic, but it does not make sense to talk about
  the bias of a parameter.
* Plug-in estimate of a parameter: an estimate of a parameter
  calculated by "plugging-in" the Empirical Distribution. For example,
  to estimate the mean of an unobserved distribution, simply calculate
  the mean of the (observed) Empirical Distribution. The plug-in
  estimate is a *statistic*.
* Bootstrap sample: a sample drawn with replacement from the Empirical
  Distribution, having size equal to the size of the original dataset.
* Standard Error: the square root of the variance of a statistic,
  typically used to quantify accuracy.
* Bias: the difference between the expected value of a statistic and
  the parameter it purports to estimate.
* Confidence Interval: a range of plausible values of a parameter
  consistent with the data.

## Examples
This library includes some datasets that can be used for trying out
methods. The test cases themselves (in `tests/`) contain many
practical examples.

```
>>> import numpy as np
>>> from bootstrap_stat import bootstrap_stat as bp
>>> from bootstrap_stat import datasets as d
>>>
>>> df = d.law_data()
>>> print(df)
    LSAT   GPA
0    576  3.39
1    635  3.30
2    558  2.81
3    578  3.03
4    666  3.44
5    580  3.07
6    555  3.00
7    661  3.43
8    651  3.36
9    605  3.13
10   653  3.12
11   575  2.74
12   545  2.76
13   572  2.88
14   594  2.96
```

The law data are a collection of N = 82 American law schools
participating in a large study of admissions practices. Two
measurements were made on the entering classes of each school in 1973:
LSAT, the average score for the class on a national law test, and GPA,
the average undergraduate grade-point average for the class. Both the
full data set, and a sample are available. The above is a sample of 15
schools. The law data are taken from [EF93].

Suppose we are interested in the correlation between LSAT and
GPA. Numpy can be used to compute the observed correlation for the
sample (a *statistic*), but we hope to draw inferences about the
population (all 82 schools) correlation coefficient (a *parameter*)
based just on the sample. In this case, the entire population is
available, and we could just compute the parameter directly. In most
cases, the entire population is not available.

To use the bootrap method, we need to specify the statistic as well as
the dataset. Specifically, we need to be able to sample with
replacement from the Empirical Distribution. `bootstrap_stat` has a class
facilitating just that.

```
>>> dist = bp.EmpiricalDistribution(df)
>>> dist.sample(reset_index=False)
    LSAT   GPA
14   594  2.96
3    578  3.03
0    576  3.39
6    555  3.00
10   653  3.12
12   545  2.76
6    555  3.00
0    576  3.39
10   653  3.12
8    651  3.36
3    578  3.03
4    666  3.44
13   572  2.88
5    580  3.07
8    651  3.36
```

Generating the Empirical Distribution is as simple as feeding either
an array, pandas Series, or pandas DataFrame into the
constructor. Under the hood, the bootstrap methods make frequent use
of the `sample` method, which samples with replacement from the
original dataset. Such samples are called *bootstrap samples*. Notice
in the example above how school 0 makes multiple appearances in the
bootstrap sample. Since the sampling is random, if you run the above
you will likely get different results than in this example. (In some
of the more exotic use cases, we need to reset the index for technical
reasons relating to pandas indexing, so the default behavior is to
reset, hence the `reset_index=False` in this example.)

Next we need to implement the statistic, which will be applied to
bootstrap samples.

```
>>> def statistic(df):
...     return np.corrcoef(df["LSAT"], df["GPA"])[0, 1]
...
>>> obs_correlation = statistic(df)  # Observed correlation coefficient
>>> print(obs_correlation)
0.776374491289407
```

Notice how we can apply the statistic to the original dataset to
calculate the observed value. The statistic should take as input
either an array or a pandas Series or DataFrame, whatever was used to
generate the Empirical Distribution. It should output a single
number. Other than that, it can be anything: a simple calculation like
a mean, a parameter from a linear regression model, or even a
prediction from a neural network.

Now we can compute the standard error, which is a way of quantifying
the variability of a statistic:

```
>>> se = bp.standard_error(dist, statistic)
>>> print(se)
0.13826565276176475
```

Since the bootstrap involves random sampling, you will likely get a
slightly different answer than above, but it should be within 1% or
so.

Or we can compute a confidence interval, a range of plausible values
for the parameter consistent with the data.

```
>>> ci_low, ci_high = bp.bcanon_interval(dist, statistic, df)
>>> print(ci_low, ci_high)
0.44968698948896413 0.9230026418265834
```

These represent lower and upper bounds on a 90% confidence interval,
the default behavior of `bcanon_interval`. We can do a 95% confidence
interval by specifying `alpha`:

```
>>> ci_low, ci_high = bp.bcanon_interval(dist, statistic, df, alpha=0.025)
>>> print(ci_low, ci_high)
0.3120414479586675 0.9425059323691073
```

In general, `bcanon_interval` returns a 100(1-2`alpha`)% confidence
interval. The `bcanon` terminology is a nod to the S implementation
discussed in [ETF93]. (BCa is an algorithm for Bias-Corrected and
Accelerated confidence intervals, and the function is NONparametric.)

Basic multicore functionality is implemented, allowing parallel
calculation of bootstrap samples. Simply specify the `num_threads`
argument in applicable functions. See the function documentation for
details.


## Running the test cases

```
$ poetry shell
$ python -m pytest
```

## Documentation
Documentation is built using Sphinx and is hosted at
[Convex Analytics](https://www.convexanalytics.com/bootstrap-stat/index.html).

To update the docs (e.g. after updating the code), just change
directory to `docs` and type `make html`. You'll need to be in a
poetry shell.

## Architecture
We use Poetry to manage dependencies, pytest as our test runner, black
for code formatting, and sphinx for generating documentation.

Basic multicore functionality is implemented, using the
[pathos](https://pathos.readthedocs.io/en/latest/) version of
multiprocessing. We chose this version over the official python
multiprocessing library since pathos uses `dill` instead of `pickle`
to manage shared memory, and `pickle` cannot be used with locally
defined functions. For users of this library, hopefully that
implementation detail is irrelevant.

## Licensing
Bootstrap-Stat is licensed under the Apache License, Version 2.0. See
`LICENSE.txt` for the full license text.

## References

[ET93] Bradley Efron and Robert J. Tibshirani, "An Introduction to the
       Bootstrap". Chapman & Hall, 1993.
