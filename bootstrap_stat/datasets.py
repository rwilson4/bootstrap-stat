# pyre-unsafe
"""Dataset loading functions for bootstrap examples."""

import os
from typing import Literal

import pandas as pd


def mouse_data(dataset: Literal["control", "treatment"]) -> list[int]:
    """Mouse data from [ET93], Table 2.2.

    Survival times in days of mice in a treatment/control experiment.

    Parameters
    ----------
     dataset : ["control", "treatment"]
        Which group to return.

    Returns
    -------
     data : list of int
        Survival times in days.

    """
    treatment = [94, 197, 16, 38, 99, 141, 23]
    control = [52, 104, 146, 10, 51, 30, 40, 27, 46]

    if dataset == "control":
        return control
    elif dataset == "treatment":
        return treatment
    else:
        raise ValueError("Please specify either 'control' or 'treatment'")


def law_data(full: bool = False) -> pd.DataFrame:
    """Law School Data

    Collection of N = 82 American law schools participating in a large
    study of admissions practices. Two measurements were made on the
    entering classes of each school in 1973: LSAT, the average score
    for the class on a national law test, and GPA, the average
    undergraduate grade-point average for the class.

    Additionally, a sample of n = 15 schools were taken.

    Data from Table 3.2 in An Introduction to the Bootstrap by Bradley
    Efron and Robert J. Tibshirani.

    Parameters
    ----------
     full : boolean, optional
        If True, return the full (N = 82) dataset. If False (default),
        return the sampled (n = 15) dataset.

    Returns
    -------
     df : pandas DataFrame
        The dataset.

    """
    fn = os.path.join(os.path.dirname(__file__), "data", "law_school.csv")
    df = pd.read_csv(fn)
    df.set_index("school", inplace=True)

    sample = [6, 13, 79, 35, 70, 52, 50, 15, 47, 31, 4, 82, 45, 36, 53]
    sample = [s - 1 for s in sample]

    if not full:
        df = df.iloc[sample]
        df.reset_index(inplace=True, drop=True)

    return df


def rainfall_data() -> pd.DataFrame:
    """Rainfall data from [ET93], Table 4.2.

    Yearly rainfall in inches in Nevada City, California, 1873 through
    1978. An example of time series data.

    Returns
    -------
     df : pandas DataFrame
        DataFrame indexed by ``year`` with column ``rainfall``.

    """
    fn = os.path.join(os.path.dirname(__file__), "data", "rainfall.csv")
    df = pd.read_csv(fn)
    df.set_index("year", inplace=True)
    return df


def spatial_test_data(
    test: Literal["A", "B", "both"] = "both",
) -> pd.DataFrame | pd.Series:
    """Spatial Test Data.

    n = 26 children have each taken two tests of spatial ability,
    called A and B. Table 14.1 in An Introduction to the Bootstrap by
    Bradley Efron and Robert J. Tibshirani.

    Parameters
    ----------
     test : ["A", "B", "both"], optional
        Which test results to return. Defaults to "both".

    Returns
    -------
     df : pandas DataFrame or Series
        The data.

    """
    fn = os.path.join(os.path.dirname(__file__), "data", "spatial_test.csv")
    df = pd.read_csv(fn)
    df.set_index("Child", inplace=True)

    if test == "both":
        return df
    elif test in ["A", "B"]:
        return df[test]
    else:
        raise ValueError("Invalid test")


def hormone_data() -> pd.DataFrame:
    """Hormone data from [ET93], Table 9.1.

    Amount in milligrams of anti-inflammatory hormone remaining in 27
    devices after a certain number of hours of wear. Devices were
    sampled from three manufacturing lots (A, B, C). Lot C appears to
    have greater remaining hormone but was worn the fewest hours; a
    regression analysis clarifies the situation.

    Returns
    -------
     df : pandas DataFrame
        Observations with columns for hours worn, manufacturing lot,
        and remaining hormone level.

    """
    fn = os.path.join(os.path.dirname(__file__), "data", "hormone_data.csv")
    df = pd.read_csv(fn)
    return df


def patch_data() -> pd.DataFrame:
    """Patch data from [ET93], Table 10.1.

    Eight subjects wore medical patches designed to increase blood
    levels of a natural hormone. Each subject wore three patches: a
    placebo, an "old" patch from an established plant, and a "new"
    patch from a newly opened plant. Derived columns: ``z`` =
    oldpatch - placebo, ``y`` = newpatch - oldpatch. The purpose of
    the experiment was to show equivalence between the two plants.
    Chapter 25 of [ET93] has an extended analysis.

    Returns
    -------
     df : pandas DataFrame
        DataFrame indexed by ``subject`` with columns ``placebo``,
        ``oldpatch``, ``newpatch``, ``z``, and ``y``.

    """
    fn = os.path.join(os.path.dirname(__file__), "data", "patch.csv")
    df = pd.read_csv(fn)
    df.set_index("subject", inplace=True)
    df["z"] = df["oldpatch"] - df["placebo"]
    df["y"] = df["newpatch"] - df["oldpatch"]
    return df
