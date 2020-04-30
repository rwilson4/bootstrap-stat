import os

import pandas as pd


def mouse_data(dataset):
    """Mouse data

    Data from An Introduction to the Bootstrap by Bradley Efron and
    Robert J. Tibshirani.

    """
    treatment = [94, 197, 16, 38, 99, 141, 23]
    control = [52, 104, 146, 10, 51, 30, 40, 27, 46]

    if dataset == "control":
        return control
    elif dataset == "treatment":
        return treatment
    else:
        raise ValueError("Please specify either 'control' or 'treatment'")


def law_data(full=False):
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


def rainfall_data():
    """Rainfall data.

    The yearly rainfall, in inches, in Nevada City, California, 1873
    through 1978. An example of time series data.

    Table 4.2 in An Introduction to the Bootstrap by Bradley Efron and
    Robert J. Tibshirani.

    """
    fn = os.path.join(os.path.dirname(__file__), "data", "rainfall.csv")
    df = pd.read_csv(fn)
    df.set_index("year", inplace=True)
    return df


def spatial_test_data(test="both"):
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


def hormone_data():
    """The hormone data.

    Taken from Table 9.1 of [ET93].

    Amount in milligrams of anti-inflammatory hormone remaining in 27
    devices, after a certain number of hours of wear. The devices were
    sampled from 3 different manufacturing lots, called A, B, and
    C. Lot C looks like it had greater amounts of remaining hormone,
    but it also was worn the least number of hours. A regression
    analysis clarifies the situation.

    """
    fn = os.path.join(os.path.dirname(__file__), "data", "hormone_data.csv")
    df = pd.read_csv(fn)
    return df


def patch_data():
    """The patch data.

    Taken from Table 10.1 of [ET93].

    Eight subjects wore medical patches designed to increase the blood
    levels of a certain natural hormone. Each subject had his blood
    levels of the hormone measured after wearing three different
    patches: a placebo patch, which had no medicine in it, an "old"
    patch which was from a lot manufactured at an old plant, and a
    "new" patch, which was from a lot manufactured at a newly opened
    plant. For each subject, z = oldpatch - placebo measurement and y
    = newpatch - oldpatch measurement. The purpose of the experiment
    was to show that the new plant was producing patches equivalent to
    those from the old plant. Chapter 25 of [ET93] has an extended
    analysis of this data set.

    """
    fn = os.path.join(os.path.dirname(__file__), "data", "patch.csv")
    df = pd.read_csv(fn)
    df.set_index("subject", inplace=True)
    df["z"] = df["oldpatch"] - df["placebo"]
    df["y"] = df["newpatch"] - df["oldpatch"]
    return df
