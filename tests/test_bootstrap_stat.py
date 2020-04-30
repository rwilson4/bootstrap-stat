import time

import pytest
import numpy as np
import scipy.stats as ss
import scipy.optimize as optimize
import statsmodels.formula.api as smf
import pandas as pd

from .context import bootstrap_stat as bp
from .context import datasets


class TestMisc:
    def test_percentile(self):
        z = np.array(range(1, 1001))
        alpha = 0.05
        expected_low = 50
        expected_high = 950

        actual = bp._percentile(z, [alpha, 1 - alpha])
        assert actual[0] == expected_low
        assert actual[1] == expected_high

    def test_percentile_uneven(self):
        z = np.array(range(1, 1000))
        alpha = 0.05
        expected_low = 50
        expected_high = 950

        actual = bp._percentile(z, [alpha, 1 - alpha])
        assert actual[0] == expected_low
        assert actual[1] == expected_high

    def test_percentile_partial_sort(self):
        z = np.array(range(1, 1000))
        alpha = 0.05
        expected_low = 50
        expected_high = 950

        actual = bp._percentile(z, [alpha, 1 - alpha], full_sort=False)
        assert actual[0] == expected_low
        assert actual[1] == expected_high

    def test_adjust_percentiles(self):
        alpha = 0.05
        a_hat = 0.061
        z0_hat = 0.146

        expected_alpha1 = 0.110
        expected_alpha2 = 0.985

        actual_alpha1, actual_alpha2 = bp._adjust_percentiles(
            alpha, a_hat, z0_hat
        )

        assert actual_alpha1 == pytest.approx(expected_alpha1, abs=1e-3)
        assert actual_alpha2 == pytest.approx(expected_alpha2, abs=1e-3)

    def test_jackknife_values_series(self):
        df = datasets.spatial_test_data("A")
        expected = 0.061

        def statistic(x):
            return np.var(x, ddof=0)

        jv = bp.jackknife_values(df, statistic)
        actual = bp._bca_acceleration(jv)
        assert actual == pytest.approx(expected, abs=1e-3)

    def test_jackknife_values_array(self):
        df = datasets.spatial_test_data("A")
        x = np.array(df)
        expected = 0.061

        def statistic(x):
            return np.var(x, ddof=0)

        jv = bp.jackknife_values(x, statistic)
        actual = bp._bca_acceleration(jv)
        assert actual == pytest.approx(expected, abs=1e-3)

    def test_jackknife_values_dataframe(self):
        df = datasets.spatial_test_data("both")
        expected = 0.061

        def statistic(df):
            return np.var(df["A"], ddof=0)

        jv = bp.jackknife_values(df, statistic)
        actual = bp._bca_acceleration(jv)
        assert actual == pytest.approx(expected, abs=1e-3)

    def test_loess(self):
        z = np.linspace(0, 1, num=50)
        y = np.sin(12 * (z + 0.2)) / (z + 0.2)
        np.random.seed(0)
        ye = y + np.random.normal(0, 1, (50,))
        alpha = 0.20
        expected = 0.476

        actual = [bp.loess(z0, z, ye, alpha) for z0 in z]
        actual = np.array(actual)
        actual = np.sqrt(np.mean((actual - y) ** 2))
        assert actual == pytest.approx(expected, abs=1e-3)

    def test_resampling_vector(self):
        n = 8
        expected = [1 / 8, 0, 0, 3 / 8, 1 / 8, 1 / 8, 0, 2 / 8]

        np.random.seed(0)
        actual = bp._resampling_vector(n)
        np.testing.assert_array_equal(actual, expected)

    def test_parametric_bootstrap(self):
        df = datasets.law_data(full=False)
        expected = 0.124

        class EmpiricalGaussian(bp.EmpiricalDistribution):
            def __init__(self, df):
                self.mean = df.mean()
                self.cov = np.cov(df["LSAT"], df["GPA"], ddof=1)
                self.n = len(df)

            def sample(self, size=None):
                if size is None:
                    size = self.n

                samples = np.random.multivariate_normal(
                    self.mean, self.cov, size=size
                )

                df = pd.DataFrame(samples)
                df.columns = ["LSAT", "GPA"]
                return df

        def statistic(df):
            return np.corrcoef(df["LSAT"], df["GPA"])[0, 1]

        dist = EmpiricalGaussian(df)
        np.random.seed(5)
        actual = bp.standard_error(dist, statistic, B=3200)
        assert actual == pytest.approx(expected, abs=0.002)


class TestStandardError:
    def test_standard_error(self):
        df = datasets.law_data(full=False)
        expected = 0.132

        def statistic(df):
            return np.corrcoef(df["LSAT"], df["GPA"])[0, 1]

        dist = bp.EmpiricalDistribution(df)

        np.random.seed(0)
        actual = bp.standard_error(dist, statistic, B=2000)
        assert actual == pytest.approx(expected, abs=0.01)

    def test_standard_error_robust(self):
        df = datasets.law_data(full=False)
        robustness = 0.95
        expected = 0.132

        def statistic(df):
            return np.corrcoef(df["LSAT"], df["GPA"])[0, 1]

        dist = bp.EmpiricalDistribution(df)

        np.random.seed(0)
        actual = bp.standard_error(
            dist, statistic, robustness=robustness, B=2000
        )
        assert actual == pytest.approx(expected, abs=0.01)

    def test_jackknife_after_bootstrap(self):
        x = datasets.mouse_data("treatment")
        expected_se = 24.27
        expected_se_jack = 6.83

        dist = bp.EmpiricalDistribution(x)

        def stat(x):
            return np.mean(x)

        np.random.seed(0)
        actual_se, actual_se_jack = bp.standard_error(
            dist, stat, B=200, jackknife_after_bootstrap=True
        )
        assert actual_se == pytest.approx(expected_se, abs=0.01)
        assert actual_se_jack == pytest.approx(expected_se_jack, abs=0.01)

    def test_infinitesimal_jackknife(self):
        df = datasets.law_data(full=False)
        expected = 0.1243

        def statistic(df, p):
            mean_gpa = np.dot(df["GPA"], p)
            mean_lsat = np.dot(df["LSAT"], p)
            sigma_gpa = np.dot(p, (df["GPA"] - mean_gpa) ** 2)
            sigma_lsat = np.dot(p, (df["LSAT"] - mean_lsat) ** 2)
            corr = (df["GPA"] - mean_gpa) * (df["LSAT"] - mean_lsat)
            corr = np.dot(corr, p) / np.sqrt(sigma_gpa * sigma_lsat)
            return corr

        actual = bp.infinitesimal_jackknife(df, statistic)
        assert actual == pytest.approx(expected, abs=1e-4)


class TestConfidenceIntervals:
    def test_t_interval(self):
        df = datasets.mouse_data("control")
        alpha = 0.05
        expected_low = 35.8251
        expected_high = 116.6049

        def statistic(x):
            return np.mean(x)

        dist = bp.EmpiricalDistribution(df)
        theta_hat = statistic(df)

        np.random.seed(0)
        se_hat = bp.standard_error(dist, statistic, B=2000)

        actual_low, actual_high = bp.t_interval(
            dist,
            statistic,
            theta_hat,
            se_hat=se_hat,
            alpha=alpha,
            Bouter=1000,
            Binner=25,
        )
        assert actual_low == pytest.approx(expected_low, abs=1)
        assert actual_high == pytest.approx(expected_high, abs=1)

    def test_t_interval_fast(self):
        df = datasets.mouse_data("control")
        alpha = 0.05
        expected_low = 35.8251
        expected_high = 116.6049

        def statistic(x):
            return np.mean(x)

        def fast_std_err(x):
            return np.sqrt(np.var(x, ddof=1) / len(x))

        dist = bp.EmpiricalDistribution(df)
        theta_hat = statistic(df)

        np.random.seed(0)
        se_hat = fast_std_err(df)

        actual_low, actual_high = bp.t_interval(
            dist,
            statistic,
            theta_hat,
            se_hat=se_hat,
            fast_std_err=fast_std_err,
            alpha=alpha,
            Bouter=1000,
        )
        assert actual_low == pytest.approx(expected_low, abs=3)
        assert actual_high == pytest.approx(expected_high, abs=5)

    @pytest.mark.slow
    def test_t_interval_robust(self):
        df = datasets.mouse_data("control")
        alpha = 0.05
        expected_low = 35.8251
        expected_high = 116.6049

        def statistic(x):
            return np.mean(x)

        def robust_std_err(x):
            dist = bp.EmpiricalDistribution(x)
            return bp.standard_error(dist, statistic, robustness=0.95, B=1000)

        dist = bp.EmpiricalDistribution(df)
        theta_hat = statistic(df)

        np.random.seed(0)
        se_hat = bp.standard_error(
            dist, statistic, robustness=0.95, B=2000, num_threads=12
        )

        actual_low, actual_high = bp.t_interval(
            dist,
            statistic,
            theta_hat,
            se_hat=se_hat,
            fast_std_err=robust_std_err,
            alpha=alpha,
            Bouter=1000,
            num_threads=12,
        )
        assert actual_low == pytest.approx(expected_low, abs=1)
        assert actual_high == pytest.approx(expected_high, abs=3)

    def test_t_interval_law_data(self):
        df = datasets.law_data(full=False)
        alpha = 0.05
        expected_low = 0.45
        expected_high = 0.93

        def statistic(df):
            theta = np.corrcoef(df["LSAT"], df["GPA"])[0, 1]
            return 0.5 * np.log((1 + theta) / (1 - theta))

        def inverse(phi):
            return (np.exp(2 * phi) - 1) / (np.exp(2 * phi) + 1)

        def fast_std_err(df):
            n = len(df.index)
            return 1 / np.sqrt(n - 3)

        dist = bp.EmpiricalDistribution(df)
        theta_hat = statistic(df)
        se_hat = fast_std_err(df)

        np.random.seed(4)
        actual_low, actual_high = bp.t_interval(
            dist,
            statistic,
            theta_hat,
            se_hat=se_hat,
            fast_std_err=fast_std_err,
            alpha=alpha,
            Bouter=1000,
        )
        actual_low = inverse(actual_low)
        actual_high = inverse(actual_high)
        assert actual_low == pytest.approx(expected_low, abs=0.3)
        assert actual_high == pytest.approx(expected_high, abs=0.03)

    @pytest.mark.slow
    def test_t_interval_law_data_variance_adjusted(self):
        df = datasets.law_data(full=False)
        alpha = 0.05
        expected_low = 0.45
        expected_high = 0.93

        def statistic(df):
            theta = np.corrcoef(df["LSAT"], df["GPA"])[0, 1]
            return theta

        dist = bp.EmpiricalDistribution(df)
        theta_hat = statistic(df)

        np.random.seed(0)
        actual_low, actual_high = bp.t_interval(
            dist,
            statistic,
            theta_hat,
            stabilize_variance=True,
            alpha=alpha,
            Bouter=1000,
            Binner=25,
            Bvar=100,
            num_threads=12,
        )
        assert actual_low == pytest.approx(expected_low, abs=0.05)
        assert actual_high == pytest.approx(expected_high, abs=0.03)

    def test_percentile_interval(self):
        df = datasets.mouse_data("treatment")
        alpha = 0.05
        expected_low = 49.7
        expected_high = 126.7

        def statistic(x):
            return np.mean(x)

        dist = bp.EmpiricalDistribution(df)

        np.random.seed(0)

        actual_low, actual_high = bp.percentile_interval(
            dist, statistic, alpha=alpha, B=1000
        )

        assert actual_low == pytest.approx(expected_low, abs=1)
        assert actual_high == pytest.approx(expected_high, abs=4)

    def test_percentile_interval_return_samples(self):
        df = datasets.mouse_data("treatment")
        alpha = 0.05
        expected_low = 49.7
        expected_high = 126.7

        def statistic(x):
            return np.mean(x)

        dist = bp.EmpiricalDistribution(df)

        np.random.seed(0)
        ci_low, ci_high, theta_star = bp.percentile_interval(
            dist, statistic, alpha=alpha, B=1000, return_samples=True
        )

        actual_low, actual_high = bp.percentile_interval(
            dist, statistic, alpha=alpha, theta_star=theta_star
        )

        assert actual_low == pytest.approx(expected_low, abs=1)
        assert actual_high == pytest.approx(expected_high, abs=4)

    def test_bca(self):
        """Compare confidence intervals.

        This test is intended to reproduce Table 14.2 in [ET93].

        The results are show below. What we see is that the ABC method
        is just as fast as the standard approach (theta_hat +/- 1.645
        standard errors), but much more accurate. It gives similar
        answers to the BCa method, but 10x as fast.

               Method   	CI Low	CI High	Time
            standard    	98.7	244.4	0.030
            percentile  	99.4	234.2	0.225
            BCa         	116.2	258.7	0.241
            ABC         	116.7	260.9	0.028
            bootstrap-t 	110.0	303.6	0.316

        """
        df = datasets.spatial_test_data("A")
        alpha = 0.05

        expected_standard_low = 98.8
        expected_standard_high = 244.2  # I think [ET93] has a typo.
        expected_percentile_low = 100.8
        expected_percentile_high = 233.9
        expected_bca_low = 115.8
        expected_bca_high = 259.6
        expected_abc_low = 116.7
        expected_abc_high = 260.9
        expected_t_low = 112.3
        expected_t_high = 314.8

        print("   Method   \tCI Low\tCI High\tTime")

        def statistic(x):
            return np.var(x, ddof=0)

        def resampling_statistic(x, p):
            mu = np.dot(x, p)
            return np.dot(p, (x - mu) ** 2)

        def fast_std_err(x):
            """Fast calculation for the standard error of variance estimator"""
            xhat = np.mean(x)
            u2 = np.mean([(xi - xhat) ** 2 for xi in x])
            u4 = np.mean([(xi - xhat) ** 4 for xi in x])
            return np.sqrt((u4 - u2 * u2) / len(x))

        theta_hat = statistic(df)
        dist = bp.EmpiricalDistribution(df)

        np.random.seed(6)
        st = time.time()
        se = bp.standard_error(dist, statistic)
        actual_low = theta_hat - 1.645 * se
        actual_high = theta_hat + 1.645 * se
        duration = time.time() - st
        print(
            f"{'standard'.ljust(12)}\t{actual_low:.01f}\t{actual_high:.01f}"
            f"\t{duration:.03f}"
        )
        assert actual_low == pytest.approx(expected_standard_low, abs=0.2)
        assert actual_high == pytest.approx(expected_standard_high, abs=0.2)

        st = time.time()
        actual_low, actual_high, theta_star = bp.percentile_interval(
            dist, statistic, alpha=alpha, B=2000, return_samples=True
        )
        duration = time.time() - st
        print(
            f"{'percentile'.ljust(12)}\t{actual_low:.01f}"
            f"\t{actual_high:.01f}\t{duration:.03f}"
        )
        assert actual_low == pytest.approx(expected_percentile_low, abs=1.5)
        assert actual_high == pytest.approx(expected_percentile_high, abs=0.4)

        actual_low, actual_high = bp.bcanon_interval(
            dist, statistic, df, alpha=alpha, theta_star=theta_star
        )
        duration = time.time() - st
        print(
            f"{'BCa'.ljust(12)}\t{actual_low:.01f}\t{actual_high:.01f}"
            f"\t{duration:.03f}"
        )
        assert actual_low == pytest.approx(expected_bca_low, abs=0.5)
        assert actual_high == pytest.approx(expected_bca_high, abs=1.0)

        st = time.time()
        actual_low, actual_high = bp.abcnon_interval(
            df, resampling_statistic, alpha=alpha
        )
        duration = time.time() - st
        print(
            f"{'ABC'.ljust(12)}\t{actual_low:.01f}"
            f"\t{actual_high:.01f}\t{duration:.03f}"
        )
        assert actual_low == pytest.approx(expected_abc_low, abs=0.1)
        assert actual_high == pytest.approx(expected_abc_high, abs=0.1)

        st = time.time()
        se_hat = fast_std_err(df)
        actual_low, actual_high = bp.t_interval(
            dist,
            statistic,
            theta_hat,
            se_hat=se_hat,
            fast_std_err=fast_std_err,
            alpha=alpha,
            Bouter=1000,
        )
        duration = time.time() - st
        print(
            f"{'bootstrap-t'.ljust(12)}\t{actual_low:.01f}"
            f"\t{actual_high:.01f}\t{duration:.03f}"
        )
        assert actual_low == pytest.approx(expected_t_low, abs=2.5)
        assert actual_high == pytest.approx(expected_t_high, abs=12.0)

    @pytest.mark.slow
    def test_compare_intervals(self):
        """Compare confidence intervals.

        This test is similar to the above test; here we explicitly
        want to verify we can compute the bootstrap stats once and
        recycle them in a variety of interval calculations. The
        results are show below. In addition, unlike the above test we
        calculate the variance-stabilized bootstrap-t interval. Note
        how it gives very similar answers to BCa.

               Method   	CI Low	CI High
            standard    	103.0	240.1
            percentile  	100.6	236.2
            BCa         	115.1	261.6
            bootstrap-t 	111.6	295.8
            var-stab-t  	117.5	263.7

        """
        df = datasets.spatial_test_data("A")
        alpha = 0.05

        expected_standard_low = 103.0
        expected_standard_high = 240.1
        expected_percentile_low = 100.6
        expected_percentile_high = 236.2
        expected_bca_low = 115.1
        expected_bca_high = 261.6
        expected_t_low = 111.6
        expected_t_high = 295.8
        expected_stab_low = 117.5
        expected_stab_high = 263.7

        print("   Method   \tCI Low\tCI High")

        def statistic(x):
            return np.var(x, ddof=0)

        def resampling_statistic(x, p):
            mu = np.dot(x, p)
            return np.dot(p, (x - mu) ** 2)

        def fast_std_err(x):
            """Fast calculation for the standard error of variance estimator"""
            xhat = np.mean(x)
            u2 = np.mean([(xi - xhat) ** 2 for xi in x])
            u4 = np.mean([(xi - xhat) ** 4 for xi in x])
            return np.sqrt((u4 - u2 * u2) / len(x))

        theta_hat = statistic(df)
        dist = bp.EmpiricalDistribution(df)

        np.random.seed(6)
        B = 2000
        statistics = {"theta_star": statistic, "se_star": fast_std_err}
        boot_stats = bp.bootstrap_samples(dist, statistics, B, num_threads=12)
        theta_star = boot_stats["theta_star"]
        se_star = boot_stats["se_star"]

        se = bp.standard_error(dist, statistic, theta_star=theta_star)
        actual_low = theta_hat - 1.645 * se
        actual_high = theta_hat + 1.645 * se
        print(f"{'standard'.ljust(12)}\t{actual_low:.01f}\t{actual_high:.01f}")
        assert actual_low == pytest.approx(expected_standard_low, abs=0.1)
        assert actual_high == pytest.approx(expected_standard_high, abs=0.1)

        actual_low, actual_high = bp.percentile_interval(
            dist, statistic, alpha=alpha, theta_star=theta_star
        )
        print(
            f"{'percentile'.ljust(12)}\t{actual_low:.01f}\t{actual_high:.01f}"
        )
        assert actual_low == pytest.approx(expected_percentile_low, abs=0.1)
        assert actual_high == pytest.approx(expected_percentile_high, abs=0.1)

        actual_low, actual_high = bp.bcanon_interval(
            dist, statistic, df, alpha=alpha, theta_star=theta_star
        )
        print(f"{'BCa'.ljust(12)}\t{actual_low:.01f}\t{actual_high:.01f}")
        assert actual_low == pytest.approx(expected_bca_low, abs=0.1)
        assert actual_high == pytest.approx(expected_bca_high, abs=0.1)

        se_hat = fast_std_err(df)
        actual_low, actual_high = bp.t_interval(
            dist,
            statistic,
            theta_hat,
            stabilize_variance=False,
            se_hat=se_hat,
            fast_std_err=fast_std_err,
            alpha=alpha,
            theta_star=theta_star,
            se_star=se_star,
        )
        print(
            f"{'bootstrap-t'.ljust(12)}\t{actual_low:.01f}\t{actual_high:.01f}"
        )
        assert actual_low == pytest.approx(expected_t_low, abs=0.1)
        assert actual_high == pytest.approx(expected_t_high, abs=0.1)

        actual_low, actual_high = bp.t_interval(
            dist,
            statistic,
            theta_hat,
            stabilize_variance=True,
            se_hat=se_hat,
            fast_std_err=fast_std_err,
            alpha=alpha,
            theta_star=theta_star,
            se_star=se_star,
            Bvar=400,
            num_threads=12,
        )
        print(
            f"{'var-stab-t'.ljust(12)}\t{actual_low:.01f}\t{actual_high:.01f}"
        )
        assert actual_low == pytest.approx(expected_stab_low, abs=0.1)
        assert actual_high == pytest.approx(expected_stab_high, abs=0.1)

    @pytest.mark.slow
    def test_calibrate_interval(self):
        df = datasets.law_data(full=False)
        alpha = 0.05
        expected_ci_low = 0.1596
        expected_ci_high = 0.9337
        expected_a_low = 0.0090
        expected_a_high = 0.9662

        def statistic(df):
            return np.corrcoef(df["LSAT"], df["GPA"])[0, 1]

        def resampling_statistic(df, p):
            mean_gpa = np.dot(df["GPA"], p)
            mean_lsat = np.dot(df["LSAT"], p)
            sigma_gpa = np.dot(p, (df["GPA"] - mean_gpa) ** 2)
            sigma_lsat = np.dot(p, (df["LSAT"] - mean_lsat) ** 2)
            corr = (df["GPA"] - mean_gpa) * (df["LSAT"] - mean_lsat)

            sigma_gpa = max([sigma_gpa, 1e-9])
            sigma_lsat = max([sigma_lsat, 1e-9])

            corr = np.dot(corr, p) / np.sqrt(sigma_gpa * sigma_lsat)
            return corr

        theta_hat = statistic(df)
        dist = bp.EmpiricalDistribution(df)
        np.random.seed(3)
        ci_low, ci_high, a_low, a_high = bp.calibrate_interval(
            dist,
            resampling_statistic,
            df,
            theta_hat,
            alpha=alpha,
            B=200,
            return_confidence_points=True,
            num_threads=12,
        )

        assert ci_low == pytest.approx(expected_ci_low, abs=1e-4)
        assert ci_high == pytest.approx(expected_ci_high, abs=1e-4)
        assert a_low == pytest.approx(expected_a_low, abs=1e-4)
        assert a_high == pytest.approx(expected_a_high, abs=1e-4)

    @pytest.mark.skip
    def test_importance_sampling(self):
        np.random.seed(0)
        n = 100
        alpha = 0.025

        # The mean of n standard gaussians is gaussians with mean 0
        # and variance 1/n. Thus theta ~ N(0, 1/n).
        c = ss.norm.isf(alpha, 0, 1 / np.sqrt(n))

        x = np.random.normal(loc=0.0, scale=1.0, size=n)

        def statistic(x, p):
            return np.dot(x, p)

        def g(lmbda):
            p = np.exp(lmbda * (x - np.mean(x)))
            return p / sum(p)

        lmbda = optimize.root_scalar(
            lambda lmbda: c - sum(g(lmbda) * x),
            method="bisect",
            bracket=(-10, 10),
        ).root

        p_i = g(lmbda)
        assert sum(x * p_i) == pytest.approx(c)

        B = 1000
        ind = range(n)
        prob = 0.0
        for i in range(B):
            x_star = np.random.choice(ind, n, True, p_i)
            p_star = np.array(
                [np.count_nonzero(x_star == j) for j in ind], np.float64
            )
            p_star /= n
            theta_star = statistic(x, p_star)
            if theta_star > c:
                w = np.exp(-np.sum(np.log(p_i[x_star] * n)))
                prob += w

        prob /= B
        assert prob == pytest.approx(alpha)


class TestBias:
    def test_bias(self):
        df = datasets.patch_data()
        expected_tF_hat = -0.0713
        expected_bias_hat = 0.00631
        expected_jackknife_bias = 0.0080

        def statistic(df):
            return df["y"].mean() / df["z"].mean()

        dist = bp.EmpiricalDistribution(df)

        actual = dist.calculate_parameter(statistic)
        assert actual == pytest.approx(expected_tF_hat, abs=1e-4)

        np.random.seed(4)
        actual = bp.bias(dist, statistic, statistic, B=400)
        assert actual == pytest.approx(expected_bias_hat, abs=1e-5)

        actual = bp.jackknife_bias(df, statistic)
        assert actual == pytest.approx(expected_jackknife_bias, abs=1e-4)

    def test_se_bias(self):
        df = datasets.patch_data()
        expected = 0.0081

        def stat(df):
            return df["y"].mean() / df["z"].mean()

        def statistic(df):
            return bp.jackknife_bias(df, stat)

        dist = bp.EmpiricalDistribution(df)
        np.random.seed(0)
        actual = bp.standard_error(dist, statistic, B=200)
        assert actual == pytest.approx(expected, abs=1e-3)

    @pytest.mark.slow
    def test_bias_correction(self):
        df = datasets.patch_data()
        exp_unc_value = -0.0713
        exp_unc_se = 0.1021
        exp_unc_bias = 0.0077

        exp_bb_value = -0.0777
        exp_bb_se = 0.1100

        exp_bbb_value = -0.0787
        exp_bbb_se = 0.1014

        exp_jb_value = -0.0793
        exp_jb_se = 0.0967

        def stat(df):
            return df["y"].mean() / df["z"].mean()

        def resampling_stat(df, p):
            return np.dot(p, df["y"]) / np.dot(p, df["z"])

        dist = bp.EmpiricalDistribution(df)
        np.random.seed(0)

        # Compute the (uncorrected) value of the statistic and its
        # standard error and bias.
        st = time.time()
        uncorrected_value = stat(df)
        unc_dur = time.time() - st
        uncorrected_bias, theta_star = bp.better_bootstrap_bias(
            df, resampling_stat, B=4000, return_samples=True, num_threads=12
        )
        uncorrected_se = bp.standard_error(
            dist, stat, B=200, theta_star=theta_star
        )

        # Compute the better bootstrap bias corrected value of the
        # statistic, and the standard error of that bias-correction.
        np.random.seed(0)
        st = time.time()
        bbb_actual = bp.bias_corrected(
            df, resampling_stat, method="better_bootstrap_bias", B=400
        )
        bbb_dur = time.time() - st

        bbb_bc_se = bp.standard_error(
            dist,
            lambda df: bp.bias_corrected(
                df, resampling_stat, method="better_bootstrap_bias", B=400
            ),
            B=200,
            num_threads=12,
        )

        # Compute the regular bootstrap bias corrected value of the
        # statistic, and the standard error of that bias-correction.
        st = time.time()
        bb_actual = bp.bias_corrected(
            df, stat, method="bias", t=stat, dist=dist, B=400
        )
        bb_dur = time.time() - st

        bb_bc_se = bp.standard_error(
            dist,
            lambda df: bp.bias_corrected(
                df, stat, method="bias", t=stat, dist=dist, B=400
            ),
            B=25,
            num_threads=12,
        )

        # Compute the jackknife bias corrected value of the
        # statistic, and the standard error of that bias-correction.
        st = time.time()
        jb_actual = bp.bias_corrected(df, stat, method="jackknife")
        jb_dur = time.time() - st

        jb_bc_se = bp.standard_error(
            dist,
            lambda df: bp.bias_corrected(df, stat, method="jackknife"),
            B=200,
            num_threads=12,
        )

        print("      Stat      \t Value \t  se  \t bias \t Time")
        print(
            f"Uncorrected     \t{uncorrected_value:0.04f}"
            f"\t{uncorrected_se:0.04f}\t{uncorrected_bias:0.04f}"
            f"\t{unc_dur:.03f}"
        )
        print(
            f"Bootstrap       \t{bb_actual:0.04f}"
            f"\t{bb_bc_se:0.04f}\t\t{bb_dur:0.03f}"
        )

        print(
            f"Better Bootstrap\t{bbb_actual:0.04f}"
            f"\t{bbb_bc_se:0.04f}\t\t{bbb_dur:0.03f}"
        )

        print(
            f"Jackknife       \t{jb_actual:0.04f}"
            f"\t{jb_bc_se:0.04f}\t\t{jb_dur:0.03f}"
        )
        """
              Stat      	 Value 	  se  	 bias 	 Time
        Uncorrected     	-0.0713	0.1021	0.0077	0.000
        Bootstrap       	-0.0777	0.1100		0.274
        Better Bootstrap	-0.0787	0.1014		0.098
        Jackknife       	-0.0793	0.0967		0.011

        These results show that the uncorrected estimate:
          mean(y) / mean(z)
        is biased. The bias is about 10% of the value. The bias shown
        in the first row was calculated using the better bootstrap
        bias estimate. We can compute bias-corrected values using the
        regular bootstrap method, the better bootstrap method, or the
        jackknife. For both bootstrap methods, 400 samples were
        used. The Better Bootstrap and Jackknife gave very similar
        values and standard errors In fact, the standard errors for
        both bootstrap estimates were in line with the uncorrected
        statistic. When we used 4000 bootstrap samples for the regular
        bootstrap method, the value was in line with the Jackknife and
        better bootstrap methods.

        Conclusions: the Jackknife is fast, and in this case gave
        accurate answers with low variance. [ET93] reports that the
        Jackknife can have high variance, but we did not see that
        here. The Better Bootstrap also gave accurate answers and has
        low variance but takes 10x as long to run. The regular
        Bootstrap gave inaccurate answers with 400 samples and took
        30x as long as the jackknife.

        """
        assert uncorrected_value == pytest.approx(exp_unc_value, abs=1e-4)
        assert uncorrected_se == pytest.approx(exp_unc_se, abs=1e-4)
        assert uncorrected_bias == pytest.approx(exp_unc_bias, abs=1e-4)

        assert bb_actual == pytest.approx(exp_bb_value, abs=1e-4)
        assert bb_bc_se == pytest.approx(exp_bb_se, abs=1e-4)

        assert bbb_actual == pytest.approx(exp_bbb_value, abs=1e-4)
        assert bbb_bc_se == pytest.approx(exp_bbb_se, abs=1e-4)

        assert jb_actual == pytest.approx(exp_jb_value, abs=1e-4)
        assert jb_bc_se == pytest.approx(exp_jb_se, abs=1e-4)

    def test_better_bootstrap_bias(self):
        df = datasets.patch_data()
        expected = 0.0073

        def stat(df, p):
            return np.dot(p, df["y"]) / np.dot(p, df["z"])

        np.random.seed(0)
        actual = bp.better_bootstrap_bias(df, stat, B=400)
        assert actual == pytest.approx(expected, abs=1e-4)

    def test_two_sample_mouse_data(self):
        control = datasets.mouse_data("control")
        treatment = datasets.mouse_data("treatment")

        expected_theta = 30.63
        expected_se = 26.85
        expected_bias = 1.2409
        expected_ci_low_bca = -13.6984
        expected_ci_high_bca = 77.6508
        expected_ci_low_vst = -3.6571
        expected_ci_high_vst = 92.6593
        expected_jse = 28.9361

        def statistic(ab):
            a, b = ab
            return np.mean(b) - np.mean(a)

        actual = statistic((control, treatment))
        assert actual == pytest.approx(expected_theta, abs=0.01)

        dist = bp.MultiSampleEmpiricalDistribution((control, treatment))
        actual = dist.calculate_parameter(statistic)
        assert actual == pytest.approx(expected_theta, abs=0.01)

        np.random.seed(0)
        actual, theta_star = bp.standard_error(
            dist, statistic, B=1400, return_samples=True
        )
        assert actual == pytest.approx(expected_se, abs=0.6)

        actual = bp.bias(dist, statistic, statistic, theta_star=theta_star)
        assert actual == pytest.approx(expected_bias, abs=1e-4)

        actual_low, actual_high = bp.bcanon_interval(
            dist, statistic, (control, treatment), theta_star=theta_star
        )
        assert actual_low == pytest.approx(expected_ci_low_bca, abs=1e-4)
        assert actual_high == pytest.approx(expected_ci_high_bca, abs=1e-4)

        actual_low, actual_high = bp.t_interval(
            dist,
            statistic,
            statistic((control, treatment)),
            stabilize_variance=True,
            empirical_distribution=bp.MultiSampleEmpiricalDistribution,
        )
        assert actual_low == pytest.approx(expected_ci_low_vst, abs=1e-4)
        assert actual_high == pytest.approx(expected_ci_high_vst, abs=1e-4)

        actual = bp.jackknife_standard_error((control, treatment), statistic)
        assert actual == pytest.approx(expected_jse, abs=1e-4)


class TestSignificance:
    def test_achieved_significance_levels(self):
        control = datasets.mouse_data("control")
        treatment = datasets.mouse_data("treatment")
        expected = 0.132
        expected_bcanon = 0.147
        expected_se = 0.181
        expected_seatm = 30.2939
        expected_asl_atm = 0.167

        def statistic(ab):
            a, b = ab
            return np.mean(b) - np.mean(a)

        def alpha_trimmed_mean(ab, alpha=0.10):
            a, b = ab
            return ss.trim_mean(b, alpha) - ss.trim_mean(a, alpha)

        theta_hat = statistic((control, treatment))
        dist = bp.MultiSampleEmpiricalDistribution((control, treatment))
        np.random.seed(0)
        actual = bp.percentile_asl(
            dist, statistic, (control, treatment), theta_hat=theta_hat, B=1000
        )
        assert actual == pytest.approx(expected, abs=0.012)

        # Try flipping treatment/control -- should get similar (but not
        # exactly the same) answer.
        flipped_dist = bp.MultiSampleEmpiricalDistribution(
            (treatment, control)
        )
        actual = bp.percentile_asl(
            flipped_dist,
            statistic,
            (treatment, control),
            theta_hat=-theta_hat,
            B=1000,
        )
        assert actual == pytest.approx(expected, abs=0.015)

        # bca correction
        actual = bp.bcanon_asl(
            dist, statistic, (control, treatment), theta_hat=theta_hat, B=1000
        )
        assert actual == pytest.approx(expected_bcanon, abs=0.025)

        # Assess the standard error in the ASL. Commentary: it's
        # large!  The ASL was about 0.122, and the standard error was
        # about 0.181, so there is tremendous uncertainty in the
        # ASL. This uncertainty is attributable to the original sample
        # size, not the number of bootstrap iterations. Increasing the
        # latter to 100_000 made no meaningful difference in the
        # standard error.
        def asl(ab):
            dist = bp.MultiSampleEmpiricalDistribution(ab)
            return bp.bcanon_asl(dist, statistic, ab, B=1000)

        actual = bp.standard_error(dist, asl, B=25)
        assert actual == pytest.approx(expected_se, abs=1e-3)

        # These are just to reproduce some of the examples in the book.
        actual = bp.standard_error(
            dist, lambda x: alpha_trimmed_mean(x, alpha=0.15)
        )
        assert actual == pytest.approx(expected_seatm, abs=1e-4)

        actual = bp.bcanon_asl(
            dist,
            lambda x: alpha_trimmed_mean(x, alpha=0.25),
            (control, treatment),
            B=1000,
        )
        # The achieved significance level I get is pretty different than
        # in the book!
        assert actual == pytest.approx(expected_asl_atm, abs=0.09)

    def test_asl_variance(self):
        control = datasets.mouse_data("control")
        treatment = datasets.mouse_data("treatment")
        # expected_ratio = 2.48
        expected_ratio = 0.907
        expected_asl = 0.119

        def statistic(ab):
            a, b = ab
            return np.log(np.var(b, ddof=1) / np.var(a, ddof=1))

        theta_hat = statistic((control, treatment))
        assert theta_hat == pytest.approx(expected_ratio, abs=0.01)

        dist = bp.MultiSampleEmpiricalDistribution((control, treatment))
        np.random.seed(0)
        actual = bp.percentile_asl(
            dist, statistic, (control, treatment), theta_hat=theta_hat, B=1000
        )
        assert actual == expected_asl

    def test_combined_asl(self):
        """Algorithm 16.1 in [ET93].

        """
        control = datasets.mouse_data("control")
        treatment = datasets.mouse_data("treatment")
        combined = control + treatment

        expected_obs = 30.63
        expected_obs_st = 1.12
        expected_asl = 0.120
        expected_asl_st = 0.134

        n = len(control)
        m = len(treatment)

        def statistic(x):
            return np.mean(x[n:]) - np.mean(x[0:n])

        def studentized(x):
            num = np.mean(x[n:]) - np.mean(x[0:n])
            sigma_bar = n * np.var(x[0:n], ddof=0) + m * np.var(x[n:], ddof=0)
            sigma_bar = np.sqrt(sigma_bar / (n + m - 2))
            den = sigma_bar * np.sqrt(1 / n + 1 / m)
            return num / den

        dist = bp.EmpiricalDistribution(combined)

        t_obs = statistic(combined)
        assert t_obs == pytest.approx(expected_obs, abs=0.01)
        np.random.seed(0)
        actual = bp.bootstrap_asl(
            dist, statistic, combined, theta_hat=t_obs, B=1000
        )
        assert actual == pytest.approx(expected_asl, abs=0.011)

        t_obs = studentized(combined)
        assert t_obs == pytest.approx(expected_obs_st, abs=0.01)
        actual = bp.bootstrap_asl(
            dist, studentized, combined, theta_hat=t_obs, B=1000
        )
        assert actual == pytest.approx(expected_asl_st, abs=0.01)

    def test_asl_and_power(self):
        class EqualMeansEmpiricalDistribution(
            bp.MultiSampleEmpiricalDistribution
        ):
            """Translates datasets to have a common mean.

            This test is intended to demonstrate that we can easily extend
            Empirical Distributions to have functionality not supported by
            the built-in methods. In this case, we modify the
            MultiSampleEmpiricalDistribution so that the datasets have a
            common mean.

            """

            def __init__(self, datasets, lift=0.0):
                y, z = datasets
                y_bar = np.mean(y)
                z_bar = np.mean(z)
                x_bar = (sum(y) + sum(z)) / (len(y) + len(z))

                # Translate datasets to have the same mean, x_bar, which
                # is the mean of the combined datasets.
                yy = [yi - y_bar + x_bar for yi in y]
                zz = [zi - z_bar + x_bar for zi in z]
                if lift != 0:
                    zz = [zi * (1 + lift) for zi in zz]

                # Defer to MultiSampleEmpiricalDistribution for all
                # remaining functionality.
                super().__init__((yy, zz))

        control = datasets.mouse_data("control")
        treatment = datasets.mouse_data("treatment")
        expected = 0.152

        def studentized(yz):
            y, z = yz
            num = np.mean(z) - np.mean(y)
            den = np.sqrt(
                np.var(z, ddof=0) / len(z) + np.var(y, ddof=0) / len(y)
            )
            return num / den

        dist = EqualMeansEmpiricalDistribution((control, treatment))
        t_obs = studentized((control, treatment))
        np.random.seed(0)
        actual = bp.bootstrap_asl(
            dist, studentized, (control, treatment), theta_hat=t_obs, B=1000
        )
        assert actual == pytest.approx(expected, abs=0.01)

        class AlternativeEmpiricalDistribution(
            EqualMeansEmpiricalDistribution
        ):
            def __init__(self, data):
                super().__init__(data, lift=0.10)

        alt_dist = AlternativeEmpiricalDistribution((control, treatment))
        size = (500, 500)
        P = 10
        expected = 0.8
        np.random.seed(0)
        actual = bp.bootstrap_power(
            alt_dist,
            EqualMeansEmpiricalDistribution,
            studentized,
            bp.bootstrap_asl,
            alpha=0.05,
            size=size,
            P=P,
            B=100,
        )
        assert actual == pytest.approx(expected, abs=0.01)


class TestPredictionError:
    @pytest.mark.slow
    def test_bootstrap_prediction_error(self):
        """Prediction Errors

        The table below shows various estimates of prediction error
        for the hormone data and the standard error or those
        estimates. The apparent error is simply the training error and
        is overly optimistic. Leave-one-out cross validation (LOOCV)
        has low bias but can have high variance, though we see no
        evidence of that in this example. The optimism based approach
        has fairly high standard error and is pretty far away from the
        LOOCV value. The .632 and .632+ estimates give similar answers
        to LOOCV, but the .632+ estimate has disturbingly high
        variance, despite what [ET93] reports.

            Method     	Prediction Error	Standard Error	Time
        Apparent       	            2.20	          0.42	0.033
        LOOCV          	            3.09	          0.59	0.038
        Optimism       	            2.93	          0.79	0.683
        Bootstrap .632 	            3.15	          0.59	0.581
        Bootstrap .632+	            3.06	          0.81	0.598

        """
        df = datasets.hormone_data()
        formula = "amount ~ C(lot, levels=['A', 'B', 'C']) + hrs"
        expected_ae = 2.20
        expected_cve = 3.09
        expected_be_opt = 2.93
        expected_be_632 = 3.15
        expected_be_632p = 3.06

        def train(df):
            mdl = smf.ols(formula=formula, data=df)
            return mdl.fit()

        def predict(mdl, dataset):
            return mdl.predict(dataset)

        def apparent_error(df):
            mdl = train(df)
            pred = predict(mdl, df)
            return np.mean(error(pred, df))

        def loocv_error(df):
            N = len(df)
            mdl = train(df)
            pred = predict(mdl, df)
            hi = np.empty((N,), dtype=mdl.model.wexog.dtype)
            for i in range(N):
                hi[i] = (
                    mdl.model.wexog[i, :]
                    .dot(mdl.normalized_cov_params)
                    .dot(mdl.model.wexog[i, :])
                )

            cv_error = np.mean(((pred - df["amount"]) / (1 - hi)) ** 2)
            return cv_error

        def error(pred, dataset):
            return (pred - dataset["amount"]) ** 2

        def no_inf_err(pred, dataset):
            s = 0
            n = len(dataset)
            for i in range(n):
                for j in range(n):
                    err = (pred[i] - dataset["amount"].iloc[j]) ** 2
                    s += err
            return s / (n * n)

        np.random.seed(0)
        dist = bp.EmpiricalDistribution(df)
        st = time.time()
        ae = apparent_error(df)
        duration = time.time() - st
        se = bp.standard_error(dist, apparent_error, B=25, num_threads=12)
        assert ae == pytest.approx(expected_ae, abs=0.01)
        print("    Method     \tPrediction Error\tStandard Error\tTime")
        print(
            f"Apparent       \t            {ae:.02f}\t          {se:.02f}\t{duration:.03f}"
        )

        st = time.time()
        loocve = loocv_error(df)
        duration = time.time() - st
        se = bp.standard_error(dist, loocv_error, B=25, num_threads=12)
        assert loocve == pytest.approx(expected_cve, abs=0.01)
        print(
            f"LOOCV          \t            {loocve:.02f}\t          {se:.02f}\t{duration:.03f}"
        )

        st = time.time()
        boot_error = bp.prediction_error_optimism(
            dist, df, train, predict, error, B=100, num_threads=12
        )
        duration = time.time() - st
        se = bp.standard_error(
            dist,
            lambda x: bp.prediction_error_optimism(
                bp.EmpiricalDistribution(x), x, train, predict, error, B=100
            ),
            B=25,
            num_threads=12,
        )
        print(
            f"Optimism       \t            {boot_error:.02f}\t          {se:.02f}\t{duration:.03f}"
        )
        assert boot_error == pytest.approx(expected_be_opt, abs=0.01)

        st = time.time()
        boot_error = bp.prediction_error_632(
            dist,
            df,
            train,
            predict,
            error,
            B=100,
            use_632_plus=False,
            num_threads=12,
        )
        duration = time.time() - st
        se = bp.standard_error(
            dist,
            lambda x: bp.prediction_error_632(
                bp.EmpiricalDistribution(x),
                x,
                train,
                predict,
                error,
                B=100,
                use_632_plus=False,
            ),
            B=25,
            num_threads=12,
        )
        print(
            f"Bootstrap .632 \t            {boot_error:.02f}\t          {se:.02f}\t{duration:.03f}"
        )
        assert boot_error == pytest.approx(expected_be_632, abs=0.03)

        st = time.time()
        boot_error = bp.prediction_error_632(
            dist,
            df,
            train,
            predict,
            error,
            B=100,
            use_632_plus=True,
            no_inf_err_rate=no_inf_err,
            num_threads=12,
        )
        duration = time.time() - st
        se = bp.standard_error(
            dist,
            lambda x: bp.prediction_error_632(
                bp.EmpiricalDistribution(x),
                x,
                train,
                predict,
                error,
                B=100,
                use_632_plus=True,
                no_inf_err_rate=no_inf_err,
            ),
            B=25,
            num_threads=12,
        )
        print(
            f"Bootstrap .632+\t            {boot_error:.02f}\t          {se:.02f}\t{duration:.03f}"
        )
        assert boot_error == pytest.approx(expected_be_632p, abs=0.03)


class TestPredictionIntervals:
    def test_prediction_intervals(self):
        np.random.seed(0)
        x = datasets.mouse_data("control")
        dist = bp.EmpiricalDistribution(x)
        B = 2000
        alpha = 0.05

        expected_low_norm = -27.0361
        expected_high_norm = 139.4806
        expected_low_boot = 1.7674
        expected_high_boot = 165.0320

        x_bar = np.mean(x)
        s = np.std(x, ddof=1)

        n = len(x)
        t_norm_alpha = ss.t.ppf(alpha, df=(n - 1))
        t_norm_one_minus_alpha = ss.t.ppf(1 - alpha, df=(n - 1))
        pred_low_norm = x_bar - t_norm_one_minus_alpha * np.sqrt(1 + 1 / n) * s
        pred_high_norm = x_bar - t_norm_alpha * np.sqrt(1 + 1 / n) * s

        assert pred_low_norm == pytest.approx(expected_low_norm, abs=1e-4)
        assert pred_high_norm == pytest.approx(expected_high_norm, abs=1e-4)

        pred_low_boot, pred_high_boot = bp.prediction_interval(
            dist,
            x,
            mean=np.mean,
            std=lambda x: np.std(x, ddof=1),
            B=B,
            alpha=alpha,
            num_threads=12,
        )

        assert pred_low_boot == pytest.approx(expected_low_boot, abs=1e-4)
        assert pred_high_boot == pytest.approx(expected_high_boot, abs=1e-4)
