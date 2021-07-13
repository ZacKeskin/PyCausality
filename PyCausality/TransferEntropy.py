from copy import deepcopy

import numpy as np
import pandas as pd
import statsmodels.api as sm

# from dateutil.relativedelta import relativedelta

from .LaggedTimeSeries import LaggedTimeSeries
from .Utils.Functions.get_entropy import get_entropy
from .Utils.Functions.sanitise import sanitise
from .Utils.Functions.shuffle_series import shuffle_series


class TransferEntropy:
    """Functional class to calculate Transfer Entropy between time series, to detect causal signals.

    Currently accepts two series: X(t) and Y(t). Future extensions planned to accept additional endogenous
    series: X1(t), X2(t), X3(t) etc.

    """

    def __init__(self, DF, endog, exog, lag=None, window_size=None, window_stride=None):
        """init.

        Arguments:
            DF            -   (DataFrame) Time series data for X and Y (NOT including lagged variables)
            endog         -   (string)    Fieldname for endogenous (dependent) variable Y
            exog          -   (string)    Fieldname for exogenous (independent) variable X
            lag           -   (integer)   Number of periods (rows) by which to lag timeseries data
            window_size   -   (Dict)      Must contain key-value pairs only from within: {'YS':0,'MS':0,'D':0,'H':0,'min':0,'S':0,'ms':0}
                                          Describes the desired size of each window, provided the data is indexed with datetime type. Leave as
                                          None for no windowing. Units follow http://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases
            window_stride -   (Dict)      Must contain key-value pairs only from within: {'YS':0,'MS':0,'D':0,'H':0,'min':0,'S':0,'ms':0}
                                          Describes the size of the step between consecutive windows, provided the data is indexed with datetime type. Leave as
                                          None for no windowing. Units follow http://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases

        Returns:
            n/a

        """
        self.lts = LaggedTimeSeries(
            df=sanitise(DF),
            lag=lag,
            window_size=window_size,
            window_stride=window_stride,
        )

        if self.lts.has_windows is True:
            self.df = self.lts.windows
            self.date_index = self.lts.daterange[self.lts.headstart :]
            self.results = pd.DataFrame(index=self.date_index)
            self.results.index.name = "windows_ending_on"
        else:
            self.df = [self.lts.df]
            self.results = pd.DataFrame(index=[0])
        self.max_lag_only = True
        self.endog = endog  # Dependent Variable Y
        self.exog = exog  # Independent Variable X
        self.lag = lag

        """ If using KDE, this ensures the covariance matrices are calculated once over all data, rather
            than for each window. This saves computational time and provides a fair point for comparison."""
        self.covars = [[], []]
        for i, (X, Y) in enumerate(
            {self.exog: self.endog, self.endog: self.exog}.items()
        ):
            X_lagged = X + "_lag" + str(self.lag)
            Y_lagged = Y + "_lag" + str(self.lag)

            self.covars[i] = [
                np.cov(self.lts.df[[Y, Y_lagged, X_lagged]].values.T),
                np.cov(self.lts.df[[X_lagged, Y_lagged]].values.T),
                np.cov(self.lts.df[[Y, Y_lagged]].values.T),
                np.ones(shape=(1, 1)) * self.lts.df[Y_lagged].std() ** 2,
            ]

    def linear_TE(self, df=None, n_shuffles=0):
        """Linear Transfer Entropy for directional causal inference.

        Defined:            G-causality * 0.5, where G-causality described by the reduction in variance of the residuals
                            when considering side information.

        Calculated using:   log(var(e_joint)) - log(var(e_independent)) where e_joint and e_independent
                            represent the residuals from OLS fitting in the joint (X(t),Y(t)) and reduced (Y(t)) cases

        Arguments:
            n_shuffles  -   (integer)   Number of times to shuffle the dataframe, destroying the time series temporality, in order to
                                        perform significance testing.

        Returns:
            transfer_entropies  -  (list) Directional Linear Transfer Entropies from X(t)->Y(t) and Y(t)->X(t) respectively

        """
        # Prepare lists for storing results
        TEs = []
        shuffled_TEs = []
        p_values = []
        z_scores = []

        # Loop over all windows
        for i, df in enumerate(self.df):
            df = deepcopy(df)

            # Shows user that something is happening
            if self.lts.has_windows is True:
                print("Window ending: ", self.date_index[i])

            # Initialise list to return TEs
            transfer_entropies = [0, 0]

            # Require us to compare information transfer bidirectionally
            for i, (X, Y) in enumerate(
                {self.exog: self.endog, self.endog: self.exog}.items()
            ):

                # Note X-t, Y-t
                X_lagged = X + "_lag" + str(self.lag)
                Y_lagged = Y + "_lag" + str(self.lag)

                # Calculate Residuals after OLS Fitting, for both Independent and Joint Cases
                joint_residuals = (
                    sm.OLS(df[Y], sm.add_constant(df[[Y_lagged, X_lagged]])).fit().resid
                )
                independent_residuals = (
                    sm.OLS(df[Y], sm.add_constant(df[Y_lagged])).fit().resid
                )

                # Use Geweke's formula for Granger Causality
                granger_causality = np.log(
                    np.var(independent_residuals) / np.var(joint_residuals)
                )

                # Calculate Linear Transfer Entropy from Granger Causality
                transfer_entropies[i] = granger_causality / 2

            TEs.append(transfer_entropies)

            # Calculate Significance of TE during this window
            if n_shuffles > 0:
                p, z, TE_mean = significance(
                    df=df,
                    TE=transfer_entropies,
                    endog=self.endog,
                    exog=self.exog,
                    lag=self.lag,
                    n_shuffles=n_shuffles,
                    method="linear",
                )

                shuffled_TEs.append(TE_mean)
                p_values.append(p)
                z_scores.append(z)

        # Store Linear Transfer Entropy from X(t)->Y(t) and from Y(t)->X(t)
        self.add_results(
            {
                "TE_linear_XY": np.array(TEs)[:, 0],
                "TE_linear_YX": np.array(TEs)[:, 1],
                "p_value_linear_XY": None,
                "p_value_linear_YX": None,
                "z_score_linear_XY": 0,
                "z_score_linear_YX": 0,
            }
        )

        if n_shuffles > 0:
            # Store Significance Transfer Entropy from X(t)->Y(t) and from Y(t)->X(t)

            self.add_results(
                {
                    "p_value_linear_XY": np.array(p_values)[:, 0],
                    "p_value_linear_YX": np.array(p_values)[:, 1],
                    "z_score_linear_XY": np.array(z_scores)[:, 0],
                    "z_score_linear_YX": np.array(z_scores)[:, 1],
                    "Ave_TE_linear_XY": np.array(shuffled_TEs)[:, 0],
                    "Ave_TE_linear_YX": np.array(shuffled_TEs)[:, 1],
                }
            )

        return transfer_entropies

    def nonlinear_TE(
        self,
        df=None,
        pdf_estimator="histogram",
        bins=None,
        bandwidth=None,
        gridpoints=20,
        n_shuffles=0,
    ):
        """Nonlinear Transfer Entropy for directional causal inference.

        Defined:            TE = TE_XY - TE_YX      where TE_XY = H(Y|Y-t) - H(Y|Y-t,X-t)
        Calculated using:   H(Y|Y-t,X-t) = H(Y,Y-t,X-t) - H(Y,Y-t)  and finding joint entropy through density estimation

        Arguments:
            pdf_estimator   -   (string)    'Histogram' or 'kernel' Used to define which method is preferred for density estimation
                                            of the distribution - either histogram or KDE
            bins            -   (dict of lists) Optional parameter to provide hard-coded bin-edges. Dict keys
                                            must contain names of variables - including lagged columns! Dict values must be lists
                                            containing bin-edge numerical values.
            bandwidth       -   (float)     Optional parameter for custom bandwidth in KDE. This is a scalar multiplier to the covariance
                                            matrix used (see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.covariance_factor.html)
            gridpoints      -   (integer)   Number of gridpoints (in each dimension) to discretise the probablity space when performing
                                            integration of the kernel density estimate. Increasing this gives more precision, but significantly
                                            increases execution time
            n_shuffles      -   (integer)   Number of times to shuffle the dataframe, destroying the time series temporality, in order to
                                            perform significance testing.

        Returns:
            transfer_entropies  -  (list) Directional Transfer Entropies from X(t)->Y(t) and Y(t)->X(t) respectively

        (Also stores TE, Z-score and p-values in self.results - for each window if windows defined.)

        """
        # Retrieve user-defined bins
        self.bins = bins
        if self.bins is None:
            self.bins = {self.endog: None}

        # Prepare lists for storing results
        TEs = []
        shuffled_TEs = []
        p_values = []
        z_scores = []

        # Loop over all windows
        for i, df in enumerate(self.df):
            df = deepcopy(df)

            # Shows user that something is happening
            if self.lts.has_windows is True:
                print("Window ending: ", self.date_index[i])

            # Initialise list to return TEs
            transfer_entropies = [0, 0]

            # Require us to compare information transfer bidirectionally
            for i, (X, Y) in enumerate(
                {self.exog: self.endog, self.endog: self.exog}.items()
            ):

                # Entropy calculated using Probability Density Estimation:
                # Following: https://stat.ethz.ch/education/semesters/SS_2006/CompStat/sk-ch2.pdf
                # Also: https://www.cs.cmu.edu/~aarti/Class/10704_Spring15/lecs/lec5.pdf

                # Note Lagged Terms
                X_lagged = X + "_lag" + str(self.lag)
                Y_lagged = Y + "_lag" + str(self.lag)

                # Estimate PDF using Gaussian Kernels and use H(x) = p(x) log p(x)

                # 1. H(Y,Y-t,X-t)
                H1 = get_entropy(
                    df=df[[Y, Y_lagged, X_lagged]],
                    gridpoints=gridpoints,
                    bandwidth=bandwidth,
                    estimator=pdf_estimator,
                    bins={
                        k: v
                        for (k, v) in self.bins.items()
                        if k in [Y, Y_lagged, X_lagged]
                    },
                    covar=self.covars[i][0],
                )
                # 2. H(Y-t,X-t)
                H2 = get_entropy(
                    df=df[[X_lagged, Y_lagged]],
                    gridpoints=gridpoints,
                    bandwidth=bandwidth,
                    estimator=pdf_estimator,
                    bins={
                        k: v
                        for (k, v) in self.bins.items()
                        if k in [X_lagged, Y_lagged]
                    },
                    covar=self.covars[i][1],
                )
                # 3. H(Y,Y-t)
                H3 = get_entropy(
                    df=df[[Y, Y_lagged]],
                    gridpoints=gridpoints,
                    bandwidth=bandwidth,
                    estimator=pdf_estimator,
                    bins={k: v for (k, v) in self.bins.items() if k in [Y, Y_lagged]},
                    covar=self.covars[i][2],
                )
                # 4. H(Y-t)
                H4 = get_entropy(
                    df=df[[Y_lagged]],
                    gridpoints=gridpoints,
                    bandwidth=bandwidth,
                    estimator=pdf_estimator,
                    bins={k: v for (k, v) in self.bins.items() if k in [Y_lagged]},
                    covar=self.covars[i][3],
                )

                # Calculate Conditonal Entropy using: H(Y|X-t,Y-t) = H(Y,X-t,Y-t) - H(X-t,Y-t)
                conditional_entropy_joint = H1 - H2

                # And Conditional Entropy independent of X(t) H(Y|Y-t) = H(Y,Y-t) - H(Y-t)
                conditional_entropy_independent = H3 - H4

                # Directional Transfer Entropy is the difference between the conditional entropies
                transfer_entropies[i] = (
                    conditional_entropy_independent - conditional_entropy_joint
                )

            TEs.append(transfer_entropies)

            # Calculate Significance of TE during this window
            if n_shuffles > 0:
                p, z, TE_mean = significance(
                    df=df,
                    TE=transfer_entropies,
                    endog=self.endog,
                    exog=self.exog,
                    lag=self.lag,
                    n_shuffles=n_shuffles,
                    pdf_estimator=pdf_estimator,
                    bins=self.bins,
                    bandwidth=bandwidth,
                    method="nonlinear",
                )

                shuffled_TEs.append(TE_mean)
                p_values.append(p)
                z_scores.append(z)

        # Store Transfer Entropy from X(t)->Y(t) and from Y(t)->X(t)
        self.add_results(
            {
                "TE_XY": np.array(TEs)[:, 0],
                "TE_YX": np.array(TEs)[:, 1],
                "p_value_XY": None,
                "p_value_YX": None,
                "z_score_XY": 0,
                "z_score_YX": 0,
            }
        )
        if n_shuffles > 0:
            # Store Significance Transfer Entropy from X(t)->Y(t) and from Y(t)->X(t)

            self.add_results(
                {
                    "p_value_XY": np.array(p_values)[:, 0],
                    "p_value_YX": np.array(p_values)[:, 1],
                    "z_score_XY": np.array(z_scores)[:, 0],
                    "z_score_YX": np.array(z_scores)[:, 1],
                    "Ave_TE_XY": np.array(shuffled_TEs)[:, 0],
                    "Ave_TE_YX": np.array(shuffled_TEs)[:, 1],
                }
            )

        return transfer_entropies

    def add_results(self, dict):
        """Add results.

        Arguments:
            dict    -   JSON style data to store in existing self.results DataFrame

        Returns:
            n/a

        """
        for (k, v) in dict.items():
            self.results[str(k)] = v


def significance(
    df,
    TE,
    endog,
    exog,
    lag,
    n_shuffles,
    method,
    pdf_estimator=None,
    bins=None,
    bandwidth=None,
    both=True,
):
    """Perform significance analysis on the hypothesis test of statistical causality, for both X(t)->Y(t) and Y(t)->X(t) directions.

    Calculated using:  Assuming stationarity, we shuffle the time series to provide the null hypothesis.
                       The proportion of tests where TE > TE_shuffled gives the p-value significance level.
                       The amount by which the calculated TE is greater than the average shuffled TE, divided
                       by the standard deviation of the results, is the z-score significance level.

    Arguments:
        TE              -      (list)    Contains the transfer entropy in each direction, i.e. [TE_XY, TE_YX]
        endog           -      (string)  The endogenous variable in the TE analysis being significance tested (i.e. X or Y)
        exog            -      (string)  The exogenous variable in the TE analysis being significance tested (i.e. X or Y)
        pdf_estimator   -      (string)  The pdf_estimator used in the original TE analysis
        bins            -      (Dict of lists)  The bins used in the original TE analysis

        n_shuffles      -      (float) Number of times to shuffle the dataframe, destroyig temporality
        both            -      (Bool) Whether to shuffle both endog and exog variables (z-score) or just exog variables (giving z*-score)

    Returns:
        p_value         -      Probablity of observing the result given the null hypothesis
        z_score         -      Number of Standard Deviations result is from mean (normalised)

    """
    # Prepare array for Transfer Entropy of each Shuffle
    shuffled_TEs = np.zeros(shape=(2, n_shuffles))

    if both is True:
        pass  # TBC

    for i in range(n_shuffles):
        # Perform Shuffle
        df = shuffle_series(df)

        # Calculate New TE
        shuffled_causality = TransferEntropy(DF=df, endog=endog, exog=exog, lag=lag)
        if method == "linear":
            TE_shuffled = shuffled_causality.linear_TE(df, n_shuffles=0)
        else:
            TE_shuffled = shuffled_causality.nonlinear_TE(
                df, pdf_estimator, bins, bandwidth, n_shuffles=0
            )
        shuffled_TEs[:, i] = TE_shuffled

    # Calculate p-values for each direction
    p_values = (
        np.count_nonzero(TE[0] < shuffled_TEs[0, :]) / n_shuffles,
        np.count_nonzero(TE[1] < shuffled_TEs[1, :]) / n_shuffles,
    )

    # Calculate z-scores for each direction
    z_scores = (
        (TE[0] - np.mean(shuffled_TEs[0, :])) / np.std(shuffled_TEs[0, :]),
        (TE[1] - np.mean(shuffled_TEs[1, :])) / np.std(shuffled_TEs[1, :]),
    )

    TE_mean = (np.mean(shuffled_TEs[0, :]), np.mean(shuffled_TEs[1, :]))

    # Return the self.DF value to the unshuffled case
    return p_values, z_scores, TE_mean
