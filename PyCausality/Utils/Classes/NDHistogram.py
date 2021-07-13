import sys
import warnings

import numpy as np
from numpy import ma
import pandas as pd
from scipy.special import gammaln
from scipy.stats.mstats import mquantiles

from ..Functions.sanitise import sanitise


class NDHistogram:
    """Custom histogram class wrapping the default numpy implementations (np.histogram, np.histogramdd).

    This allows for dimension-agnostic histogram calculations, custom auto-binning and
    associated data and methods to be stored for each object (e.g. Probability Density etc.)
    """

    def __init__(self, df, bins=None, max_bins=15):
        """init.

        Arguments:
            df          -   DataFrame passed through from the TransferEntropy class
            bins        -   Bin edges passed through from the TransferEntropy class
            max_bins    -   Number of bins per each dimension passed through from the TransferEntropy class

        Returns:
            self.pdf    -   This is an N-dimensional Probability Density Function, stored as a
                            Numpy histogram, representing the proportion of samples in each bin.

        """
        df = sanitise(df)
        self.df = df.reindex(columns=sorted(df.columns))  # Sort axes by name
        self.max_bins = max_bins
        self.axes = list(self.df.columns.values)
        self.bins = bins
        self.n_dims = len(self.axes)

        # Bins must match number and order of dimensions
        if self.bins is None:
            AB = AutoBins(self.df)
            self.bins = AB.sigma_bins(max_bins=max_bins)
        elif set(self.bins.keys()) != set(self.axes):
            warnings.warn("Incompatible bins provided - defaulting to sigma bins")
            AB = AutoBins(self.df)
            self.bins = AB.sigma_bins(max_bins=max_bins)

        ordered_bins = [sorted(self.bins[key]) for key in sorted(self.bins.keys())]

        # Create ND histogram (np.histogramdd doesn't scale down to 1D)
        if self.n_dims == 1:
            self.Hist, self.Dedges = np.histogram(
                self.df.values, bins=ordered_bins[0], normed=False
            )
        elif self.n_dims > 1:
            self.Hist, self.Dedges = np.histogramdd(
                self.df.values, bins=ordered_bins, normed=False
            )

        # Empirical Probability Density Function
        if self.Hist.sum() == 0:
            print(self.Hist.shape)

            with pd.option_context("display.max_rows", None, "display.max_columns", 3):
                print(self.df.tail(40))

            sys.exit(
                "User-defined histogram is empty. Check bins or increase data points"
            )
        else:
            self.pdf = self.Hist / self.Hist.sum()
            self._set_entropy_(self.pdf)

    def _set_entropy_(self, pdf):
        """Set entropy.

        Arguments:
            pdf   -   Probabiiity Density Function; this is calculated using the N-dimensional histogram above.

        Returns:
            n/a

        Sets entropy for marginal distributions: H(X), H(Y) etc. as well as joint entropy H(X,Y)

        """
        # Prepare empty dict for marginal entropies along each dimension
        self.H = {}

        if self.n_dims > 1:

            # Joint entropy H(X,Y) = -sum(pdf(x,y) * log(pdf(x,y)))
            self.H_joint = -np.sum(
                pdf * ma.log2(pdf).filled(0)
            )  # Use masking to replace log(0) with 0

            # Single entropy for each dimension H(X) = -sum(pdf(x) * log(pdf(x)))
            for a, axis_name in enumerate(self.axes):
                self.H[axis_name] = -np.sum(
                    pdf.sum(axis=a) * ma.log2(pdf.sum(axis=a)).filled(0)
                )  # Use masking to replace log(0) with 0
        else:
            # Joint entropy and single entropy are the same
            self.H_joint = -np.sum(pdf * ma.log2(pdf).filled(0))
            self.H[self.df.columns[0]] = self.H_joint


class AutoBins:
    """Prototyping class for generating data-driven binning.

    Handles lagged time series, so only DF[X(t), Y(t)] required.

    """

    def __init__(self, df, lag=None):
        """init.

        Args:
            df      -   (DateFrame) Time series data to classify into bins
            lag     -   (float)     Lag for data to provided bins for lagged columns also

        Returns:
            n/a

        """
        # Ensure data is in DataFrame form
        self.df = sanitise(df)
        self.axes = self.df.columns.values
        self.ndims = len(self.axes)
        self.N = len(self.df)
        self.lag = lag

    def __extend_bins__(self, bins):
        """Function to generate bins for lagged time series not present in self.df.

        Args:
            bins    -   (Dict of List)  Bins edges calculated by some AutoBins.method()

        Returns:
            bins    -   (Dict of lists) Bin edges keyed by column name

        """
        self.max_lag_only = True  # still temporary until we kill this

        # Handle lagging for bins, and calculate default bins where edges are not provided
        if self.max_lag_only is True:
            bins.update(
                {
                    fieldname + "_lag" + str(self.lag): edges
                    for (fieldname, edges) in bins.items()
                }
            )
        else:
            bins.update(
                {
                    fieldname + "_lag" + str(t): edges
                    for (fieldname, edges) in bins.items()
                    for t in range(self.lag)
                }
            )

        return bins

    def MIC_bins(self, max_bins=15):
        """MIC method to find optimal bin widths in each dimension.

        Uses a naive search to maximise the mutual information divided by number of bins.
        Only accepts data with two dimensions [X(t),Y(t)]. We increase the n_bins parameter
        in each dimension, and take the bins which result in the greatest Maximum
        Information Coefficient (MIC)

        (Note that this is restricted to equal-width bins only.)

        Defined:            MIC = I(X,Y)/ max(n_bins)
                            edges = {Y:[a,b,c,d], Y-t:[a,b,c,d], X-t:[e,f,g]},
                            n_bins = [bx,by]

        Calculated using:   argmax { I(X,Y)/ max(n_bins) }

        Args:
            max_bins        -   (int)       The maximum allowed bins in each dimension

        Returns:
            opt_edges       -   (dict)      The optimal bin-edges for pdf estimation
                                            using the histogram method, keyed by df column names
                                            All bins equal-width.

        """
        if len(self.df.columns.values) > 2:
            raise ValueError(
                "Too many columns provided in DataFrame. MIC_bins only accepts 2 columns (no lagged columns)"
            )

        min_bins = 3

        # Initialise array to store MIC values
        MICs = np.zeros(shape=[1 + max_bins - min_bins, 1 + max_bins - min_bins])

        # Loop over each dimension
        for b_x in range(min_bins, max_bins + 1):

            for b_y in range(min_bins, max_bins + 1):

                # Update parameters
                n_bins = [b_x, b_y]

                # Update dict of bin edges
                edges = {
                    dim: list(
                        np.linspace(
                            self.df[dim].min(), self.df[dim].max(), n_bins[i] + 1
                        )
                    )
                    for i, dim in enumerate(self.df.columns.values)
                }

                # Calculate Maximum Information Coefficient
                HDE = NDHistogram(self.df, edges)

                I_xy = sum([H for H in HDE.H.values()]) - HDE.H_joint

                MIC = I_xy / np.log2(np.min(n_bins))

                MICs[b_x - min_bins][b_y - min_bins] = MIC

        # Get Optimal b_x, b_y values
        n_bins[0] = np.where(MICs == np.max(MICs))[0][0] + min_bins
        n_bins[1] = np.where(MICs == np.max(MICs))[1][0] + min_bins

        bins = {
            dim: list(
                np.linspace(self.df[dim].min(), self.df[dim].max(), n_bins[i] + 1)
            )
            for i, dim in enumerate(self.df.columns.values)
        }

        if self.lag is not None:
            bins = self.__extend_bins__(bins)

        # Return the optimal bin-edges
        return bins

    def knuth_bins(self, max_bins=15):
        """Knuth method to find optimal bin widths in each dimension.

        Uses a naive search to maximise the log-likelihood given data.
        Only accepts data with two dimensions [X(t),Y(t)].
        Derived from Matlab code provided in Knuth (2013):  https://arxiv.org/pdf/physics/0605197.pdf

        (Note that this is restricted to equal-width bins only.)

        Args:
            max_bins        -   (int)       The maximum allowed bins in each dimension

        Returns:
            bins            -   (dict)      The optimal bin-edges for pdf estimation
                                            using the histogram method, keyed by df column names
                                            All bins equal-width.

        """
        if len(self.df.columns.values) > 2:
            raise ValueError(
                "Too many columns provided in DataFrame. knuth_bins only accepts 2 columns (no lagged columns)"
            )

        min_bins = 3

        # Initialise array to store MIC values
        log_probabilities = np.zeros(
            shape=[1 + max_bins - min_bins, 1 + max_bins - min_bins]
        )

        # Loop over each dimension
        for b_x in range(min_bins, max_bins + 1):

            for b_y in range(min_bins, max_bins + 1):

                # Update parameters
                Ms = [b_x, b_y]

                # Update dict of bin edges
                bins = {
                    dim: list(
                        np.linspace(self.df[dim].min(), self.df[dim].max(), Ms[i] + 1)
                    )
                    for i, dim in enumerate(self.df.columns.values)
                }

                # Calculate Maximum log Posterior

                # Create N-d histogram to count number per bin
                HDE = NDHistogram(self.df, bins)
                nk = HDE.Hist

                # M = number of bins in total =  Mx * My * Mz ... etc.
                M = np.prod(Ms)

                log_prob = (
                    self.N * np.log(M)
                    + gammaln(0.5 * M)
                    - M * gammaln(0.5)
                    - gammaln(self.N + 0.5 * M)
                    + np.sum(gammaln(nk.ravel() + 0.5))
                )

                log_probabilities[b_x - min_bins][b_y - min_bins] = log_prob

        # Get Optimal b_x, b_y values
        Ms[0] = np.where(log_probabilities == np.max(log_probabilities))[0] + min_bins
        Ms[1] = np.where(log_probabilities == np.max(log_probabilities))[1] + min_bins

        bins = {
            dim: list(np.linspace(self.df[dim].min(), self.df[dim].max(), Ms[i] + 1))
            for i, dim in enumerate(self.df.columns.values)
        }

        if self.lag is not None:
            bins = self.__extend_bins__(bins)

        # Return the optimal bin-edges
        return bins

    def sigma_bins(self, max_bins=15):
        """Returns bins for N-dimensional data, using standard deviation binning.

        Each bin is one S.D in width, with bins centered on the mean.
        Where outliers exist beyond the maximum number of SDs dictated by the max_bins
        parameter, the bins are extended to minimum/maximum values to ensure all data
        points are captured. This may mean larger bins in the tails, and up to two bins
        greater than the max_bins parameter suggests in total (in the unlikely case of
        huge outliers on both sides).

        Args:
            max_bins        -   (int)       The maximum allowed bins in each dimension\

        Returns:
            bins            -   (dict)      The optimal bin-edges for pdf estimation
                                            using the histogram method, keyed by df column names

        """
        bins = {
            k: [
                np.mean(v) - int(max_bins / 2) * np.std(v) + i * np.std(v)
                for i in range(max_bins + 1)
            ]
            for (k, v) in self.df.iteritems()  # noqa
        }  # Note: same as:  self.df.to_dict('list').items()}

        # Since some outliers can be missed, extend bins if any points are not yet captured
        [
            bins[k].append(self.df[k].min())
            for k in self.df.keys()
            if self.df[k].min() < min(bins[k])
        ]
        [
            bins[k].append(self.df[k].max())
            for k in self.df.keys()
            if self.df[k].max() > max(bins[k])
        ]

        if self.lag is not None:
            bins = self.__extend_bins__(bins)
        return bins

    def equiprobable_bins(self, max_bins=15):
        """Returns bins for N-dimensional data, such that each bin should contain equal numbers of samples.

        *** Note that due to SciPy's mquantiles() functional design, the equipartion is not strictly true -
        it operates independently on the marginals, and so with large bin numbers there are usually
        significant discrepancies from desired behaviour. Fortunately, for TE we find equipartioning is
        extremely beneficial, so we find good accuracy with small bin counts ***

        Args:
            max_bins        -   (int)       The number of bins in each dimension

        Returns:
            bins            -   (dict)      The calculated bin-edges for pdf estimation
                                            using the histogram method, keyed by df column names

        """
        quantiles = np.array([i / max_bins for i in range(0, max_bins + 1)])
        bins = dict(
            zip(self.axes, mquantiles(a=self.df, prob=quantiles, axis=0).T.tolist())
        )

        # Remove_duplicates
        bins = {k: sorted(set(bins[k])) for (k, v) in bins.items()}

        if self.lag is not None:
            bins = self.__extend_bins__(bins)
        return bins
