import numpy as np

from .sanitise import sanitise
from ..Classes._kde_ import _kde_
from ..Classes.NDHistogram import NDHistogram


def get_pdf(df, gridpoints=None, bandwidth=None, estimator=None, bins=None, covar=None):
    """Function for non-parametric density estimation.

    Args:
        df          -       (DataFrame) Samples over which to estimate density
        gridpoints  -       (int)       Number of gridpoints when integrating KDE over
                                        the domain. Used if estimator='kernel'
        bandwidth   -       (float)     Bandwidth for KDE (scalar multiple to covariance
                                        matrix). Used if estimator='kernel'
        estimator   -       (string)    'histogram' or 'kernel'
        bins        -       (Dict of lists) Bin edges for NDHistogram. Used if estimator = 'histogram'
        covar       -       (Numpy ndarray) Covariance matrix between dimensions of df.
                                        Used if estimator = 'kernel'

    Returns:
        pdf         -       (Numpy ndarray) Probability of a sample being in a specific
                                        bin (technically a probability mass)

    """
    DF = sanitise(df)

    if estimator == "histogram":
        pdf = pdf_histogram(DF, bins)
    else:
        pdf = pdf_kde(DF, gridpoints, bandwidth, covar)
    return pdf


def pdf_kde(df, gridpoints=None, bandwidth=1, covar=None):
    """Function for non-parametric density estimation using Kernel Density Estimation.

    Args:
        df          -       (DataFrame) Samples over which to estimate density
        gridpoints  -       (int)       Number of gridpoints when integrating KDE over
                                        the domain. Used if estimator='kernel'
        bandwidth   -       (float)     Bandwidth for KDE (scalar multiple to covariance
                                        matrix).
        covar       -       (Numpy ndarray) Covariance matrix between dimensions of df.
                                        If None, these are calculated from df during the
                                        KDE analysis

    Returns:
        Z/Z.sum()   -       (Numpy ndarray) Probability of a sample being between
                                        specific gridpoints (technically a probability mass)

    """
    # Create Meshgrid to capture data
    if gridpoints is None:
        gridpoints = 20

    N = complex(gridpoints)

    slices = [slice(dim.min(), dim.max(), N) for dimname, dim in df.iteritems()]  # noqa
    grids = np.mgrid[slices]

    # Pass Meshgrid to Scipy Gaussian KDE to Estimate PDF
    positions = np.vstack([X.ravel() for X in grids])
    values = df.values.T
    kernel = _kde_(values, bw_method=bandwidth, covar=covar)
    Z = np.reshape(kernel(positions).T, grids[0].shape)

    # Normalise
    return Z / Z.sum()


def pdf_histogram(df, bins):
    """Function for non-parametric density estimation using N-Dimensional Histograms.

    Args:
        df            -       (DataFrame) Samples over which to estimate density
        bins          -       (Dict of lists) Bin edges for NDHistogram.

    Returns:
        histogram.pdf -       (Numpy ndarray) Probability of a sample being in a specific
                                    bin (technically a probability mass)

    """
    histogram = NDHistogram(df=df, bins=bins)
    return histogram.pdf
