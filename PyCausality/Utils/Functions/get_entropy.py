import numpy as np
from numpy import ma

from .pdf import get_pdf


def get_entropy(
    df, gridpoints=15, bandwidth=None, estimator="kernel", bins=None, covar=None
):
    """Function for calculating entropy from a probability mass.

    Args:
        df          -       (DataFrame) Samples over which to estimate density
        gridpoints  -       (int)       Number of gridpoints when integrating KDE over
                                        the domain. Used if estimator='kernel'
        bandwidth   -       (float)     Bandwidth for KDE (scalar multiple to covariance
                                        matrix). Used if estimator='kernel'
        estimator   -       (string)    'histogram' or 'kernel'
        bins        -       (Dict of lists) Bin edges for NDHistogram. Used if estimator
                                        = 'histogram'
        covar       -       (Numpy ndarray) Covariance matrix between dimensions of df.
                                        Used if estimator = 'kernel'

    Returns:
        entropy     -       (float)     Shannon entropy in bits

    """
    pdf = get_pdf(df, gridpoints, bandwidth, estimator, bins, covar)

    # log base 2 returns H(X) in bits
    return -np.sum(pdf * ma.log2(pdf).filled(0))
