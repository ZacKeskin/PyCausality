import numpy as np


def shuffle_series(DF, only=None):
    """Function to return time series shuffled rowwise along each desired column.

    Each column is shuffled independently, removing the temporal relationship.

    This is to calculate Z-score and Z*-score. See P. Boba et al (2015)

    Calculated using:       df.apply(np.random.permutation)

    Arguments:
        df              -   (DataFrame) Time series data
        only            -   (list)      Fieldnames to shuffle. If none, all columns shuffled

    Returns:
        df_shuffled     -   (DataFrame) Time series shuffled along desired columns

    """
    if only is not None:
        shuffled_DF = DF.copy()
        for col in only:
            series = DF.loc[:, col].to_frame()
            shuffled_DF[col] = series.apply(np.random.permutation)
    else:
        shuffled_DF = DF.apply(np.random.permutation)

    return shuffled_DF
