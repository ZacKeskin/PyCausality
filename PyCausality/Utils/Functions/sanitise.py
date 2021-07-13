import pandas as pd


def sanitise(df):
    """Function to convert DataFrame-like objects into pandas DataFrames.

    Args:
        df          -        Data in pd.Series or pd.DataFrame format

    Returns:
        df          -        Data as pandas DataFrame

    """
    # Ensure data is in DataFrame form
    if isinstance(df, pd.DataFrame):
        df = df
    elif isinstance(df, pd.Series):
        df = df.to_frame()
    else:
        raise ValueError(
            "Data passed as %s Please ensure your data is stored as a Pandas DataFrame"
            % (str(type(df)))
        )

    return df
