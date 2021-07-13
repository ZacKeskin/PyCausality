from dateutil.relativedelta import relativedelta
import pandas as pd

from .Utils.Functions.sanitise import sanitise


class LaggedTimeSeries:
    """Custom wrapper class for pandas DataFrames for performing predictive analysis.

    Generates lagged time series and performs custom windowing over datetime indexes

    """

    def __init__(
        self, df, lag=None, max_lag_only=True, window_size=None, window_stride=None
    ):
        """init.

        Arguments:
            df              -   Pandas DataFrame object of N columns. Must be indexed as an increasing
                                time series (i.e. past-to-future), with equal timesteps between each row
            lags            -   The number of steps to be included. Each increase in Lags will result
                                in N additional columns, where N is the number of columns in the original
                                dataframe. It will also remove the first N rows.
            max_lag_only    -   Defines whether the returned dataframe contains all lagged timeseries up to
                                and including the defined lag, or only the time series equal to this lag value
            window_size     -   Dict containing key-value pairs only from within: {'YS':0,'MS':0,'D':0,'H':0,'min':0,'S':0,'ms':0}
                                Describes the desired size of each window, provided the data is indexed with datetime type. Leave as
                                None for no windowing. Units follow http://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases
            window_stride   -   Dict containing key-value pairs only from within: {'YS':0,'MS':0,'D':0,'H':0,'min':0,'S':0,'ms':0}
                                Describes the size of the step between consecutive windows, provided the data is indexed with datetime type. Leave as
                                None for no windowing. Units follow http://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases

        Returns:
            n/a

        """
        self.df = sanitise(df)
        self.axes = list(self.df.columns.values)  # Variable names

        self.max_lag_only = max_lag_only
        if lag is not None:
            self.t = lag
            self.df = self.__apply_lags__()

        if window_size is not None and window_stride is not None:
            self.has_windows = True
            self.__apply_windows__(window_size, window_stride)
        else:
            self.has_windows = False

    def __apply_lags__(self):
        """Apply lags.

        Args:
            n/a

        Returns:
            new_df.iloc[self.t:]    -   This is a new dataframe containing the original columns and
                                        all lagged columns. Note that the first few rows (equal to self.lag) will
                                        be removed from the top, since lagged values are of coursenot available
                                        for these indexes.

        """
        # Create a new dataframe to maintain the new data, dropping rows with NaN
        new_df = self.df.copy(deep=True).dropna()

        # Create new column with lagged timeseries for each variable
        col_names = self.df.columns.values.tolist()

        # If the user wants to only consider the time series lagged by the
        # maximum number specified or by every series up to an including the maximum lag:
        if self.max_lag_only is True:
            for col_name in col_names:
                new_df[col_name + "_lag" + str(self.t)] = self.df[col_name].shift(
                    self.t
                )

        elif self.max_lag_only is False:
            for col_name in col_names:
                for t in range(1, self.t + 1):
                    new_df[col_name + "_lag" + str(t)] = self.df[col_name].shift(t)
        else:
            raise ValueError("Error")

        # Drop the first t rows, which now contain NaN
        return new_df.iloc[self.t :]

    def __apply_windows__(self, window_size, window_stride):
        """Apply windows.

        Args:
            window_size      -   Dict passed from self.__init__
            window_stride    -   Dict passed from self.__init__

        Returns:
            n/a              -   Sets the daterange for the self.windows property to iterate along

        """
        self.window_size = {"YS": 0, "MS": 0, "D": 0, "H": 0, "min": 0, "S": 0, "ms": 0}
        self.window_stride = {
            "YS": 0,
            "MS": 0,
            "D": 0,
            "H": 0,
            "min": 0,
            "S": 0,
            "ms": 0,
        }

        self.window_stride.update(window_stride)
        self.window_size.update(window_size)
        freq = ""
        daterangefreq = freq.join(
            [str(v) + str(k) for (k, v) in self.window_stride.items() if v != 0]
        )
        self.daterange = pd.date_range(
            self.df.index.min(), self.df.index.max(), freq=daterangefreq
        )

    def date_diff(self, window_size):
        """Date diff.

        Args:
            window_size     -    Dict passed from self.windows function

        Returns:
            start_date      -    The start date of the proposed window
            end_date        -    The end date of the proposed window

        This function is TBC - proposed due to possible duplication of the relativedelta usage in self.windows and self.headstart

        """
        pass

    @property
    def windows(self):
        """Windows.

        Args:
            n/a

        Returns:
            windows         -   Generator defining a pandas DataFrame for each window of the data.
                                Usage like:   [window for window in LaggedTimeSeries.windows]

        """
        if self.has_windows is False:
            return self.df  # noqa

        # Loop Over TimeSeries Range
        for _i, dt in enumerate(self.daterange):

            # Ensure Each Division Contains Required Number of Months
            if (
                dt
                - relativedelta(
                    years=self.window_size["YS"],
                    months=self.window_size["MS"],
                    days=self.window_size["D"],
                    hours=self.window_size["H"],
                    minutes=self.window_size["min"],
                    seconds=self.window_size["S"],
                    microseconds=self.window_size["ms"],
                )
                >= self.df.index.min()
            ):

                # Create Window
                yield self.df.loc[
                    (
                        dt
                        - relativedelta(
                            years=self.window_size["YS"],
                            months=self.window_size["MS"],
                            days=self.window_size["D"],
                            hours=self.window_size["H"],
                            minutes=self.window_size["min"],
                            seconds=self.window_size["S"],
                            microseconds=self.window_size["ms"],
                        )
                    ) : dt
                ]

    @property
    def headstart(self):
        """Headstart.

        Args:
            n/a

        Returns:
            len(windows)    -   The number of windows which would have start dates before the desired date range.
                                Used in TransferEntropy class to slice off incomplete windows.

        """
        windows = [
            i
            for i, dt in enumerate(self.daterange)
            if dt
            - relativedelta(
                years=self.window_size["YS"],
                months=self.window_size["MS"],
                days=self.window_size["D"],
                hours=self.window_size["H"],
                minutes=self.window_size["min"],
                seconds=self.window_size["S"],
                microseconds=self.window_size["ms"],
            )
            < self.df.index.min()
        ]
        # i.e. count from the first window which falls entirely after the earliest date
        return len(windows)
