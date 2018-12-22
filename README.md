# PyCausality

PyCausality is a Python package enabling the rapid and flexible calculation of causality between multiple time series.



# Installation

The package can be downloaded from PyPi using pip. We recommend using Anaconda for creating virtual environments https://www.anaconda.com/download. In your chosen conda environment, or globally, you can then install PyCausality using:

`pip install PyCausality`

# Usage

It is recommended to import the package object into your research code using:

`from PyCausality.TransferEntropy import *`

Which will give you access to:

- `LaggedTimeSeries` objects
- `TransferEntropy` objects
- `AutoBins` objects

With time series data stored in a pandas DataFrame, perform predictive analysis by invoking the TransferEntropy class. Using an example of daily stock price data, to calculate the information transfer in each direction between MSFT and AAPL, you would perform this as follows:

    ## Initialise Object to Calculate Transfer Entropy
    TE = TransferEntropy(   DF = DataFrame,
                            endog = 'MSFT',     # Dependent Variable
                            exog = 'AAPL',      # Independent Variable
                            lag = 2,
                            window_size = {'MS':6},
                            window_stride = {'W':2}
                            )

    ## Calculate TE using KDE
    TE.nonlinear_TE(pdf_estimator = 'kernel', n_shuffles=100)

    ## Display TE_XY, TE_YX and significance values
    print(TE.results)

This example shows windowing of the time series, to reduce the risk of non-stationarity impacting the significance of the causality, and to provide multiple views on the overall timeseries (some periods may express large dependencies which could be cancelled out by earlier periods). 

To make use of automatic binning, you may supply the data you would like to bin to the AutoBins class, and generate bins as follows:

    Auto = AutoBins(df = Data, lag = 2)

    knuth_bins  =    Auto.knuth_bins()
    MIC_bins    =    Auto.MIC_bins()
    sigma_bins  =    Auto.sigma_bins()
    


These bins are returned in the expected format (dict of lists) for direct passage into the TransferEntropy class.

# Examples

PyCausality provides a number of fully worked-through examples, for model validation and to help users understand the syntax and usage. These use synthetic data common to the literature (e.g. random walks), and can be found in the PyCausality/Examples folder.

Plots produced from running the code are also provided, and are saved in PyCausality/Examples/Plots.


# Tests

The package comes with both unit and integration tests, designed to be run using nose, the Python testing framework:  http://nose.readthedocs.io/en/latest/index.html 

To run unit tests, in the installation root directory run:

`nosetests PyCausality.Testing.unit_tests`

To run integration tests, in the installation root directory run:

`nosetests PyCausality.Testing.integration_tests`

Alternatively,all tests can be run at once by simply running:

`nosetests`


# Contribute

Contributions to PyCausality are actively welcomed. 

Planned features for future releases include:

- Multiple Endogenous Variables
    - Comparing (useful for Conditional TE - conditioning on extra side-information)
- New automatic binning techniques
    - z-score_bins() method for generating bins which maximise significance
    - knn_bins() method for variable-width bins corresponding to equal samples per bin
- Extending automatic binning to higher dimensions (currently only sigma_bins() accepts >2D)
- Inclusion of other causality detection  techniques

Additional functionality must be accompanied with thorough unit tests, and any PRs must be shown to pass all unit tests before being accepted.


## References

The study of information theoretic techniques for predictive analytics is sizeable and growing quickly. It is therefore not feasible to provide a comprehensive list of references informing the development of the PyCausality package. 

However, a solid overview of Transfer Entropy and the key techniques underlying this package will be gleaned from:

- Boba, P., Bollmann, D., Schoepe, D., Wester, N., Wiesel, J., & Hamacher, K. (2015). Efficient computation and statistical assessment of transfer entropy. Frontiers in Physics, 3, 10.


- Bossomaier, T., Barnett, L., Harré, M., & Lizier, J. T. (2016). An introduction to transfer entropy. Springer: Berlin, Germany.


- Marschinski, R., & Kantz, H. (2002). Analysing the information flow between financial time series. The European Physical Journal B-Condensed Matter and Complex Systems, 30(2), 275-281.

- Knuth, K. H. (2006). Optimal data-based binning for histograms. arXiv preprint physics/0605197.


- San Liang, X. (2014). Unraveling the cause-effect relation between time series. Physical Review E, 90(5), 052150.


- Schreiber, T. (2000). Measuring information transfer. Physical review letters, 85(2), 461.

- Zaremba, A., & Aste, T. (2014). Measures of causality in complex datasets with application to financial data. Entropy, 16(4), 2309-2349.