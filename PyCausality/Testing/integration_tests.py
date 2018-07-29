import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from PyCausality.TransferEntropy import TransferEntropy, LaggedTimeSeries, AutoBins
from PyCausality.Testing.Test_Utils.Time_Series_Generate import *
from PyCausality.Testing.Test_Utils.Integration_Fixtures import *


def nonlinear_TE(fixture):
    # Unpack fixtures data
    bins    =   fixture.get('bins')
    S1      =   fixture.get('S1')
    S2      =   fixture.get('S2')
    T       =   fixture.get('T')
    N       =   fixture.get('N')
    mu1     =   fixture.get('mu1')
    mu2     =   fixture.get('mu2')
    sigma1  =   fixture.get('sigma1')
    sigma2  =   fixture.get('sigma2')
    alpha   =   fixture.get('alpha')
    lag     =   fixture.get('lag')
    seed    =   fixture.get('seed')
    TE      =   fixture.get('expected_TE')
    tol     =   fixture.get('tolerance')

    ## Generate Test Time Series Data
    DF = coupled_random_walks(  S1, S2, T, N, mu1, mu2, 
                                sigma1, sigma2, alpha, lag,
                                seed)
    ## Initialise TE object
    causality = TransferEntropy(DF = DF,
                                endog = 'S2',          # Dependent Variable
                                exog = 'S1',           # Independent Variable
                                lag = lag
                                )
    ## Calculate NonLinear TE
    (TE_XY, TE_YX) = causality.nonlinear_TE(pdf_estimator = 'histogram',
                                            bins = bins,
                                            n_shuffles= 100)

    ## Ensure TE X_Y is at least as great as expected (must always be positive)
    assert TE_XY > TE


def test_generator():

    for i, (fixture_name,fixture) in enumerate(fixtures.items()):  
        # Pass each structure to test function
        yield nonlinear_TE, fixture
