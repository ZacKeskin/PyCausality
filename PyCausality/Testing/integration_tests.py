import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from PyCausality.TransferEntropy import TransferEntropy, LaggedTimeSeries, AutoBins
from PyCausality.Testing.Test_Utils.Time_Series_Generate import *
from PyCausality.Testing.Test_Utils.Integration_Fixtures import linear_tests, nonlinear_tests


def calculate_TE(fixture, method):
    # Unpack fixtures data
    test    =   fixture.get('name')
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
    estimator = fixture.get('estimator')
    assertion = fixture.get('assertion')
    bandwidth = fixture.get('bandwidth')
    n_shuffles = fixture.get('shuffles')


    ## Generate Test Time Series Data
    DF = coupled_random_walks(  S1, S2, T, N, mu1, mu2, 
                                sigma1, sigma2, alpha, 0, lag,
                                seed)

    ## Take log difference of each series:
    DF['S1'] = DF['S1'].pct_change()
    DF['S2'] = DF['S2'].pct_change()
    DF = DF.iloc[1:]

    print(test)
    print(DF.head(6))    
    ## Initialise TE object
    causality = TransferEntropy(DF = DF,
                                endog = 'S2',          # Dependent Variable
                                exog = 'S1',           # Independent Variable
                                lag = lag
                                )

    if method == 'linear':
        (TE_XY, TE_YX) = causality.linear_TE(n_shuffles= n_shuffles)
        (p_XY, p_YX) = (causality.results['p_value_linear_XY'].iloc[0], causality.results['p_value_linear_YX'].iloc[0])
        (Z_XY, Z_YX) = (causality.results['z_score_linear_XY'].iloc[0], causality.results['z_score_linear_YX'].iloc[0])
    else:
        (TE_XY, TE_YX) = causality.nonlinear_TE(pdf_estimator = estimator,
                                                bins = bins,
                                                bandwidth= bandwidth,
                                                n_shuffles= n_shuffles)
        (p_XY, p_YX) = (causality.results['p_value_XY'].iloc[0], causality.results[['p_value_YX']].iloc[0])
        (Z_XY, Z_YX) = (causality.results['z_score_XY'].values, causality.results[['z_score_YX']].values)

    
    ## Evaluate the assertion contained within the test
    print('assertion', assertion)
    print(eval(assertion))

    print('TE_XY:' + str(TE_XY), 'TE_YX:' + str(TE_YX))
    print('p_XY:' + str(p_XY), 'p_YX:' + str(p_YX))
    print('z_score_XY:' + str(Z_XY), 'z_score_YX:' + str(Z_YX))
    assert eval(assertion)


def test_generator():

    ## Test linear_TE()
    for i, (fixture_name,fixture) in enumerate(linear_tests.items()):  
        # Pass each structure to test function
        yield calculate_TE, fixture, 'linear'

    ## Test nonlinear_TE()
    for i, (fixture_name,fixture) in enumerate(nonlinear_tests.items()):  
        # Pass each structure to test function
        yield calculate_TE, fixture, 'nonlinear'
    
    
test_generator()