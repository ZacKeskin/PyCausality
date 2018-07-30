import numpy as numpy
import pandas as pd
from scipy.stats import skewnorm, entropy
import os

from nose.tools import assert_almost_equal, assert_raises, raises

from PyCausality.TransferEntropy import *
from PyCausality.Testing.Test_Utils.Time_Series_Generate import *

import matplotlib.pyplot as plt


##------------   Classes ---------------##

def test_NDHistogram():
    """
        Test function to ensure that the custom NDHistogram class correctly captures all
        data from 1 to 4 dimensions. Entropy must correspond to get_entropy() function.

    """
    data = pd.read_csv(os.path.join(os.getcwd(),'PyCausality','Testing',
                                    'Test_Utils','test_data.csv'))

    # 1D Data
    Hist1D =  NDHistogram(df=data[['S1']])
    assert len(data) == np.sum(Hist1D.Hist)

    # 2D Data
    Hist2D =  NDHistogram(df=data[['S1','S2']])
    assert len(data) == np.sum(Hist2D.Hist)

    # 3D Data
    Hist3D =  NDHistogram(df=data[['S1','S2','S3']])
    assert len(data) == np.sum(Hist3D.Hist)

    # 4D Data
    Hist4D =  NDHistogram(df=data[['S1','S2','S3','S4']])
    assert len(data) == np.sum(Hist4D.Hist)

    # Check sigma bins are correctly calculated when bins parameter is None
    #   (regression check against AutoBins.sigma_bins() )
    AB1 = AutoBins(df=data[['S1']])
    assert sorted(Hist1D.Dedges) == sorted(AB1.sigma_bins()['S1'])

    # Check entropy values correspond to test_get_entropy()


def test_AutoBins():
    """
        Test Function TBC
    """


def test_LaggedTimeSeries():
    """
        Test Function to ensure LaggedTimeSeries creates new lagged columns for each
        column passed in as a DataFrame
    """

    ## Import data
    data = pd.read_csv(os.path.join(os.getcwd(),'PyCausality','Testing',
                                    'Test_Utils','test_data.csv'))
    data['date'] = pd.to_datetime(data['date'],format="%d/%m/%Y")
    data.set_index('date', inplace=True)

    LAG = 1
    lagDF = LaggedTimeSeries(df=data ,lag=LAG).df

    ## Test that a new lagged column is created for each column in data
    
    for i, dim in enumerate(data.columns.values):
        ## Assert new column for each existing column
        assert len(lagDF.columns.values) == 2 * len(data.columns.values)

    for LAG in [1,2,3,4]:  
        lagDF = LaggedTimeSeries(df=data ,lag=LAG).df
        ## Test that the values are correctly lagged for each time series
        assert all(lagDF['S1_lag'+str(LAG)].values == data['S1'][:-LAG].values)
        assert all(lagDF['S2_lag'+str(LAG)].values == data['S2'][:-LAG].values)
        assert all(lagDF['S3_lag'+str(LAG)].values == data['S3'][:-LAG].values)
        assert all(lagDF['S4_lag'+str(LAG)].values == data['S4'][:-LAG].values)


    ## Test that windowing works as expected 

    # Note: must consider sizes at least on the scale of the Index
    window_sizes = [
        {'YS':0,'MS':0,'D':10,'H':0,'min':0,'S':0,'ms':0},       # Days
        {'YS':0,'MS':1,'D':0,'H':0,'min':0,'S':0,'ms':0},        # Months
        {'YS':1,'MS':0,'D':0,'H':0,'min':0,'S':0,'ms':0},        # Years
        {'YS':0,'MS':3,'D':14,'H':0,'min':0,'S':0,'ms':0},       # Months + Days  
    ]

    # Note: only one non-zero value is permitted for strides:
    window_strides = [
        {'YS':0,'MS':0,'D':7,'H':0,'min':0,'S':0,'ms':0},
        {'YS':0,'MS':3,'D':0,'H':0,'min':0,'S':0,'ms':0},
        {'YS':1,'MS':0,'D':0,'H':0,'min':0,'S':0,'ms':0},
        None
    ]

    for stride in window_strides:
        for size in window_sizes:

            lagDF = LaggedTimeSeries(df=data[['S1','S2']], lag=LAG, window_size=size, window_stride=stride)
            #print(lagDF.daterange)
            #[print(df.head()) for df in lagDF.windows]
            #[print(len(df)) for df in lagDF.windows]

            print('\n Number of Windows:', len(list(lagDF.windows)))
            try:
                print('Items per window:', len(next(lagDF.windows)))
            except:
                print('Items per window:', len(lagDF.df))

            ## ASSERTIONS TO FOLLOW HERE ##


    ## Test that the function handles DataFrames with fieldnames ending '_lag'
        # TBC

    ## Test that the function handles other data types (pd.series, np.ndarray)
        # TBC



##------------  Utils Functions ---------------##


def test_shuffle_series():
    """
        Test function to ensure Inference.shuffle_series() reorders data within columns.
        If parameter 'only' is specified, only that column should be shuffled; otherwise
        all columns must be shufffled.
    """
    DF = coupled_random_walks(  S1 = 100, S2 = 100, T = 10, 
                                N = 200, mu1 = 0.1, mu2 = 0.02, 
                                sigma1 = 0.01, sigma2 = 0.01, 
                                alpha = 0.1, epsilon=0, lag = 2)
    
    ### Shuffling both time series S1 & S2
    DF_shuffled = shuffle_series(DF)

    # Ensure rows are shuffled but not lost
    assert_almost_equal(np.mean(DF['S1']), np.mean(DF_shuffled['S1']))
    assert_almost_equal(np.mean(DF['S2']), np.mean(DF_shuffled['S2']))
    # Ensure rows are shuffled independently 
    assert not np.sum(DF['S1'].head(20)) == np.sum(DF_shuffled['S1'].head(20))
    assert not DF.head(10).equals(DF_shuffled.head(10))


    ### Shuffling only time series S1
    S1_shuffled = shuffle_series(DF, only=['S1'])

    # Ensure rows are shuffled but not lost
    assert_almost_equal(np.mean(DF['S1']), np.mean(S1_shuffled['S1']))
    assert_almost_equal(np.mean(DF['S2']), np.mean(S1_shuffled['S2']))

    # Ensure only S1 has been shuffled
    assert DF['S2'].head(10).equals(S1_shuffled['S2'].head(10))
    assert not DF['S1'].head(10).equals(S1_shuffled['S1'].head(10))

    ### Shuffling only time series S1
    S2_shuffled = shuffle_series(DF, only=['S2'])

    # Ensure rows are shuffled but not lost
    assert_almost_equal(np.mean(DF['S1']), np.mean(S1_shuffled['S1']))
    assert_almost_equal(np.mean(DF['S2']), np.mean(S1_shuffled['S2']))

    # Ensure only S2 has been shuffled
    assert DF['S1'].head(10).equals(S2_shuffled['S1'].head(10))
    assert not DF['S2'].head(10).equals(S2_shuffled['S2'].head(10))

def test_get_pdf():
    """
        Test Function to check that get_pdf() accurately models a probability density function.
        Tests both kernel and histogram density estimation on a set of randonly skewed normals, such that
        the average cumulative distribution function is approximately equal to 1/2 

    """
    ## Prepare list of CDF values
    cdfs_kde = []
    cdfs_hist = []

    ## Use np.random to generate skewed distributions, with different means and skews
    N = 1000
    sd = 0.5
    alphas = np.linspace(-5,5,10)   # Skews
    means = np.linspace(-5,5,10)    # Means

    for a in alphas:
        for mean in means:
            ## Generate N data points following this distribution
            data = skewnorm.rvs(size=N, a=a, loc=mean, scale=sd)
            TS = pd.DataFrame({'data':data})

            ## Calculate the PDF
            pdf_kde  = get_pdf(TS[['data']], estimator='kernel', gridpoints=20,bandwidth=0.6)
            pdf_hist = get_pdf(TS[['data']], estimator='histogram')
            

            ## Test that a pdf sums to 1 (to 8 d.p.)
            assert_almost_equal(pdf_hist.sum(),1.0,8)
            assert_almost_equal(pdf_kde.sum(),1.0,8)

            plt.plot(np.linspace(data.min(),data.max(),20),pdf_kde)
            ## Find probability of being < mean
            cdfs_kde.append(np.sum(pdf_kde[:10]))
            cdfs_hist.append(np.sum(pdf_kde[:10]))
    
    ## Over different skews, the average CDF at x=mean must equal 0.5 (to 1 d.p.)
    assert_almost_equal(np.average(cdfs_kde),0.5,1)
    assert_almost_equal(np.average(cdfs_hist),0.5,1)

    #plt.show() # It's a cool plot


def test_get_pdfND():
    """
        We make use of the analytical solution of Entropy of a multivariate normal to ensure that
        our KDE PDF estimation is accurate. Assuming test_joint_entropy passes, we know that our
        calculation of entropy from a multidimensional PDF is accurate. Then, we compare how close
        our entropy from the estimated PDF is, compared to the analytical solution. 
        
        We note that this is hugely dependent on the discretisation of our PDF over a grid,
        as well as the bandwidth used to perform the KDE in the first place. 
    """
    gridpoints = 5
    bandwidth = 0.25

    ## Generate Normally Distibuted data
    X = skewnorm.rvs(size=2000, a=0, loc=0, scale=1.5)
    Y = skewnorm.rvs(size=2000, a=0, loc=0, scale=0.1) 
    data = pd.DataFrame({'X':X, 'Y':Y})
    
    ## Calculate Theoretical Entropy
    covar = np.cov(data.values.T)
    H = 0.5 * np.log( np.linalg.det(2* np.pi * np.e * covar) )      # Note: in units of 'Nats'
    H = 2 ** (np.log(H) / np.log(2))    # Convert to 'Bits'

    print(H)
    print(get_entropy(data, gridpoints=gridpoints, bandwidth=bandwidth))
    
    assert False # Until tests completed

def test_joint_entropy():
    """
        Test that our implemented function to return the entropy corresponds 
        to Scipy's entropy method
    """
    gridpoints = 10 # for KDE estimation

    S = np.random.normal(0,0.1,10000)

    data = pd.DataFrame(S)

    ## pdf required for scipy.stats.entropy. Uses KDE to match with scipy. 
    pdf = get_pdf(data,gridpoints=gridpoints) # Valid providing test_get_pdf() passes   

    ## The estimated entropy should correspond to scipy's value (to 5 d.p.) 
    assert_almost_equal(get_entropy(data,gridpoints=gridpoints), entropy(pdf,base=2), 5)

def test_joint_entropyND():
    """
        Test that our implemented function to return the entropy corresponds 
        to Scipy's entropy method in multiple dimensions.
    """
    gridpoints = 10 # for KDE estimation

    ## Test 2D joint entropy:
    X = skewnorm.rvs(size=1000, a=-3, loc=0, scale=2)
    Y = skewnorm.rvs(size=1000, a=-3, loc=0, scale=2) 
    
    data = pd.DataFrame({'X':X, 'Y':Y})
    
    ## So this is valid if test_get_pdf passes   
    pdf = get_pdf(data,gridpoints=gridpoints) 

    ## The estimated entropy should correspond to scipy's value (to 5 d.p.) 
    assert_almost_equal(get_entropy(data,gridpoints=gridpoints), entropy(pdf.flatten(),base=2), 5)


def test_sanitise():
    """
        Test function to ensure user-defined time series data is sanitised to minimise
        the risk of avoidable errors
    """
    assert False

