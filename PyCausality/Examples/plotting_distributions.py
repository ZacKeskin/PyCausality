"""
    This example demonstrates how to plot 3D density distributions of your time series data using PyCausality. We find 
    this is a useful exploratory technique for quickly identifying possible bivariate relationships, and predicting
    the effectiveness of estimation techniques over the data. This is epecially true for considering manual 
    parameter selection, as the results from different bandwidths etc. can quickly be assessed.

    Note that the functionality is currently limited to plotting bivariate relationships, i.e. 2D histograms with a z-xis projection.

    Also note that the z-axis represents probability - NOT probability density. For equal-sized bins, the visual
    effect is the same, but if variable-width bins are provided then the VOLUME of each bin's 3D projection does not 
    reflect the probabilty of falling within the bin; it remains the HEIGHT of the bin which shows this.

    This efect is clearly seen when using AutoBins().equiprobable_bins.
"""

import os # for saving plots
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

from PyCausality.TransferEntropy import *
from PyCausality.Testing.Test_Utils.Time_Series_Generate import *


## Set Parameters
LAG = 1
SIMILARITY = 0            # Choose value between 0 (independent) to 1 (exact lagged value)
AUTOSIMILARITY = 0        # Choose value between 0 (independent) to 1 (exact lagged value)
SEED = None                 # Change pseudo-RNG seed; useful for repeat results & comparing bin choices
DATA_POINTS = 500
BANDWIDTH = 0.8           # Choose bandwidth for KDE
N_BINS = 15


## Create Coupled GBM Time Series
DF = coupled_random_walks(  S1 = 100, S2 = 100, T = 0.1, 
                            N = DATA_POINTS, mu1 = 0, mu2 = 0, 
                            sigma1 = 0.1, sigma2 = 0.1, 
                            alpha = SIMILARITY, epsilon = AUTOSIMILARITY,
                            lag = LAG, seed=SEED)


# Use Differenced data
DF['S2'] = DF['S2'].pct_change()
DF['S1'] = DF['S1'].pct_change()
DF = DF.iloc[LAG:]

## Define equal-width (per dimension) histogram bins
bins = {'S1':np.linspace(DF['S1'].min(),DF['S1'].max(),N_BINS),
         'S2':np.linspace(DF['S2'].min(),DF['S2'].max(),N_BINS)}

## Generate plots of the probability distribution. These are returned as axes objects
hist_axis = plot_pdf(df=DF[['S1','S2']],estimator='histogram', bins=bins, cmap='viridis',show=False)
kernel_axis = plot_pdf(df=DF[['S1','S2']],estimator='kernel', bandwidth=BANDWIDTH, cmap='viridis', show=False)



## Strip the returned axes objects and apply them to subplots
fig, axes = plt.subplots(nrows=0,ncols=0,figsize=(8,5.5))

kernel_axis.figure = fig
hist_axis.figure= fig
AXES = [hist_axis, kernel_axis]
fig.axes.append(AXES)
[fig.add_axes(ax) for ax in AXES]

## Position subplots
pos1 = Bbox([[0.5,1.2],[1.5,2]])
pos2 = Bbox([[0.5,0.1],[1.5,0.9]])
hist_axis.set_position(pos1) 
kernel_axis.set_position(pos2)

plt.suptitle("Demonstrating PyCausality's Probablity Plotting Functionality")
plt.savefig(os.path.join(os.getcwd(),'PyCausality', 'Examples','Plots','Plotting_Distributions.png'))
plt.show()

