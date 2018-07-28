import os
import matplotlib.pyplot as plt

from PyCausality.TransferEntropy import *
from PyCausality.Testing.Test_Utils.Time_Series_Generate import *


## Set Parameters
LAG = 1
SIMILARITY = 0.8          # Choose value between 0 (independent) to 1 (exact lagged value)
AUTOSIMILARITY = 0.2      # Choose value between 0 (independent) to 1 (exact lagged value)
SEED = 11                 # Change pseudo-RNG seed; useful for repeat results & comparing bin choices
DATA_POINTS = 400
BANDWIDTH = 0.8         # Choose bandwidth for KDE

## Create Highly Coupled Time Series
DF = coupled_random_walks(  S1 = 100, S2 = 95, T = 1, 
                            N = DATA_POINTS, mu1 = 0.1, mu2 = 0.2, 
                            sigma1 = 0.8, sigma2 = 0.3, 
                            alpha = SIMILARITY, epsilon = AUTOSIMILARITY,
                            lag = LAG, seed=SEED)


## (Optional) Use Differenced data
stocks = DF
DF = LaggedTimeSeries(DF,LAG).df
DF = np.log(DF).diff()
DF = DF.iloc[LAG:]


## Initialise TE object
causality = TransferEntropy(DF = DF,
                            endog = 'S2',          # Dependent Variable
                            exog = 'S1',           # Independent Variable
                            lag = LAG)


## Uncomment to Choose Auto Bins
edges = {'S1':[-5, -2.5, -1, -0.5, 0, 0.5, 1, 2.5, 5],'S2':[-5, -2.5, -1, -0.5, 0, 0.5, 1, 2.5, 5]}
#auto = AutoBins(DF[['S1','S2']])
#edges = auto.knuth_bins(max_bins=5)

## Calculate NonLinear TE
#(TE_XY, TE_YX) = causality.nonlinear_TE(df=DF, pdf_estimator = 'kernel', n_shuffles=100, bandwidth=BANDWIDTH)
(TE_XY, TE_YX) = causality.nonlinear_TE(df=DF, pdf_estimator = 'histogram', bins= edges, n_shuffles=100)

print("Calculated TE: \t\t",TE_XY,'\t', TE_YX)

## Plot Time Series and Calculated PDF
stocks.plot()
plt.xlabel('S1')
plt.ylabel('S2')
plt.title('Movements in Time Series S1 and S2')


plot_pdf(DF[['S1','S2']])
plt.show()