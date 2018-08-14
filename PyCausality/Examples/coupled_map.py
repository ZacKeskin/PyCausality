"""
    Replicating the Method and Significance Test using the Logistic Coupled Map function

        This file is used for technical validation of the Transfer Entropy histogram method. 
        It replicates the analysis and results produced by Boba et al. 'Efficient computation 
        and statistical assessment of transfer entropy' https://doi.org/10.3389/fphy.2015.00010

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

from PyCausality.TransferEntropy import *
from PyCausality.Testing.Test_Utils.Time_Series_Generate import *

AUTO_BINS = True
N_OBSERVATIONS = 6    # Error Bars
N_SHUFFLES = 50       # Z-score

Ns = [2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16]
SIMILARITY = np.linspace(0,0.9,9)


## Prepare Plot
plt.style.use('seaborn')
fig, (TE_axis, Z_axis) = plt.subplots(figsize=(5, 8), nrows=2,ncols=1,sharex=True)


markers = ["o","^","s"]
## Perform Analysis for different bins                           
for i,n_bins in enumerate([4,8,16]):

    ## Split dimensions into equal-width bins
    bins = {    'S1':list(np.linspace(0,1,n_bins+1)),
                'S1_lag1':list(np.linspace(0,1,n_bins+1)),
                'S2':list(np.linspace(0,1,n_bins+1)),
                'S2_lag1':list(np.linspace(0,1,n_bins+1))
            }

    # Prepare lists for results
    TE_XYs = []
    errors = []
    Z_XYs =  []

    ## Perform multiple observations for each N
    for N in Ns:
        print('Using ' + str(N) + ' samples')
        
        observations = []

        for j in range(N_OBSERVATIONS):
            
            ## Randomise initial conditions
            S1 = np.random.rand()
            S2 = np.random.rand()
            DF = coupled_logistic_map(S1=S1, S2=S2, T=1, N=N, alpha=0.4, epsilon=1, r=4)
            
            ## Initialise TE object
            causality = TransferEntropy(DF = DF,
                                        endog = 'S2',          # Dependent Variable
                                        exog = 'S1',           # Independent Variable
                                        lag = 1
                                        )
            
            ## Calculate NonLinear TE
            (TE_XY, TE_YX) = causality.nonlinear_TE(pdf_estimator = 'histogram',
                                                    bins = bins,
                                                    n_shuffles= N_SHUFFLES)

            observations.append((TE_XY,TE_XY))

        ## Store average over multiple observations
        observations = np.array(observations).reshape(N_OBSERVATIONS,2)
        
        TE_XYs.append(np.average(observations[:,0]))
        errors.append(np.std(observations[:,0]))
        Z_XYs.append(causality.results['z_score_XY'].iloc[0])
        

    TE_axis.errorbar(Ns, TE_XYs, yerr=errors, fmt='-', marker = markers[i], linewidth = 1, capsize=2, elinewidth=0.5, markeredgewidth=0.5)
    Z_axis.semilogy(Ns, Z_XYs, linewidth = 1.5, linestyle = ':')
    Z_axis.set_ylim(ymin=1)


## Format Plots
TE_axis.set_xscale('log', basex=2)
TE_axis.set_ylabel('Transfer Entropy (bits)')
Z_axis.set_xlabel("Sample Size (N)")
Z_axis.set_ylabel('Significance (z-Score)')
TE_axis.legend(['4 bins','8 bins', '16 bins'])
TE_axis.set_ylim(ymin=0,ymax=1.5)
TE_axis.set_title('Replicating Fig. 4 from Boba et al. - TE vs Data Size',fontsize=11)

plt.savefig(os.path.join(os.getcwd(),'PyCausality','Examples','Plots','Coupled_Map.png'))
plt.savefig(os.path.join(os.getcwd(),'PyCausality','Examples','Coupled_Map.pdf'))
plt.show()