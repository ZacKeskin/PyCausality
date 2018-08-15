"""
    Comparison of Density Estimation Techniques

        This example shows the effect of selecting histogram or kernel density estimation when
        approximating the probability distribution of the provided data. We use a coupled random
        walk, with a Gaussian transition probablity; hence we can compare the accuracy of the estimate to 
        the theoretical transfer entropy calculated using the analytical solution to differential entropy of 
        the multivariate Gaussian distribution.

        The analysis is performed over multiple realisations, defined by the N_OBSERVATIONS parameter, and the 
        N_SHUFFLES drives a significance test, returning Z-score values for each method. 

"""

import os
import matplotlib.pyplot as plt

from PyCausality.TransferEntropy import *
from PyCausality.Testing.Test_Utils.Time_Series_Generate import *

## Set Parameters
LAG = 1

AUTOSIMILARITY = 0         # Choose value in range [0,1) (where  0 =independent and 1=exact lagged value)
DATA_POINTS = 1000
BANDWIDTH = 0.8            # Choose bandwidth for KDE
SIMILARITY = np.linspace(0,0.9,10)
N_SHUFFLES = 50
N_OBSERVATIONS = 3

## Prepare containers to store results
TE_Hists = np.zeros(shape=(N_OBSERVATIONS,len(SIMILARITY),2))
TE_KDEs  = np.zeros(shape=(N_OBSERVATIONS,len(SIMILARITY),2))
Z_Hists  = np.zeros(shape=(N_OBSERVATIONS,len(SIMILARITY),2))
Z_KDEs   = np.zeros(shape=(N_OBSERVATIONS,len(SIMILARITY),2))
TE_XY_Theos   = np.zeros(shape=(N_OBSERVATIONS,len(SIMILARITY)))



## Calculate TE for increasing coupling strength 
for i,alpha in enumerate(SIMILARITY):
    print("\n\n Coupling Strength:", alpha)

    ## Loop over multiple realisations at each alpha, for improved reliability
    for observation in range(N_OBSERVATIONS):  
    
        ## Create Coupled Time Series
        DF = coupled_random_walks(  S1 = 200, S2 = 200, T = 0.5, 
                                    N = DATA_POINTS, mu1 = 0, mu2 = 0, 
                                    sigma1 = 0.5, sigma2 = 0.25, 
                                    alpha = alpha, epsilon = AUTOSIMILARITY,
                                    lag = LAG, seed=None)

        ## To get data which is Gaussian distributed, we use a scaled %-change with mu = 0
        DF = DF.pct_change().iloc[1:]

        #plot_pdf(DF,show=True)


        ## Initialise TE object
        causality = TransferEntropy(DF = DF,
                                    endog = 'S2',          # Dependent Variable
                                    exog = 'S1',           # Independent Variable
                                    lag = LAG
                                    )

        ## Generate histogram bins
        auto = AutoBins(DF[['S1','S2']], lag=LAG)
        edges = auto.sigma_bins(max_bins=5)

        ## Calculate TE using Histogram
        TE_Hists[observation,i,:] = causality.nonlinear_TE(pdf_estimator = 'histogram', bins= edges, n_shuffles=N_SHUFFLES)
        Z_Hists[observation,i,:] = (causality.results['z_score_XY'].iloc[0], causality.results['z_score_YX'].iloc[0])
        
        ## Calculate TE using KDE
        TE_KDEs[observation,i,:] = causality.nonlinear_TE(pdf_estimator = 'kernel', gridpoints=25,bandwidth=BANDWIDTH, n_shuffles=N_SHUFFLES)
        Z_KDEs[observation,i,:] = (causality.results['z_score_XY'].iloc[0], causality.results['z_score_YX'].iloc[0])

        ## Calculate Theoretical Entropy
        H1 = 0.5 * np.log( np.linalg.det(2* np.pi * np.e * np.cov(causality.df[0].drop(columns=['S1']).values.T)) )      
        H2 = 0.5 * np.log( np.linalg.det(2* np.pi * np.e * np.cov(causality.df[0].drop(columns=['S1','S2']).values.T)) )
        H3 = 0.5 * np.log( np.linalg.det(2* np.pi * np.e * np.cov(causality.df[0].drop(columns=['S1','S1_lag'+str(LAG)]).values.T)) )
        H4 = 0.5 * np.log( 2 * np.pi * np.e * np.std(causality.df[0]['S2_lag'+str(LAG)])**2)
        TE_XY_Theos[observation,i] = (H3 - H4) - (H1 - H2)


    # Print results averaged over N_OBSERVATIONs
    print("Histogram TE: \t\t",TE_Hists[:,i,:].mean(axis=0))
    print("Histogram Z-scores:", Z_Hists[:,i,:].mean(axis=0))
    print("Kernel TE: \t\t",TE_KDEs[:,i,:].mean(axis=0))
    print("Kernel Z-scores:", Z_KDEs[:,i,:].mean(axis=0))
    

### -------------------------------------- Store & Plot Results ---------------------------------------- ###


results = pd.DataFrame({'Coupling':SIMILARITY,
                        'TE_XY':TE_Hists.mean(axis=0)[:,0],
                        'TE_YX':TE_Hists.mean(axis=0)[:,1],
                        'TE_XY(kde)':TE_KDEs.mean(axis=0)[:,0],
                        'TE_YX(kde)':TE_KDEs.mean(axis=0)[:,1],
                        'TE_XY(Theoretical)':TE_XY_Theos.mean(axis=0),
                        'z_scores_XY':Z_Hists.mean(axis=0)[:,0],
                        'z_scores_YX':Z_Hists.mean(axis=0)[:,1],
                        'z_scores_XY(kde)':Z_KDEs.mean(axis=0)[:,0],
                        'z_scores_YX(kde)':Z_KDEs.mean(axis=0)[:,1]
                        })
results.set_index('Coupling',inplace=True)


## Prepare Plots (2 y-axis with shared x-axis)
import matplotlib.pyplot as plt
plt.style.use('seaborn')
fig, (TE_axis, Z_axis) = plt.subplots(nrows=2,ncols=1,sharex=True)

## Plot Results
results['TE_XY'].plot(linewidth=1,ax=TE_axis)
results['TE_XY(kde)'].plot(linewidth=1,ax=TE_axis)
results['TE_YX'].plot(linewidth=1,ax=TE_axis)
results['TE_YX(kde)'].plot(linewidth=1,ax=TE_axis)
results['TE_XY(Theoretical)'].plot(linewidth=1.25,ax=TE_axis, color='k', linestyle='--')

results['z_scores_XY'].plot(linewidth=0.9, linestyle=':', ax=Z_axis)
results['z_scores_XY(kde)'].plot(linewidth=0.9, linestyle=':', ax=Z_axis)
results['z_scores_YX'].plot(linewidth=0.9, linestyle=':', ax=Z_axis)
results['z_scores_YX(kde)'].plot(linewidth=0.9, linestyle=':', ax=Z_axis)

## Format Plots
TE_axis.set_title('Calculation of Transfer Entropy for Synthetic Coupled Geometric Random Walks', fontsize=11)
TE_axis.set_ylabel('Transfer Entropy (bits)',fontsize=10)
TE_axis.legend(['TE from S1->S2 (Histogram Estimation)', 'TE from S1->S2 (KDE)',
                'TE from S2->S1 (Histogram Estimation)', 'TE from S2->S1 (KDE)',
                'Theoretical TE of Normally Distributed Stochastic Process'], fontsize=8)
Z_axis.set_yscale('log')
Z_axis.set_ylim(ymin=2)
Z_axis.set_ylabel('Significance (z-score)',fontsize=10)
Z_axis.legend(['Significance of TE from S1->S2 (Histogram)','Significance of TE from S1->S2 (KDE)',
                'Significance of TE from S2->S1 (Histogram)','Significance of TE from S2->S1 (KDE)'], fontsize=8)

plt.grid(linestyle='dashed')
plt.savefig(os.path.join(os.getcwd(),'PyCausality','Examples','Plots','Histogram_vs_KDE (sigma bins).png'))
plt.show()