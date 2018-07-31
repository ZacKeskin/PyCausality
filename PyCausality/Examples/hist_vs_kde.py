"""
    Comparison of Density Estimation Techniques

        This example shows the effect of selecting histogram or kernel density estimation when
        approximating the probability distribution of the provided data. We use a coupled random
        walk, with transition probablity which follows the normal distribution; hence we can 
        calculate the theoretical transfer entropy using the analytical solution to multivariate 
        entropy. 

"""

import os
import matplotlib.pyplot as plt

from PyCausality.TransferEntropy import *
from PyCausality.Testing.Test_Utils.Time_Series_Generate import *

## Set Parameters
LAG = 1

AUTOSIMILARITY = 0         # Choose value in range [0,1) (where  0 =independent and 1=exact lagged value)
SEED = None                # Change pseudo-RNG seed; useful for repeat results & comparing bin choices
DATA_POINTS = 250
BANDWIDTH = 0.8            # Choose bandwidth for KDE
SIMILARITY = np.linspace(0,0.9,9)
N_SHUFFLES = 50

## Prepare Results
TE_XY_Hists = []
TE_YX_Hists = []
TE_XY_KDEs = []
TE_YX_KDEs = []
TE_XY_theos = []

z_scores_XY_Hists = []
z_scores_YX_Hists = []
z_scores_XY_KDEs = []
z_scores_YX_KDEs = []

for a in SIMILARITY:
    print("\n\n Coupling Strength:",a)

    ## Create Coupled Time Series
    DF = coupled_random_walks(  S1 = 200, S2 = 200, T = 0.5, 
                                N = DATA_POINTS, mu1 = 0, mu2 = 0, 
                                sigma1 = 0.5, sigma2 = 0.25, 
                                alpha = a, epsilon = AUTOSIMILARITY,
                                lag = LAG, seed=SEED)
   

    ## To get data which is Gaussian distributed, we use a scaled %-change with mu = 0
    DF = DF.pct_change().iloc[1:]

    #plot_pdf(DF)
    #plt.show()

    ## Initialise TE object
    causality = TransferEntropy(DF = DF,
                                endog = 'S2',          # Dependent Variable
                                exog = 'S1',           # Independent Variable
                                lag = LAG
                                )

    ## Uncomment to Choose Bins
    #edges = None # Uses sigma_binning by default
    auto = AutoBins(DF[['S1','S2']])
    edges = auto.knuth_bins(max_bins=20)

    ## Calculate TE using Histogram
    #(TE_XY, TE_YX) = causality.nonlinear_TE(pdf_estimator = 'kernel', n_shuffles=N_SHUFFLES)
    (TE_XY, TE_YX) = causality.nonlinear_TE(pdf_estimator = 'histogram', bins= edges, n_shuffles=N_SHUFFLES)
    TE_XY_Hists.append(TE_XY)
    TE_YX_Hists.append(TE_YX)
    z_scores_XY_Hists.append(causality.results['z_score_XY'].iloc[0])
    z_scores_YX_Hists.append(causality.results['z_score_YX'].iloc[0])
    print("Histogram TE: \t\t",TE_XY,'\t', TE_YX)
    print("Hist Z-scores: \t\t", causality.results['z_score_XY'].iloc[0], '\t', causality.results['z_score_YX'].iloc[0])

    ## Calculate TE using KDE
    (TE_XY, TE_YX) = causality.nonlinear_TE(pdf_estimator = 'kernel', bandwidth=BANDWIDTH, gridpoints=20, n_shuffles=N_SHUFFLES)
    TE_XY_KDEs.append(TE_XY)
    TE_YX_KDEs.append(TE_YX)
    z_scores_XY_KDEs.append(causality.results['z_score_XY'].iloc[0])
    z_scores_YX_KDEs.append(causality.results['z_score_YX'].iloc[0])
    print("Kernel TE: \t\t", TE_XY,'\t', TE_YX)
    print("Kernel Z-scores: \t\t", causality.results['z_score_XY'].iloc[0], '\t', causality.results['z_score_YX'].iloc[0])
   
    ## Calculate Theoretical Entropy
    H1 = 0.5 * np.log( np.linalg.det(2* np.pi * np.e * np.cov(causality.df[0].drop(columns=['S1']).values.T)) )      
    H2 = 0.5 * np.log( np.linalg.det(2* np.pi * np.e * np.cov(causality.df[0].drop(columns=['S1','S2']).values.T)) )
    H3 = 0.5 * np.log( np.linalg.det(2* np.pi * np.e * np.cov(causality.df[0].drop(columns=['S1','S1_lag'+str(LAG)]).values.T)) )
    H4 = 0.5 * np.log( 2 * np.pi * np.e * np.std(causality.df[0]['S2_lag'+str(LAG)])**2)
    TE_XY_theo = (H3 - H4) - (H1 - H2)
    TE_XY_theos.append(2 ** (np.log(TE_XY_theo) / np.log(2)))    # Convert TE to 'Bits'



### -------------------------------------- Store & Plot Results ---------------------------------------- ###


results = pd.DataFrame({'Coupling':SIMILARITY,
                        'TE_XY':TE_XY_Hists,
                        'TE_YX':TE_YX_Hists,
                        'TE_XY(kde)':TE_XY_KDEs,
                        'TE_YX(kde)':TE_YX_KDEs,
                        'TE_XY(Theoretical)':TE_XY_theos,
                        'z_scores_XY':z_scores_XY_Hists,
                        'z_scores_YX':z_scores_YX_Hists,
                        'z_scores_XY(kde)':z_scores_XY_KDEs,
                        'z_scores_YX(kde)':z_scores_YX_KDEs
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

results['z_scores_XY'].plot(linewidth=0.9, linestyle=':',color='b',ax=Z_axis)
results['z_scores_YX'].plot(linewidth=0.9, linestyle=':',ax=Z_axis)
results['z_scores_XY(kde)'].plot(linewidth=0.9, linestyle=':',ax=Z_axis)
results['z_scores_YX(kde)'].plot(linewidth=0.9, linestyle=':',ax=Z_axis)

## Format Plots
TE_axis.set_title('Calculation of Transfer Entropy for Synthetic Coupled Geometric Random Walks', weight='bold')
TE_axis.set_ylabel('Transfer Entropy (bits)')
TE_axis.legend(['TE from S1->S2 (Histogram Estimation)', 'TE from S1->S2 (KDE)',
                'TE from S2->S1 (Histogram Estimation)', 'TE from S2->S1 (KDE)',
                'Theoretical TE of Normally Distributed Stochastic Process'])
Z_axis.set_ylabel('Significance (z-score)')
Z_axis.legend(['Significance of TE from S1->S2 (Histogram)','Significance of TE from S1->S2 (KDE)',
                'Significance of TE from S2->S1 (Histogram)','Significance of TE from S2->S1 (KDE)'])

plt.grid(linestyle='dashed')
plt.savefig(os.path.join(os.getcwd(),'PyCausality','Examples','Plots','Histogram_vs_KDE.png'))
plt.show()