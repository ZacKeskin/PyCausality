"""
    Investigation of Optimal Binning for Histogram Density Estimation

        This example demonstrates the effect of different binning techniques on calculating Transfer Entropy using coupled 
        random walks. The transition probablity of the walk follows a normal distribution, and hence we can calculate a theoretical
        transfer entropy using the analytical solution to multivariate entropy. We use a slightly modified equation derived from 
        Elements of Information Theory (Cover & Thomas) pp.230-231, which we use to calculate conditional entropy of the exogenous 
        variable given its own past, conditioned with and without the side information of the endogenous variable.
"""

import matplotlib.pyplot as plt
import os

from PyCausality.Testing.Test_Utils.Time_Series_Generate import *
from PyCausality.TransferEntropy import *

## Set Parameters
LAG = 1
SIMILARITY = 0.9          # Choose value in range [0,1) (where  0 = independent and 1 = exact lagged value)
AUTOSIMILARITY = 0.1      # Choose value in range [0,1) (where  0 = independent and 1 = exact lagged value)
SEED = 10                 # Change pseudo-RNG seed; useful for repeat results & comparing bin choices
DATA_POINTS = 400

N_SHUFFLES = 25
N_OBSERVATIONS = 5
N_BINS = [3,5,7,9,11,13,15]

## Prepare Results:
TE_equal = np.zeros(shape=(N_OBSERVATIONS,len(N_BINS),2))
TE_MIC   = np.zeros(shape=(N_OBSERVATIONS,len(N_BINS),2))
TE_Sigma = np.zeros(shape=(N_OBSERVATIONS,len(N_BINS),2))
TE_theo  = np.zeros(shape=(N_OBSERVATIONS,len(N_BINS),2))
Z_equal  = np.zeros(shape=(N_OBSERVATIONS,len(N_BINS),2))
Z_MIC    = np.zeros(shape=(N_OBSERVATIONS,len(N_BINS),2))
Z_Sigma  = np.zeros(shape=(N_OBSERVATIONS,len(N_BINS),2))

## Plot results
plt.style.use('seaborn')
fig, axes = plt.subplots(nrows=4,ncols=2,sharex=True)

for x,SIMILARITY in enumerate([0, 0.15, 0.5, 0.75]):

    for i in range(N_OBSERVATIONS):
        print('Random Walk #' + str(i))
        ## Generate coupled Gaussian Random Walks
        DF = coupled_random_walks(  S1 = 100, S2 = 100, T = 1, 
                                    N = DATA_POINTS, mu1 = 0, mu2 = 0, 
                                    sigma1 = 0.05, sigma2 = 0.05, 
                                    alpha = SIMILARITY, epsilon = AUTOSIMILARITY,
                                    lag = LAG, seed=SEED)
        ## For Gaussian-Distributed data, a scaled %-change with mu = 0
        DF = DF.pct_change().iloc[1:]



        ## Calculate TE for different bins
        for j,n_bins in enumerate(N_BINS):
            print('bins:', n_bins)

            ## Initialise TE objects
            causality_equal = TransferEntropy(DF = DF, endog = 'S2',exog = 'S1',lag = LAG)
            causality_MIC = TransferEntropy(DF = DF, endog = 'S2',exog = 'S1',lag = LAG)
            causality_Sigma = TransferEntropy(DF = DF, endog = 'S2',exog = 'S1',lag = LAG)

            ## Initialise bins
            equal_bins = {  'S1':list(np.linspace(DF['S1'].min(),DF['S1'].max(),n_bins+1)),
                            'S1_lag'+str(LAG):list(np.linspace(DF['S1'].min(),DF['S1'].max(),n_bins+1)),
                            'S2':list(np.linspace(DF['S2'].min(),DF[['S2']].max(),n_bins+1)),
                            'S2_lag'+str(LAG):list(np.linspace(DF['S1'].min(),DF['S1'].max(),n_bins+1)),
                            }
            AB = AutoBins(df = DF[['S1','S2']], lag = LAG)
            MIC_bins =  AB.MIC_bins(max_bins=n_bins)    # Note MIC_bins can have fewer bins than equal_bins
            Sigma_bins = AB.sigma_bins(max_bins=n_bins)

            ## Calculate TE & Store Significance Values
            
            # MIC Bins
            (TE_MIC[i,j,0], TE_MIC[i,j,1]) = causality_MIC.nonlinear_TE(pdf_estimator='histogram', bins=MIC_bins, n_shuffles=N_SHUFFLES) 
            (Z_MIC[i,j,0], Z_MIC[i,j,1]) = (causality_MIC.results['z_score_XY'].values, causality_MIC.results['z_score_YX'].values)

            # Equal Bins
            (TE_equal[i,j,0], TE_equal[i,j,1]) = causality_equal.nonlinear_TE(pdf_estimator='histogram', bins=equal_bins, n_shuffles=N_SHUFFLES)       
            (Z_equal[i,j,0], Z_equal[i,j,1]) = (causality_equal.results['z_score_XY'].values, causality_equal.results['z_score_YX'].values)
            
            # Sigma Bins
            (TE_Sigma[i,j,0], TE_Sigma[i,j,1]) = causality_Sigma.nonlinear_TE(pdf_estimator='histogram', bins=Sigma_bins, n_shuffles=N_SHUFFLES) 
            (Z_Sigma[i,j,0], Z_Sigma[i,j,1]) = (causality_Sigma.results['z_score_XY'].values, causality_Sigma.results['z_score_YX'].values)


            # Theoretical TE
            H1 = 0.5 * np.log( np.linalg.det(2* np.pi * np.e * np.cov(causality_equal.df[0].drop(columns=['S1']).values.T)) )      
            H2 = 0.5 * np.log( np.linalg.det(2* np.pi * np.e * np.cov(causality_equal.df[0].drop(columns=['S1','S2']).values.T)) )
            H3 = 0.5 * np.log( np.linalg.det(2* np.pi * np.e * np.cov(causality_equal.df[0].drop(columns=['S1','S1_lag'+str(LAG)]).values.T)) )
            H4 = 0.5 * np.log( 2 * np.pi * np.e * np.std(causality_equal.df[0]['S2_lag'+str(LAG)])**2)

            TE_XY_theo = (H3 - H4) - (H1 - H2)
            TE_theo[i,j,0] = 2 ** (np.log(TE_XY_theo) / np.log(2))    # Convert TE from 'Nats' to 'Bits' 



    axes[x][0].xaxis.set_tick_params(labelsize=5)
    axes[x][0].yaxis.set_tick_params(labelsize=5)
    axes[x][1].xaxis.set_tick_params(labelsize=5)
    axes[x][1].yaxis.set_tick_params(labelsize=5)

    axes[x][0].plot(N_BINS,TE_equal.mean(axis=0)[:,0],linewidth=0.8)
    axes[x][0].plot(N_BINS,TE_MIC.mean(axis=0)[:,0],linewidth=0.8)
    axes[x][0].plot(N_BINS,TE_Sigma.mean(axis=0)[:,0],linewidth=0.8)
    axes[x][0].plot(N_BINS,TE_theo.mean(axis=0)[:,0],linewidth=0.8, color='k', linestyle='--')
    axes[x][0].set_ylim(ymin=-0.05,ymax=3)
    axes[x][0].set_ylabel('Transfer Entropy (bits)', fontsize=6)
    axes[x][0].legend(['TE_X->Y with Equal Bins','TE_X->Y with MIC Bins', 
                        'TE_X->Y with Sigma Bins','Theoretical TE X->Y'], fontsize=6)

    axes[x][0].set_title('Coupling Strength: ' + str(SIMILARITY), fontsize=7)
    axes[x][1].plot(N_BINS,Z_equal.mean(axis=0)[:,0],linestyle=':',linewidth=1)
    axes[x][1].plot(N_BINS,Z_MIC.mean(axis=0)[:,0],linestyle=':',linewidth=1)
    axes[x][1].plot(N_BINS,Z_Sigma.mean(axis=0)[:,0],linestyle=':',linewidth=1)
    axes[x][1].set_xlabel('Number of Bins',fontsize=6)
    axes[x][1].set_ylabel('Significance (z-score)',fontsize=6)
    axes[x][1].set_ylim(0,150)
    axes[x][1].legend(['Significance of TE_X->Y with Equal Bins','Significance of TE_X->Y with MIC Bins',
                'Significance of TE_X->Y with Sigma Bins'],fontsize=6)
    fig.suptitle('Comparing TE and Significance for Causally-Coupled Gaussian Time Series using Different Bins.', fontsize=10)
    

plt.savefig(os.path.join(os.getcwd(),'PyCausality', 'Examples','Plots','TE_vs_bins.png'))
plt.show()
