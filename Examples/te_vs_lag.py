"""
    Ensuring the Effectiveness of Anticipatory/Dependent Time Series

        This example shows the ability of the TransferEntropy class to 
        detect specific time-lags between coupled series. The DATALAG parameter 
        defines the lag from which S2 is dependent on S1; we then loop over LAGs between
        1 and 10 to see where the greatest predictive signal is detected.
        This technique is of course applicable (and useful only) for cases where the data
        is not synthetically generated based on a fixed lag dependency.

"""


import os
import matplotlib.pyplot as plt

from PyCausality.TransferEntropy import *
from PyCausality.Testing.Test_Utils.Time_Series_Generate import *
import matplotlib.pyplot as plt

## Set Parameters
DATALAG = 3                     # The TE code should pick out this lag, which contributes most information
LAGs = [r for r in range(1,10,1)]
SIMILARITY = 0.75               # Choose value between 0 (independent) to 1 (exact lagged value)
AUTOSIMILARITY = 0.3            # Choose value between 0 (independent) to 1 (exact lagged value)
SEED = None                     # Change pseudo-RNG seed; useful for repeat results & comparing bin choices

MAX_BINS = 16
N_SHUFFLES = 50


##------------------------------ Create Highly Coupled Time Seriess ------------------------------##


### Uncomment to compare coupled geometric brownian motions
DATA_POINTS = 300
walks = coupled_random_walks(   S1 = 100, S2 = 90, T = 1, 
                                N = DATA_POINTS, mu1 = 0, mu2 = 0, 
                                sigma1 = 1, sigma2 = 1, 
                                alpha = SIMILARITY, epsilon = AUTOSIMILARITY,
                                lag = DATALAG, seed=SEED)
walks = np.log(walks).diff().iloc[1:]



### Uncomment to compare coupled logistic map
"""
DATA_POINTS = 2**12
walks = coupled_logistic_map(   S1=0.3, S2=0.8, T=1, N=DATA_POINTS, 
                                    alpha=SIMILARITY, epsilon=AUTOSIMILARITY, r=4)

bins = {    'S1':list(np.linspace(0,1,MAX_BINS+1)),
        'S1_lag1':list(np.linspace(0,1,MAX_BINS+1)),
        'S2':list(np.linspace(0,1,MAX_BINS+1)),
        'S2_lag1':list(np.linspace(0,1,MAX_BINS+1))
        }
"""


## Create Lists for Results
TEs_XY = []
TEs_YX = []
z_scoresXY = []
z_scoresYX = []


##------------------------------ Compare TE at Different Lags ------------------------------##

for LAG in LAGs:
    print('LAG', LAG)

    ## Create Lagged Time Series
    DF = LaggedTimeSeries(walks,LAG).df

    ## Initialise TE object
    causality = TransferEntropy(DF = DF,
                                endog = 'S2',          # Dependent Variable
                                exog = 'S1',           # Independent Variable
                                lag = LAG)
                                
    ## Define Bins
    bins = {    'S1':[-5, -2.5, -1, -0.5, 0, 0.5, 1, 2.5, 5],
                'S1_lag'+str(LAG):[-5, -2.5, -1, -0.5, 0, 0.5, 1, 2.5, 5],
                'S2':[-5, -2.5, -1, -0.5, 0, 0.5, 1, 2.5, 5],
                'S2_lag'+str(LAG):[-5, -2.5, -1, -0.5, 0, 0.5, 1, 2.5, 5]
                }                                

    ## Calculate NonLinear TE
    (TE_XY, TE_YX) = causality.nonlinear_TE(pdf_estimator = 'histogram',
                                            bins = bins,
                                            n_shuffles = N_SHUFFLES)

    ## Store Results
    TEs_XY.append(TE_XY)
    TEs_YX.append(TE_YX)
    z_scoresXY.append(causality.results['z_score_XY'].iloc[0])
    z_scoresYX.append(causality.results['z_score_YX'].iloc[0])



##------------------------------ Plot Results------------------------------##

results = pd.DataFrame({'TE_XY':TEs_XY,
                        'TE_YX':TEs_YX,
                        'Z_XY':z_scoresXY,
                        'Z_YX':z_scoresYX} )

## Plot TEs
plt.style.use('seaborn')
fig, (TE_axis, Z_axis) = plt.subplots(nrows=2,ncols=1,sharex=True)

results['lags'] = LAGs
results.set_index(results['lags'],inplace=True)
print(results)
results['TE_XY'].plot(ax=TE_axis,linewidth=1)
results['TE_YX'].plot(ax=TE_axis,linewidth=1)

TE_axis.set_title('TE versus Lag for Coupled Random Walks; Coupling Strength = ' + str(SIMILARITY))

TE_axis.set_ylim(ymin=0, ymax=2.5)
TE_axis.set_ylabel('Transfer Entropy (bits)')
TE_axis.legend(['TE (S1->S2)', 'TE (S2->S1)'])

## Plot Significance
results['Z_XY'].plot(ax=Z_axis,linewidth=0.75, linestyle=':')
results['Z_YX'].plot(ax=Z_axis,linewidth=0.75, linestyle=':')

Z_axis.set_xlabel('LAG')
Z_axis.set_ylim(ymin=-1)
Z_axis.set_ylabel('Significance (z-score)')
Z_axis.legend(['Significance of TE (S1->S2)','Significance of TE (S2->S1)'])

plt.grid(linestyle='dashed')


plt.savefig(os.path.join('Examples','Plots','Detect_Lag.png'))
plt.show()

