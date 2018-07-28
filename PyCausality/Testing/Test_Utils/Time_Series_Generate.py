"""
    - Useful code to generate time series data for coupling and information transfer analysis
"""
import numpy as np
import pandas as pd
import copy

def random_walk(S0, T, N, mu, sigma, seed=None):
    """
    Generate a random walk time series under Geometric Brownian Motion

    Defined:            dS/S = mu * dt + sigma * dW where dW is a Wiener process

    Arguments:
        S0              -   (float)     Inital value of series S
        T               -   (float)     Total time
        N               -   (float)     Resolution (number of steps to divide T into)
        mu              -   (float)     Estimated mean change each timestep
        sigma           -   (float)     Estimated SD of the series
        seed            -   (float)     Define np.random.seed to compare results
    Returns:
        S, time         -   (np arrays) Timesteps and the value of S at each step.
    """

    if not seed is None:
        np.random.seed(seed)

    S = [0 for n in range(N)]
    timesteps = copy.deepcopy(S)
    dt = T/N
    S[0] = S0

    for m in range(1,N):        
        S[m] =  S[m-1] * ( 1 + 
                    (mu * dt) + 
                    (np.random.normal(0,sigma) * np.sqrt(dt))
                )
        timesteps[m] = dt*m 
    return np.array(S), np.array(timesteps)

def _logistic_map_(p, r):
    return r * p * (1-p)

def _g_simple_(p,epsilon,r):
    return epsilon * _logistic_map_(p,r)

def _g_(p,epsilon,r):
    return (1-epsilon) * _logistic_map_(p,r) + epsilon * _logistic_map_(_logistic_map_(p,r),r)

def coupled_logistic_map(S1, S2, T, N, alpha, epsilon, r=4, g=None):
    """
    Generate two time series under a logistic mapping function. (See Boba et al (2015), following
    Hahs & Pethel (2011)). Note that we swap alpha and epsilon for consistency with coupled_random_walks()

    Arguments:
        S1              -   (float)     Inital value of series S1
        S2              -   (float)     Inital value of series S2
        T               -   (float)     Total time
        N               -   (float)     Resolution (number of steps to divide T into)
        alpha           -   (float)     'Dependency' quotient, between 0 and 1, of S2 on S1-lagged
        epsilon         -   (float)     'Self-Dependency' quotient, between 0 and 1, anticipates
        r               -   (float)     logistic map parameter. r>3.57 generally presents chaotic behaviour
        g               -   (string)    anticipatory function g(x). Default is as in Boba et al.
    """
    ## Generate index and initialise S1, S2 lists
    timesteps = list(np.linspace(0,T,N))
    (S1,S2) = ([S1],[S2])

    ## Select anticipatory function g(x):
    if g == 'simple':
        g = _g_simple_
    else:
        g = _g_

    ## Populate time series
    [S1.append(_logistic_map_(S1[n],r)) for n in range(N-1)]

    [S2.append((1-alpha) * _logistic_map_(S2[n],r) + 
                  alpha  * g(S1[n],epsilon,r)) for n in range(N-1)]

    ## Return DataFrame
    walk = pd.DataFrame({'S2':S2, 'S1':S1, 'time':timesteps})
    walk.set_index('time', inplace=True)
    return walk
    

def coupled_random_walks(S1, S2, T, N, mu1, mu2, sigma1, sigma2, alpha, epsilon, lag, seed=None):
    """
    Generate two time series under Geometric Brownian Motion with S2 dependent in part on S1-lagged

    Arguments:
        S1              -   (float)     Inital value of series S1
        S2              -   (float)     Inital value of series S2
        T               -   (float)     Total time
        N               -   (float)     Resolution (number of steps to divide T into)
        mu1             -   (float)     Estimated mean change each timestep for S1
        mu2             -   (float)     Estimated mean change each timestep for S2    
        sigma1          -   (float)     Estimated SD of the series S1
        sigma2          -   (float)     Estimated SD of the series S2
        alpha           -   (float)     'Dependency' quotient, between 0 and 1, of S2 on S1-lagged
        epsilon         -   (float)     'Self-Dependency' quotient, between 0 and 1, anticipates
        lag             -   (int)       Number of timesteps lag in S1 on which S2 is dependent
        seed            -   (float)     Define np.random.seed to compare results
    Returns:
        S, time         -   (np arrays) Timesteps and the value of S at each step.
    """
    if not seed is None:
        seed2 = seed+1
    else:
        seed2 = None

    S1, time = random_walk(S0=S1, T=T, N=N, mu=mu1, sigma=sigma1, seed=seed)
    S2, time = random_walk(S0=S2, T=T, N=N, mu=mu2, sigma=sigma2, seed=seed2)

    ## Apply a dependence on the lagged series S1 and S2
    S2[lag:] = (1-alpha)    * ( epsilon*S2[:-lag] + (1-epsilon) * S2[lag:] ) + \
                (alpha)     *   S1[:-lag]
    
    walk = pd.DataFrame({'S2':S2, 'S1':S1, 'time':time})
    walk.set_index('time', inplace=True)

    # Drop the first 'lagged' rows, which can't have dependence
    return walk.iloc[lag:]  
