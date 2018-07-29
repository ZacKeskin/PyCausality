import pandas as pd
import statsmodels.api as sm
import numpy as np
from copy import deepcopy

from dateutil.relativedelta import relativedelta

import sys
import warnings

from .Utils.Utils import *


class LaggedTimeSeries():
    
    def __init__(self, df, lag=None, max_lag_only=True, window_size = None, window_stride = None):
        """
        Args:
            df          -   Pandas DataFrame object of N columns. Must be indexed as an increasing 
                            time series (i.e. past-to-future), with equal timesteps between each row
            lags        -   The number of steps to be included. Each increase in Lags will result 
                            in N additional columns, where N is the number of columns in the original 
                            dataframe. It will also remove the first N rows.
            max_lag_only -  Defines whether the returned dataframe contains all lagged timeseries up to 
                            and including the defined lag, or only the time series equal to this lag value
            drop_X      -   Defines whether the returned dataframe drops X; if so we have [Y, Y-t, X-t] only

            (TBC Stride - may want e.g. every 7 days etc.)
            (TBC Timesteps - Uses standard Excel parameters for d, w, ww, m, y)
        Returns:    -   n/a
        """        
        self.df = sanitise(df)
        self.axes = list(self.df.columns.values) #Variable names

        self.max_lag_only = max_lag_only
        if lag is not None:
            self.t = lag
            self.df = self.__apply_lags__()

        if window_size is not None and window_stride is not None:
            self.has_windows = True
            self. __apply_windows__(window_size, window_stride)
        else:
            self.has_windows = False

    def __apply_lags__(self):
        """
        Arguments:
            n/a
        Returns:
            new_df.iloc[self.t:]    -   This is a new dataframe containing all lagged 
        """
        # Create a new dataframe to maintain the new data, dropping rows with NaN
        new_df = self.df.copy(deep=True).dropna()

        # Create new column with lagged timeseries for each variable
        col_names = self.df.columns.values.tolist()

        # If the user wants to only consider the time series lagged by the 
        # maximum number specified or by every series up to an including the maximum lag:
        if self.max_lag_only == True:
            for col_name in col_names:
                new_df[col_name + '_lag' + str(self.t)] = self.df[col_name].shift(self.t)

        elif self.max_lag_only == False:
            for col_name in col_names:
                for t in range(1,self.t+1):
                    new_df[col_name + '_lag' + str(t)] = self.df[col_name].shift(t)
        else:
            raise ValueError('Error')

        # Drop the first t rows, which now contain NaN
        return new_df.iloc[self.t:]

    def __apply_windows__(self, window_size, window_stride):
        self.window_size =  {'YS':0,'MS':0,'D':0,'H':0,'min':0,'S':0,'ms':0}
        self.window_stride =  {'YS':0,'MS':0,'D':0,'H':0,'min':0,'S':0,'ms':0}

        self.window_stride.update(window_stride)
        self.window_size.update(window_size)
        freq = ''
        daterangefreq = freq.join([str(v)+str(k) for (k,v) in self.window_stride.items() if v != 0])
        self.daterange = pd.date_range(self.df.index.min(),self.df.index.max() , freq=daterangefreq)

    def date_diff(self,window_size):
        pass

    @property
    def windows(self):

        if self.has_windows == False:
            return self.df
        ## Loop Over TimeSeries Range
        for i,dt in enumerate(self.daterange):
            
            ## Ensure Each Division Contains Required Number of Months
            if dt-relativedelta(years   =  self.window_size['YS'],
                                months  =  self.window_size['MS'],
                                days    =  self.window_size['D'],
                                hours   =  self.window_size['H'],
                                minutes =  self.window_size['min'],
                                seconds =  self.window_size['S'],
                                microseconds = self.window_size['ms']
                                ) >= self.df.index.min():
                
                ## Create Window 
                yield self.df.loc[(dt-relativedelta(years   =  self.window_size['YS'],
                                                    months  =  self.window_size['MS'],
                                                    days    =  self.window_size['D'],
                                                    hours   =  self.window_size['H'],
                                                    minutes =  self.window_size['min'],
                                                    seconds =  self.window_size['S'],
                                                    microseconds = self.window_size['ms']
                                                    )) : dt]


    @property
    def headstart(self):
        windows =   [i for i,dt in enumerate(self.daterange) if dt-relativedelta(years   =  self.window_size['YS'],
                                                                            months  =  self.window_size['MS'],
                                                                            days    =  self.window_size['D'],
                                                                            hours   =  self.window_size['H'],
                                                                            minutes =  self.window_size['min'],
                                                                            seconds =  self.window_size['S'],
                                                                            microseconds = self.window_size['ms']
                                                                            ) < self.df.index.min() ]
        ## i.e. count from the first window which falls entirely after the earliest date
        return len(windows)

class TransferEntropy():

    def __init__(self, DF, endog, exog, lag = None, window_size=None, window_stride=None):
        """
        Arguments:
            DF            -   (DataFrame) Time series data for X and Y (NOT including lagged variables)
            endog         -   (string)    Fieldname for endogenous (dependent) variable Y
            exog          -   (string)    Fieldname for exogenous (independent) variable X
            lag           -   (integer)   Number of periods (rows) by which to lag timeseries data
            window_size   -   (dict)
            window_stride -   (dict)
        """
        self.lts = LaggedTimeSeries(df=sanitise(DF), 
                                    lag=lag, 
                                    window_size=window_size,
                                    window_stride=window_stride)

        if self.lts.has_windows is True:
            self.df = self.lts.windows
            self.date_index = self.lts.daterange[self.lts.headstart:]
            self.results = pd.DataFrame(index=self.date_index)
            self.results.index.name = "windows_ending_on"
        else:
            self.df = [self.lts.df]
            self.results = pd.DataFrame(index=[0])
        self.max_lag_only = True
        self.endog = endog                             # Dependent Variable Y
        self.exog = exog                               # Independent Variable X
        self.lag = lag
        
    def __regression__(self, DF, endog, regressors=None):
        """
        Perform Regression analysis using OLS to calculate variance between Endogenous (Dependent) variable and 
        OLS regression line of best fit.

        Args:
            DF          -  (DataFrame)  N columns. Must be indexed as an increasing time series (i.e. past-to-future),
                                        with equal timesteps between each row
            endog       -  (string)     Name of the dependent variable
            regressors  -  (list)       Optional argument for defining specific independent variables (regressors). 
                                        Otherwise, all other fields in DF are treated as regressor variables

        Returns: (tuple)
            fitted_values   - (list)    The points corresponding to the OLS line of best fit
            residuals       - (list)    The distance (L2 norm) between the fitted line and the endogenous vector
        """

        ## If no custom list of regressors passed, we assume all other columns are required
        if regressors is None:
            regressors = list(DF.drop(endog, axis=1).columns.values)

        model = sm.OLS(DF[endog], DF.loc[:, regressors])
        results = model.fit()
        return (results.fittedvalues, results.resid)

    def linear_TE(self, df):
        """
        Linear Transfer Entropy for directional causal inference

        Defined:            g-causality * 0.5, where g-causality d
        Calculated using:   log(var(e_joint)) - log(var(e_independent)) where e_joint and e_independent
                            represent the residuals from OLS fitting in the joint (X(t),Y(t)) and reduced (Y(t)) cases

        Arguments:
            n/a
        Returns:
            transfer_entropies  -  (list) Directional Linear Transfer Entropies from X(t)->Y(t) and Y(t)->X(t) respectively
        """

         ## Initialise list to return TEs
        transfer_entropies = [0,0]
        
        ## Require us to compare information transfer bidirectionally
        for i,(X,Y) in enumerate({self.exog:self.endog, self.endog:self.exog}.items()):

            ## Note X-t 
            X_lagged = X+'_lag'+str(self.lag)

            ## Calculate Residuals after OLS Fitting, for both Independent and Joint Cases
            joint_df = deepcopy(df).drop(X,axis=1)
            joint_residuals = self.__regression__(DF=joint_df, endog=Y)
     
            independent_df = deepcopy(df).drop([X,X_lagged],axis=1)
            independent_residuals = self.__regression__(DF=independent_df, endog=Y)
            
            ## Calculate Variance of Residuals
            joint_variance = np.var(joint_residuals)
            independent_variance = np.var(independent_residuals)

            ## Calculate G-Causality
            GC = np.log(joint_variance/independent_variance)

            ## Calculate Linear Transfer Entropy
            transfer_entropies[i] = GC/2
        

        return transfer_entropies
   
    def nonlinear_TE(self, df=None, pdf_estimator='histogram', bins=None, bandwidth=None, gridpoints=20, n_shuffles=0):
        """
        NonLinear Transfer Entropy for directional causal inference

        Defined:            TE = TE_XY - TE_YX      where TE_XY = H(Y|Y-t,X-t) - H(Y|Y-t)
        Calculated using:   H(Y|Y-t,X-t) = H(Y,Y-t,X-t) - H(Y,Y-t)  and finding joint entropy through density estimation

        Arguments:
            pdf_estimator   -   (string) "Histogram" only current method available. Used to define which
                                        method is preferred for density estimation of the distribution
            bins            -   (dict of lists) optional parameter to provide hard-coded bin-edges. Dict keys 
                                        must contain names of variables; dict values must be lists
                                        containing bin-edge numerical values. Lagged variables are handled 
                                        automatically so these need not be included.
            n_shuffles      -   (float) Number of times to shuffle the dataframe, destroyig temporality

        Returns:
            transfer_entropies  -  (list) Directional Transfer Entropies from X(t)->Y(t) and Y(t)->X(t) respectively
            (p_value)           -  (set as property self.p_value) p-value with respect to input significance threshold
            (z_score)           -  (set as property self.z_score) z-score with respect to input significance threshold
        """


        ## Retrieve user-defined bins
        self.bins = bins
        if self.bins is None:
            self.bins = {self.endog: None}

        ## Prepare lists for storing results
        TEs = []
        shuffled_TEs = []
        p_values = []
        z_scores = []

        ## Loop over all windows
        for i,df in enumerate(self.df):
            df = deepcopy(df)

            ## Shows user that something is happening
            if self.lts.has_windows is True:
                print("Window ending: ", self.date_index[i])

            ## Initialise list to return TEs
            transfer_entropies = [0,0]

            ## Require us to compare information transfer bidirectionally
            for i,(X,Y) in enumerate({self.exog:self.endog, self.endog:self.exog}.items()):
                
                ### Entropy calculated using Probability Density Estimation:
                    # Following: https://stat.ethz.ch/education/semesters/SS_2006/CompStat/sk-ch2.pdf
                    # Also: https://www.cs.cmu.edu/~aarti/Class/10704_Spring15/lecs/lec5.pdf
                
                ## Note Lagged Terms
                X_lagged = X+'_lag'+str(self.lag)
                Y_lagged = Y+'_lag'+str(self.lag)

                ### Estimate PDF using Gaussian Kernels and use H(x) = p(x) log p(x)

                ## 1. H(Y,Y-t,X-t)  
                H1 = get_entropy(df = df[[Y,Y_lagged,X_lagged]], 
                                gridpoints = gridpoints,
                                bandwidth = bandwidth, 
                                estimator = pdf_estimator,
                                bins = {k:v for (k,v) in self.bins.items()
                                        if k in[Y,Y_lagged,X_lagged]})
                ## 2. H(Y-t,X-t)
                H2 = get_entropy(df = df[[X_lagged,Y_lagged]],
                                gridpoints = gridpoints,
                                bandwidth = bandwidth,
                                estimator = pdf_estimator,
                                bins = {k:v for (k,v) in self.bins.items() 
                                        if k in [X_lagged,Y_lagged]}) 
                ## 3. H(Y,Y-t)  
                H3 = get_entropy(df = df[[Y,Y_lagged]],
                                gridpoints = gridpoints,
                                bandwidth  = bandwidth,
                                estimator = pdf_estimator,
                                bins =  {k:v for (k,v) in self.bins.items() 
                                        if k in [Y,Y_lagged]})
                ## 4. H(Y-t)  
                H4 = get_entropy(df = df[[Y_lagged]],
                                gridpoints = gridpoints,
                                bandwidth  = bandwidth,
                                estimator = pdf_estimator,
                                bins =  {k:v for (k,v) in self.bins.items() 
                                        if k in [Y_lagged]})                


                ### Calculate Conditonal Entropy using: H(Y|X-t,Y-t) = H(Y,X-t,Y-t) - H(X-t,Y-t)
                conditional_entropy_joint =  H1 - H2
            
                ### And Conditional Entropy independent of X(t) H(Y|Y-t) = H(Y,Y-t) - H(Y-t)            
                conditional_entropy_independent = H3 - H4

                ### Directional Transfer Entropy is the difference between the conditional entropies
                transfer_entropies[i] =  conditional_entropy_independent - conditional_entropy_joint
            
            TEs.append(transfer_entropies)

            ## Calculate Significance of TE during this window
            if n_shuffles > 0:
                p, z, TE_mean = significance(    df = df, 
                                        TE = transfer_entropies, 
                                        endog = self.endog, 
                                        exog = self.exog, 
                                        lag = self.lag, 
                                        n_shuffles = n_shuffles, 
                                        pdf_estimator = pdf_estimator, 
                                        bins = bins,
                                        bandwidth = bandwidth)

                shuffled_TEs.append(TE_mean)
                p_values.append(p)
                z_scores.append(z)

        ## Store Transfer Entropy from X(t)->Y(t) and from Y(t)->X(t)
        self.add_results({'TE_XY' : np.array(TEs)[:,0],
                          'TE_YX' : np.array(TEs)[:,1]})
        if n_shuffles > 0:
            ## Store Significance Transfer Entropy from X(t)->Y(t) and from Y(t)->X(t)
            
            self.add_results({'p_value_XY' : np.array(p_values)[:,0],
                              'p_value_YX' : np.array(p_values)[:,1]})
            self.add_results({'z_score_XY' : np.array(z_scores)[:,0],
                              'z_score_YX' : np.array(z_scores)[:,1]})
            self.add_results({'Ave_TE_XY'  : np.array(shuffled_TEs)[:,0],
                              'Ave_TE_YX'  : np.array(shuffled_TEs)[:,1]})

        return transfer_entropies

    def add_results(self,dict):
        for (k,v) in dict.items():
            self.results[str(k)] = v 
         
def significance(df, TE, endog, exog, lag, n_shuffles, pdf_estimator, bins, bandwidth, both=True):
        """
        p-value for significance of hypothesis test, for both X(t)->Y(t) and Y(t)->X(t) directions

        Defined:            
        Calculated using:  Assuming stationarity, we shuffle the time series to provide the null hypothesis. 
                           The proportion of tests where TE > TE_shuffled gives the significance level.

        Arguments:
            TE          -      (list) Contains the transfer entropy in each direction, i.e. [TE_XY, TE_YX]
            n_shuffles  -     (float) Number of times to shuffle the dataframe, destroyig temporality
            both        -      (Bool) Whether to shuffle both endog and exog variables (z-score) or just exog                                  variables (giving z*-score)  
        Returns:
            p_value     -      Probablity of observing the result given the null hypothesis
            z_score     -      Number of Standard Deviations result is from mean (normalised)
        """ 

        ## Prepare array for Transfer Entropy of each Shuffle
        shuffled_TEs = np.zeros(shape = (2,n_shuffles))
        
        ##
        if both is True:
            pass #TBC

        for i in range(n_shuffles):
                ## Perform Shuffle
                df = shuffle_series(df)
                
                ## Calculate New TE
                shuffled_causality = TransferEntropy(   DF = df,
                                                endog = endog,     
                                                exog = exog,          
                                                lag = lag
                                            )           
                TE_shuffled = shuffled_causality.nonlinear_TE(df, pdf_estimator, bins, bandwidth, n_shuffles=0)
                shuffled_TEs[:,i] = TE_shuffled

        
        ## Calculate p-values for each direction
        p_values = (np.count_nonzero(TE[0] < shuffled_TEs[0,:]) /n_shuffles , \
                    np.count_nonzero(TE[1] < shuffled_TEs[1,:]) /n_shuffles)

        ## Calculate z-scores for each direction
        z_scores = ( ( TE[0] - np.mean(shuffled_TEs[0,:]) ) / np.std(shuffled_TEs[0,:]) , \
                     ( TE[1] - np.mean(shuffled_TEs[1,:]) ) / np.std(shuffled_TEs[1,:])  )
        
        TE_mean = ( np.mean(shuffled_TEs[0,:]), \
                     np.mean(shuffled_TEs[1,:]) )
        
        ## Return the self.DF value to the unshuffled case
        return p_values, z_scores, TE_mean
