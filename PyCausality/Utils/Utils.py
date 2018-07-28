import pandas as pd
import numpy as np
from numpy import ma, diff
from scipy import stats, optimize
from scipy.special import gammaln
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import warnings

##############################################################################################################
###   U T I L I T Y    C L A S S E S
############################################################################################################## 


class NDHistogram():
    """
        Custom histogram class wrapping the default numpy implementations (np.histogram, np.histogramdd). 
        This allows for dimension-agnostic histogram calculations, custom auto-binning and 
        associated data and methods to be stored for each object (e.g. Probability Density etc.)
    """
    def __init__(self, df, bins=None, max_bins = 5):
        """
        Arguments:
            df          -   DataFrame passed through from the TransferEntropy class
            bins        -   Bin edges passed through from the TransferEntropy class
            max_bins   -   Number of bins per each dimension passed through from the TransferEntropy class
        Returns:
            self.pdf    -   This is an N-dimensional Probability Density Function, stored as a
                            Numpy histogram, representing the proportion of samples in each bin.
        """
        self.df = sanitise(df).reindex(columns= sorted(df.columns))   # Sort axes by name
        self.max_bins = max_bins
        self.axes = list(self.df.columns.values)
        self.bins = bins
        self.n_dims = len(self.axes)
        
        ## Bins must match number and order of dimensions
        if self.bins is None or set(self.bins.keys()) != set(self.axes):
            warnings.warn('Incorrect or no bins provided - defaulting to sigma bins')
            AB = AutoBins(self.df)
            self.bins = AB.sigma_bins(max_bins=max_bins)
            
        ordered_bins = [self.bins[key] for key in sorted(self.bins.keys())]

        ## Create ND histogram (np.histogramdd doesn't scale down to 1D)
        if self.n_dims == 1:
            self.Hist, self.Dedges = np.histogram(self.df.values,bins=ordered_bins[0], normed=False)
        elif self.n_dims > 1:
            self.Hist, self.Dedges = np.histogramdd(self.df.values,bins=ordered_bins, normed=False)
        

        ## Empirical Probability Density Function
        if self.Hist.sum() == 0:   
            print(self.Hist.shape)
            
            with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
                print(self.df)

            sys.exit("Histogram has zero value. Reduce number of bins or increase data points")
        else:
            self.pdf = self.Hist/self.Hist.sum()
            self._set_entropy_(self.pdf)
  
    def _set_entropy_(self,pdf):
        """
        Arguments:
            pdf   -   Probabiiity Density Function; this is calculated using the N-dimensional histogram above.
        Returns:
            n/a   
        """
        ## Prepare empty dict for marginal entropies along each dimension
        self.H = {}

        if self.n_dims >1:
            
            ## Joint entropy H(X,Y) = -sum(pdf(x,y) * log(pdf(x,y)))     
            self.H_joint =  -np.sum(pdf * ma.log2(pdf).filled(0)) # Use masking to replace log(0) with 0

            ## Single entropy for each dimension H(X) = -sum(pdf(x) * log(pdf(x)))
            for a, axis_name in enumerate(self.axes):
                self.H[axis_name] =  -np.sum(pdf.sum(axis=a) * ma.log2(pdf.sum(axis=a)).filled(0)) # Use masking to replace log(0) with 0
        else:
            ## Joint entropy and single entropy are the same
            self.H_joint = -np.sum(pdf * ma.log2(pdf).filled(0)) 
            self.H[self.df.columns[0]] = self.H_joint

class AutoBins():

    def __init__(self, df, lag=None):
        """
        """
        ## Ensure data is in DataFrame form
        self.df = sanitise(df)
        self.axes = self.df.columns.values
        self.ndims = len(self.axes)
        self.N = len(self.df)
        self.lag = lag

    def __knuth_function__(self, M):
        """
            Argmax function for multidimensional Knuth's Rule
        """
        bins =  [list(np.linspace(  self.df[dim].min(), 
                                    self.df[dim].max(), 
                                    int(M[i])+1))
                                for i,dim in enumerate(self.df.columns.values)]

        nk, bins = np.histogramdd(self.df.values,bins)
        
        ## Note:  M is number of bins in total Mx * My * Mz ... etc.

        M = np.prod(M)

        return -( self.N * np.log(M)
                    + gammaln(0.5 * M)
                    - M * gammaln(0.5)
                    - gammaln(self.N + 0.5 * M)
                    + np.sum(gammaln(nk.ravel() + 0.5))) 

    def __extend_bins__(self, bins):
        
        self.max_lag_only = True # still temporary until we kill this

        ## Handle lagging for bins, and calculate default bins where edges are not provided
        if self.max_lag_only == True:
            bins.update({   fieldname + '_lag' + str(self.lag): edges   
                            for (fieldname,edges) in bins.items()})  
        else:
            bins.update({   fieldname + '_lag' + str(t): edges          
                            for (fieldname,edges) in bins.items() for t in range(self.lag)})
        
        return bins

    def MIC_bins(self, max_bins = 15):
        """
        Method to find optimal bin widths in each dimension, using a naive search to 
        maximise the mutual information divided by number of bins. 
        We increase the n_bins parameter in each dimension, and take the bins which
        result in the greatest Maximum Information Coefficient (MIC)
        
        (Note that this is restricted to equal-width bins only.)

        Defined:            MIC = I(X,Y)/ max(n_bins)
                            edges = {Y:[a,b,c,d], Y-t:[a,b,c,d], X-t:[e,f,g]}, 
                            n_bins = [bx,by]
        Calculated using:   argmax { I(X,Y)/ max(n_bins) }

        Arguments:
            df              -   (DataFrame) Time series data to calculate optimal bins
            max_bins        -   (int)       The maximum allowed bins in each dimension
        Returns:
            opt_edges       -   (dict)      The optimal bin-edges for pdf estimation
                                            using the histogram method, keyed by df column names
        """
        if len(self.df.columns.values) > 2:
            raise ValueError('Too many columns provided in DataFrame. MIC_bins only accepts 2 columns (no lagged columns)')

            
        min_bins = 3

        ## Initialise array to store MIC values
        MICs = np.zeros(shape=[1+max_bins-min_bins,1+max_bins-min_bins])
        
        ## Loop over each dimension 
        for b_x in range(min_bins, max_bins+1):

            for b_y in range(min_bins, max_bins+1):
                
                ## Update parameters
                n_bins = [b_x,b_y]

                ## Update dict of bin edges
                edges = {dim :  list(np.linspace(   self.df[dim].min(), 
                                                    self.df[dim].max(), 
                                                    n_bins[i]+1))
                                for i,dim in enumerate(self.df.columns.values)}

                ## Calculate Maximum Information Coefficient
                HDE = NDHistogram(self.df, edges)
                
                I_xy = sum(HDE.H.values()) - HDE.H_joint

                MIC = I_xy / np.log2(np.min(n_bins))
                
                MICs[b_x-min_bins][b_y-min_bins] = MIC 
                

        ## Get Optimal b_x, b_y values
        n_bins[0] = np.where(MICs == np.max(MICs))[0] + min_bins
        n_bins[1] = np.where(MICs == np.max(MICs))[1] + min_bins
        
        bins = {dim :  list(np.linspace(self.df[dim].min(), 
                            self.df[dim].max(), 
                            n_bins[i]+1))
                        for i,dim in enumerate(self.df.columns.values)}
        
        if self.lag is not None:
            bins = self.__extend_bins__(bins)
        ## Return the optimal bin-edges
        return bins

    def Z_bins(df, endog, exog, lag, max_bins = 10, n_shuffles = 20, MA = 2):
        """
        Method to find optimal bin widths in each dimension, using Cross Validation,
        where optimal is defined by maximising z-scored significance divided by number of bins. 
        We increase the n_bins parameter in each dimension until z_score/n_bins decreases.
        Note that this is restricted to equal-width bins only.

        Defined:            edges = {Y:[a,b,c,d], Y-t:[a,b,c,d], X-t:[e,f,g]}
        Calculated using:   

        Arguments:
            df              -   (DataFrame) Time series data to calculate optimal bins
            endog           -   (string)    Fieldname for endogenous (dependent) variable (Y)
            exog            -   (string)    Fieldname for exogenous (independent) variable (X)
            lag             -   (int)       The lag used to generate the lagged time series data
            max_bins        -   (int)       The maximum allowed bins in each dimension
            significance    -   (float)     From 0 to 1. Significance level for z-score
            MA              -   (int)       Number of examples in Moving Average calculation, which determines optimum bin number
        Returns:
            opt_edges       -   (dict)      The optimal bin-edges for pdf estimation
                                            using the histogram method
        """

        ## Split dataset into test and train
        mask = np.random.rand(len(df)) < 0.8
        train = df[mask]
        test = df[~mask]
        


        ## Initialise parameter in each dimension
        n_bins = [3 for dim in list(train.columns.values)]

        ## Initialise Object to Perform TE Analysis
        causality = TransferEntropy(    endog = endog,      # Dependent Variable
                                        exog = exog,           # Independent Variable
                                        lag = lag
                                    )

        ## Loop over each dimension 
        for i, dim in enumerate(train.columns.values):

            ## Initialise parameter for z-score/n_bins
            params_old = [0] 

            while n_bins[i] < max_bins:

                ## Update dict of bin edges
                opt_edges = {dim :  list(np.linspace(train[dim].min(), 
                                                        train[dim].max(), 
                                                        n_bins[i]))
                                for i,dim in enumerate(train.columns.values)}
                
                
                ## Calculate TE using these bin widths
                TE = causality.nonlinear_TE(df = DF, pdf_estimator='histogram', bins=opt_edges, n_shuffles=n_shuffles)
                
                ## Consider the bins resulting in most significant TE from X->Y
                z_score = causality.z_score[0]

                ## Update historical parameters, used for moving average calculation
                params_old.append(z_score)   
                if len(params_old) > MA:
                    params_old.pop(0)

                ## Continue while |z-score/b| is increasing wrt. moving average
                if abs(z_score) >= abs(np.mean(params_old)):                    
                    n_bins[i] += 1
                else:
                    n_bins[i] -= 1
                    break

        ## Return the optimal bin-edges
        print('Using bin-edges: ', opt_edges)
        return opt_edges

    def knuth_bins(self,max_bins=15):
        
        if len(self.df.columns.values) > 2:
            raise ValueError('Too many columns provided in DataFrame. knuth_bins only accepts 2 columns (no lagged columns)')

        
        min_bins = 3

        ## Initialise array to store MIC values
        log_probabilities = np.zeros(shape=[1+max_bins-min_bins,1+max_bins-min_bins])
        
        ## Loop over each dimension 
        for b_x in range(min_bins, max_bins+1):

            for b_y in range(min_bins, max_bins+1):
                
                ## Update parameters
                Ms = [b_x,b_y]
                
                ## Update dict of bin edges
                bins = {dim :  list(np.linspace(    self.df[dim].min(), 
                                                    self.df[dim].max(), 
                                                    Ms[i]+1))
                                for i,dim in enumerate(self.df.columns.values)}

                ## Calculate Maximum log Posterior
                
                # Create N-d histogram to count number per bin
                HDE = NDHistogram(self.df, bins)
                nk = HDE.Hist

                # M = number of bins in total =  Mx * My * Mz ... etc.
                M = np.prod(Ms)

                log_prob = ( self.N * np.log(M)
                            + gammaln(0.5 * M)
                            - M * gammaln(0.5)
                            - gammaln(self.N + 0.5 * M)
                            + np.sum(gammaln(nk.ravel() + 0.5)))

                log_probabilities[b_x-min_bins][b_y-min_bins] = log_prob 
        

        ## Get Optimal b_x, b_y values
        Ms[0] = np.where(log_probabilities == np.max(log_probabilities))[0] + min_bins
        Ms[1] = np.where(log_probabilities == np.max(log_probabilities))[1] + min_bins
        
        bins = {dim :  list(   np.linspace(self.df[dim].min(), 
                                                self.df[dim].max(), 
                                                Ms[i]+1))
                            for i,dim in enumerate(self.df.columns.values)}
        
        if self.lag is not None:
            bins = self.__extend_bins__(bins)
        ## Return the optimal bin-edges
        return bins

    def sigma_bins(self,max_bins=8):
        """ 
            Finds bins for N-dimensional data, using standard deviation binning.
        """
        bins = {k:[np.mean(v)-int(max_bins/2)*np.std(v) + i * np.std(v) for i in range(max_bins+1)] 
                for (k,v) in self.df.iteritems()}   # Note: same as:  self.df.to_dict('list').items()}
        
        if self.lag is not None:
            bins = self.__extend_bins__(bins)
        return bins


##############################################################################################################
###   U T I L I T Y    F U N C T I O N S 
##############################################################################################################    
    

def get_pdf(df, gridpoints=None, bandwidth=None, estimator=None, bins=None):

    if estimator == 'histogram':
        pdf = pdf_histogram(df, bins)
    else:
        pdf = pdf_kde(df, gridpoints, bandwidth)
    return pdf

def pdf_kde(df, gridpoints = None, bandwidth = 1):
    
    ## Create Meshgrid to capture data
    if gridpoints is None:
        gridpoints = 20
    
    N = complex(gridpoints)
    
    slices = [slice(dim.min(),dim.max(),N) for dimname, dim in df.iteritems()]
    grids = np.mgrid[slices]

    ## Pass Meshgrid to Scipy Gaussian KDE to Estimate PDF
    positions = np.vstack([X.ravel() for X in grids])
    values = df.values.T
    kernel = stats.gaussian_kde(values, bw_method= bandwidth)
    Z = np.reshape(kernel(positions).T, grids[0].shape) 

    ## Normalise 
    return Z/Z.sum()

def pdf_histogram(df,bins):
    histogram = NDHistogram(df=df, bins=bins)        
    return histogram.pdf

def get_entropy(df, gridpoints=20, bandwidth=None, estimator='kernel', bins=None):
    pdf = get_pdf(df, gridpoints, bandwidth, estimator, bins)
    ## log base 2 returns H(X) in bits
    return -np.sum( pdf * ma.log2(pdf).filled(0)) 

def shuffle_series(DF, only=None):
    """
    Function to return time series shuffled rowwise along each desired column. 
    Each column is shuffled independently, removing the temporal relationship.

    This is to calculate Z-score and Z*-score. See P. Boba et al (2015)

    Calculated using:       df.apply(np.random.permutation)

    Arguments:
        df              -   (DataFrame) Time series data 
        only            -   (list)      Fieldnames to shuffle. If none, all columns shuffled 
    Returns:
        df_shuffled     -   (DataFrame) Time series shuffled along desired columns    
    """
    if not only == None:
        shuffled_DF = DF.copy()
        for col in only:
            series = DF.loc[:, col].to_frame()
            shuffled_DF[col] = series.apply(np.random.permutation)
    else:
        shuffled_DF = DF.apply(np.random.permutation)
    
    return shuffled_DF

def plot_pdf(df,gridpoints=None,bandwidth=None):
    
    ## Estimate the PDF from the data
    if gridpoints is None:
        gridpoints = 20

    pdf = get_pdf(df,gridpoints=gridpoints,bandwidth=bandwidth)    
    N = complex(gridpoints) 
    slices = [slice(dim.min(),dim.max(),N) for dimname, dim in df.iteritems()]

    if len(df.columns) > 3: 
            print("DataFrame has " + str(len(df.columns)) + " dimensions. Only 3D or less can be plotted")
        
    elif len(df.columns) == 3:
        print("3D Joint Probability Density Plots TBC")

    ## If 2D Plot Surface
    elif len(df.columns) == 2:
        
        X,Y = np.mgrid[slices]
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, pdf, cmap='inferno')

        ax.set_title("Probability Density Distribution across Coupled system")
        plt.show()

    elif len(df.columns) ==1:
        X = np.mgrid[slices]
        plt.plot(np.linspace(X.min(),X.max(),gridpoints),pdf)
        plt.show()
        
def sanitise(df):
    ## Ensure data is in DataFrame form
    if isinstance(df, pd.DataFrame):
        df = df
    elif isinstance(df, pd.Series):
        df = df.to_frame()
    else:
        raise ValueError('Data passed as %s Please ensure your data is stored as a Pandas DataFrame' %(str(type(df))) )
    return df

