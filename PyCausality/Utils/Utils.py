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
    def __init__(self, df, bins=None, max_bins = 15):
        """
        Arguments:
            df          -   DataFrame passed through from the TransferEntropy class
            bins        -   Bin edges passed through from the TransferEntropy class
            max_bins    -   Number of bins per each dimension passed through from the TransferEntropy class
        Returns:
            self.pdf    -   This is an N-dimensional Probability Density Function, stored as a
                            Numpy histogram, representing the proportion of samples in each bin.
        """
        df = sanitise(df)
        self.df = df.reindex(columns= sorted(df.columns))   # Sort axes by name
        self.max_bins = max_bins
        self.axes = list(self.df.columns.values)
        self.bins = bins
        self.n_dims = len(self.axes)
        
        ## Bins must match number and order of dimensions
        if self.bins is None or set(self.bins.keys()) != set(self.axes):
            #warnings.warn('Incorrect or no bins provided - defaulting to sigma bins')
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
        Sets entropy for marginal distributions: H(X), H(Y) etc. as well as joint entropy H(X,Y)
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
    """
        Prototyping class for generating data-driven binning.
        Handles lagged time series, so only DF[X(t), Y(t)] required.
    """
    def __init__(self, df, lag=None):
        """
        Args:
            df      -   (DateFrame) Time series data to classify into bins
            lag     -   (float)     Lag for data to provided bins for lagged columns also
        Returns:
            n/a
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
            See: https://arxiv.org/pdf/physics/0605197.pdf
        Args:   
            M   -   (list) number of bins in each dimension  e.g.[Mx, My, Mz]
        Returns:
            Log Likelihood with this number of bins. (Log of marginal posterior)
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
        """
           Function to generate bins for lagged time series not present in self.df

        Args:   
            bins    -   (Dict of List)  Bins edges calculated by some AutoBins.method()
        Returns:
            bins    -   (Dict of lists) Bin edges keyed by column name
        """
        self.max_lag_only = True # still temporary until we kill this

        ## Handle lagging for bins, and calculate default bins where edges are not provided
        if self.max_lag_only == True:
            bins.update({   fieldname + '_lag' + str(self.lag): edges   
                            for (fieldname,edges) in bins.items()})  
        else:
            bins.update({   fieldname + '_lag' + str(t): edges          
                            for (fieldname,edges) in bins.items() for t in range(self.lag)})
        
        return bins

    def MIC_bins(self, max_bins=15):
        """
        Method to find optimal bin widths in each dimension, using a naive search to 
        maximise the mutual information divided by number of bins. Only accepts data
        with two dimensions [X(t),Y(t)].
        We increase the n_bins parameter in each dimension, and take the bins which
        result in the greatest Maximum Information Coefficient (MIC)
        
        (Note that this is restricted to equal-width bins only.)

        Defined:            MIC = I(X,Y)/ max(n_bins)
                            edges = {Y:[a,b,c,d], Y-t:[a,b,c,d], X-t:[e,f,g]}, 
                            n_bins = [bx,by]
        Calculated using:   argmax { I(X,Y)/ max(n_bins) }

        Args:
            max_bins        -   (int)       The maximum allowed bins in each dimension
        Returns:
            opt_edges       -   (dict)      The optimal bin-edges for pdf estimation
                                            using the histogram method, keyed by df column names
                                            All bins equal-width.
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

    def knuth_bins(self,max_bins=15):
        """
        Method to find optimal bin widths in each dimension, using a naive search to 
        maximise the log-likelihood given data. Only accepts data
        with two dimensions [X(t),Y(t)]. 

        Derived from Matlab code provided in Knuth (2013):  https://arxiv.org/pdf/physics/0605197.pdf
        
        (Note that this is restricted to equal-width bins only.)

        Args:
            max_bins        -   (int)       The maximum allowed bins in each dimension
        Returns:
            bins            -   (dict)      The optimal bin-edges for pdf estimation
                                            using the histogram method, keyed by df column names
                                            All bins equal-width.
        """
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

    def sigma_bins(self, max_bins=15):
        """ 
        Returns bins for N-dimensional data, using standard deviation binning: each 
        bin is one S.D in width, with bins centered on the mean. Where outliers exist 
        beyond the maximum number of SDs dictated by the max_bins parameter, the
        bins are extended to minimum/maximum values to ensure all data points are
        captured. This may mean larger bins in the tails, and up to two bins 
        greater than the max_bins parameter suggests in total (in the unlikely case of huge
        outliers on both sides). 

        Args:
            max_bins        -   (int)       The maximum allowed bins in each dimension
        Returns:
            bins            -   (dict)      The optimal bin-edges for pdf estimation
                                            using the histogram method, keyed by df column names
        """

        
        bins = {k:[np.mean(v)-int(max_bins/2)*np.std(v) + i * np.std(v) for i in range(max_bins+1)] 
                for (k,v) in self.df.iteritems()}   # Note: same as:  self.df.to_dict('list').items()}

        # Since some outliers can be missed, extend bins if any points are not yet captured
        [bins[k].append(self.df[k].min()) for k in self.df.keys() if self.df[k].min() < min(bins[k])]
        [bins[k].append(self.df[k].max()) for k in self.df.keys() if self.df[k].max() > max(bins[k])]

        if self.lag is not None:
            bins = self.__extend_bins__(bins)
        return bins


##############################################################################################################
###   U T I L I T Y    F U N C T I O N S 
##############################################################################################################    
    

def get_pdf(df, gridpoints=None, bandwidth=None, estimator=None, bins=None):
    """
        Function for non-parametric density estimation

    Args:
        df          -       (DataFrame) Samples over which to estimate density
        gridpoints  -       (int)       Number of gridpoints when integrating KDE over 
                                        the domain. Used if estimator='kernel'
        bandwidth   -       (float)     Bandwidth for KDE (scalar multiple to covariance
                                        matrix). Used if estimator='kernel'
        estimator   -       (string)    'histogram' or 'kernel'
        bins        -       (Dict of lists) Bin edges for NDHistogram. Used if estimator
                                        = 'histogram'
    Returns:
        pdf         -       (Numpy ndarray) Probability of a sample being in a specific 
                                        bin (technically a probability mass)
    """

    if estimator == 'histogram':
        pdf = pdf_histogram(df, bins)
    else:
        pdf = pdf_kde(df, gridpoints, bandwidth)
    return pdf

def pdf_kde(df, gridpoints = None, bandwidth = 1):
    """
        Function for non-parametric density estimation using Kernel Density Estimation

    Args:
        df          -       (DataFrame) Samples over which to estimate density
        gridpoints  -       (int)       Number of gridpoints when integrating KDE over 
                                        the domain. Used if estimator='kernel'
        bandwidth   -       (float)     Bandwidth for KDE (scalar multiple to covariance
                                        matrix).

    Returns:
        Z/Z.sum()   -       (Numpy ndarray) Probability of a sample being between
                                        specific gridpoints (technically a probability mass)
    """
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
    """
        Function for non-parametric density estimation using N-Dimensional Histograms

    Args:
        df            -       (DataFrame) Samples over which to estimate density
        bins          -       (Dict of lists) Bin edges for NDHistogram. 
    Returns:
        histogram.pdf -       (Numpy ndarray) Probability of a sample being in a specific 
                                    bin (technically a probability mass)
    """
    histogram = NDHistogram(df=df, bins=bins)        
    return histogram.pdf

def get_entropy(df, gridpoints=15, bandwidth=None, estimator='kernel', bins=None):
    """
        Function for calculating entropy from a probability mass 
        
    Args:
        df          -       (DataFrame) Samples over which to estimate density
        gridpoints  -       (int)       Number of gridpoints when integrating KDE over 
                                        the domain. Used if estimator='kernel'
        bandwidth   -       (float)     Bandwidth for KDE (scalar multiple to covariance
                                        matrix). Used if estimator='kernel'
        estimator   -       (string)    'histogram' or 'kernel'
        bins        -       (Dict of lists) Bin edges for NDHistogram. Used if estimator
                                        = 'histogram'
    Returns:
        entropy     -       (float)     Shannon entropy in bits

    """
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
    """
        Function to plot the pdf, calculated by KDE, of a dataset
        
    Args:
        df          -       (DataFrame) Samples over which to estimate density
        gridpoints  -       (int)       Number of gridpoints when integrating KDE over 
                                        the domain. Used if estimator='kernel'
        bandwidth   -       (float)     Bandwidth for KDE (scalar multiple to covariance
                                        matrix). Used if estimator='kernel'
    Returns:
        n/a
    """
    ## Estimate the PDF from the data
    if gridpoints is None:
        gridpoints = 20

    pdf = get_pdf(df,gridpoints=gridpoints,bandwidth=bandwidth)    
    N = complex(gridpoints) 
    slices = [slice(dim.min(),dim.max(),N) for dimname, dim in df.iteritems()]

    if len(df.columns) > 3: 
            print("DataFrame has " + str(len(df.columns)) + " dimensions. Only 2D or less can be plotted")
        
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
    """
        Function to convert DataFrame-like objects into pandas DataFrames
        
    Args:
        df          -        Data in pd.Series or pd.DataFrame format
    Returns:
        df          -        Data as pandas DataFrame
    """
    ## Ensure data is in DataFrame form
    if isinstance(df, pd.DataFrame):
        df = df
    elif isinstance(df, pd.Series):
        df = df.to_frame()
    else:
        raise ValueError('Data passed as %s Please ensure your data is stored as a Pandas DataFrame' %(str(type(df))) )
    return df

