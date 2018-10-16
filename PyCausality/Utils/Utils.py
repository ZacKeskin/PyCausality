import pandas as pd
import numpy as np
from numpy import ma, atleast_2d, pi, sqrt, sum, transpose
from scipy import stats, optimize, linalg, special
#from scipy.special import gammaln, logsumexp
from scipy._lib.six import callable, string_types
#from scipy.stats.mstats import mquantiles

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.cm as cm           

import warnings, sys
import os
from copy import deepcopy
import traceback
from inspect import getouterframes, currentframe
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mpl_toolkits.mplot3d as mp3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

##############################################################################################################
###   U T I L I T Y    C L A S S E S
############################################################################################################## 

class NDHistogram():
    """
        Custom histogram class replacing the numpy implementations previously implemented.
        The class creates variable-size histogram calculations for N-dimensional data. 
    """

    def __init__(self, df, bins=None): 
        """
        Arguments:
            df          -   DataFrame of N columns
            bins        -   List of Lists (of Lists) defining Bin edges. This must take the form:
                            [   
                             [x0, x1], [y0,y1], [z0,z1],
                             [x0, x1], [y0,y1], [z0,z1],
                             [x0, x1], [y0,y1], [z0,z1],
                            ]
                            where each row represents a bin (in this example using 3 dimensions), and the 
                            elements of each row contain the position in each dimension of the bin-edges. 
                            The result is cuboidal bins which are highly customisable in N-d space.

                            Note: it is important for the user to define appropriate bins which capture the data.
                            It is highly recommended that the bins all join to form a contiguous space, rather
                            like a game of tetris. Note that data points which fall outside the defined bins will
                            be ignored, potentially impacting probability estimates.
                            Equally, care must be taken to avoid overlapping bins, or else there will be double counting.
                            It is recommended to use the provided classes to generate bin edges automatically.

        Returns:
            self.pdf    -   This is an N-dimensional Probability Density Function, stored as a
                            Numpy histogram, representing the proportion of samples in each bin.

        """

        self.DF = sanitise(df) 
        self.n_dims = len(df.columns.values)

        if bins is None:
            self.bins = self.sanitise_bins(self.auto_bins())
        else:
            self.bins = self.sanitise_bins(bins)

        # Initialise empty histogram array
        self._hist_ = np.array([0 for i in range(len(self.bins))])

        for i, bin in enumerate(self.bins):
            # For each bin, count the number of datapoints inside it
            self._hist_[i] =    np.sum(
                                    np.all( a =     (self.DF.iloc[:,:] >= bin[:,0][:]) & 
                                                    (self.DF.iloc[:,:] <  bin[:,1][:]) ,
                                            axis = 1))

        ## TODO: vectorise the process of counting samples in each bin
        """
        self._hist_ = np.array([  
                    np.sum(
                                    np.all( a =     (self.DF.iloc[:,:] >= bin[:,0][:]) & 
                                                    (self.DF.iloc[:,:] <  bin[:,1][:]) ,
                                            axis = 1))
                for i in range(len(self.bins)) ], dtype=np.int32)[:,0]
        """

        ## Estimate PDF
        self.pdf = self._hist_/self._hist_.sum()    


    def sanitise_bins(self,bins):
        ## TODO: Potentially store the bins as a dict, so we have unique identifiers for each bin
        """
            Generate a dict of BIN_ID:[[x_min, x_max], [y_min,ymax]] etc. 
        """
        #bin_dict = {i:bin for (i,bin) in enumerate(bins)}
        #return bin_dict
        return np.array(bins)
    
    @property
    def hist(self):
        ## TODO: Potentially store the bins as a dict, so we have unique identifiers for each bin
        """
        #    Generate a dict of BIN_ID: Count of items therein
        """
        return self._hist_
    
    def auto_bins(self):
        return Equipartition(self.DF,3).bins
    
    def plot(self):
        """
        Function to plot the histogram in 1, 2 or 3 dimensions depending on data.
            
        Arguments:
            N/A
        Returns:
            axes         -       AxesSubplot object. To be used like:                                
                                    ## Generate Histogram
                                    hist = Custom_Histogram(DF, bins)
                                    ## Plot Histogram
                                    axes = hist.plot()
                                    plt.show()
        """
        axes = None

        ## 1D Matplotlib Histogram Plot
        if self.n_dims == 1:
            fig, axes = plt.subplots(figsize=(4, 3.5))
            
            axes.scatter(self.DF,[1 for i in range(len(self.DF))])
            axes.hist(self.DF.values,bins=100)
            axes.vlines([bin[0] for bin in self.bins], ymin=0,ymax=50)


        ## 2D Matplotlib Histogram Plot
        elif self.n_dims == 2:
            fig, axes = plt.subplots(figsize=(4, 3.5))

            plt.scatter(self.DF.iloc[:,0],self.DF.iloc[:,1], 5, 'k')

            # Create a Rectangle patch for each bin and plot
            for i,bin in enumerate(self.bins):
                rect = patches.Rectangle(   (bin[0][0],bin[1][0]),
                                            bin[0][1]-bin[0][0],
                                            bin[1][1]-bin[1][0],
                                            linewidth=1,
                                            edgecolor='r',facecolor='none')
                axes.add_patch(rect)


        ## 3D Matplotlib Histogram Plot
        elif self.n_dims ==3:
            fig = plt.figure()
            axes = fig.add_subplot(111, projection='3d')


            axes.scatter(   self.DF.iloc[:,0],
                            self.DF.iloc[:,1],
                            self.DF.iloc[:,2],
                            c='k',
                            s=2.5,
                            marker='.'
            )

            for bin in self.bins:
                ## Vertices of each bin
                x =    [bin[0][0],bin[0][1],
                        bin[0][1],bin[0][0],
                        bin[0][0],bin[0][1],
                        bin[0][1],bin[0][0]]

                y =    [bin[1][0],bin[1][0],
                        bin[1][1],bin[1][1],
                        bin[1][0],bin[1][0],
                        bin[1][1],bin[1][1]]

                z =    [bin[2][0],bin[2][0],
                        bin[2][0],bin[2][0],
                        bin[2][1],bin[2][1],
                        bin[2][1],bin[2][1]]
                # Vertices
                v = [
                            [x[0],y[0],z[0]],
                            [x[1],y[1],z[1]],
                            [x[2],y[2],z[2]],
                            [x[3],y[3],z[3]],
                            [x[4],y[4],z[4]],
                            [x[5],y[5],z[5]],
                            [x[6],y[6],z[6]],
                            [x[7],y[7],z[7]]
                            ]

                faces =     np.array([
                            [v[0],v[1],v[2],v[3]],
                            [v[0],v[1],v[5],v[4]],
                            [v[3],v[0],v[4],v[7]],
                            [v[1],v[2],v[6],v[5]],
                            [v[4],v[5],v[6],v[7]],
                            [v[3],v[2],v[6],v[7]],
                            ])


                # plot sides
                cube = mp3d.art3d.Poly3DCollection(faces, 
                                    linewidths=0.75)

                cube.set_facecolor((0.9, 0.8, 0.8, 0.15))
                cube.set_edgecolor((1,0,0,0.8))
                axes.add_collection3d(cube)

        else:
            warnings.warn('The dimensions of your data exceed the maximum for plotting (3D)')
        return axes

class Equipartition():
    """
        Factory class to generate bin edges for custom, contiguous bins in n-dimensions.

        Uses recursion to split data along dimensional medians, similar to kd-trees.

        Returns bins suited for custom histogram, with roughly equal probability
    """

    def __init__(self, df, max_depth=3):
        ## Initialise class properties
        self.DF = sanitise(df)
        self.n_axes = len(self.DF.columns.values)
        self.base_cell = [ [self.DF.iloc[:,i].min(), self.DF.iloc[:,i].max()] 
                            for i,axis in enumerate(self.DF.columns.values) ] 
        self.counter = 0
        self.max_depth = max_depth
        #self.bins = set() #{i:[] for i in range(MAX_DEPTH)} 
        self.bins = []


        ## Define baseline for depth of recursion
        self.stack_level = len(getouterframes(currentframe()))

        ## Calculate bins, starting from x-axis 
        self.split_cells(self.base_cell, self.DF, self.DF.iloc[:,0].median())
        

    def split_points(self, DF, column, location):
        left = DF[DF.iloc[:,column] < location].dropna(how='all') #.iloc[:,column] 
        right = DF[DF.iloc[:,column] >= location].dropna(how='all') #.iloc[:,column] 
        return left,right


    def split_cells(self, cell, cellDF, location):
        
        ## Calculate depth on stack using inspect frames
        self.depth = len(getouterframes(currentframe())) - (self.stack_level)

        ## Increment the column so that each slice in the same direction per layer
        column = (1 + self.depth) % len(self.DF.columns.values)

        ## Calculate the domain of each new cell
        left = deepcopy(cell)
        right = deepcopy(cell)
        left[column][1] =  np.nan_to_num(location)
        right[column][0] = np.nan_to_num(location)
        
        ## Calculate data for each new cell
        leftdf,rightdf = self.split_points(cellDF,column,location)
        
        ## Update column for next location check
        column =  (1+column) % len(self.DF.columns.values) ## the (1+) is important but can't recall why


        ## For each new cell, recursively call split_cells for each subdivision, splitting along the next axis
        if self.depth < self.max_depth-1:    ## Last loop should be before max_depth is reached
           
            lleft, lright = self.split_cells(cell=left, cellDF=leftdf,location=leftdf.iloc[:,column].median())
            rleft, rright = self.split_cells(cell=right, cellDF=rightdf,location=rightdf.iloc[:,column].median())
        
        if self.depth == self.max_depth-1: 
            
            self.bins.extend(self.split_cells(cell=left, cellDF=leftdf,location=leftdf.iloc[:,column].median()) )
            self.bins.extend(self.split_cells(cell=right, cellDF=rightdf,location=rightdf.iloc[:,column].median()) )
           

        return left, right

    ## TODO: Potentially return the bins as a dict, so we have unique identifiers for each bin
    #@property
    #def bins(self)
        # Once we swap the self.bins above for self._bins_

class AutoBins():
    
    def __init__(self, df, lag=None):
        """
        Args:
            df      -   (DateFrame) Time series data to classify into bins
            lag     -   (float)     Lag for data to provided bins for lagged columns also
        Returns:
            n/a
        """
        ## Ensure data is in DataFrame form
        self.DF = sanitise(df)
        self.axes = self.df.columns.values
        self.ndims = len(self.axes)
        self.N = len(self.df)
        self.lag = lag

    def equiprobable_bins(self,max_bins=15):
        bins = Equipartition(self.DF,np.log2(max_bins))
        return bins
    


class oldNDHistogram():
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
        if self.bins is None:
            AB = AutoBins(self.df)
            self.bins = AB.sigma_bins(max_bins=max_bins)
        elif set(self.bins.keys()) != set(self.axes):
            warnings.warn('Incompatible bins provided - defaulting to sigma bins')
            AB = AutoBins(self.df)
            self.bins = AB.sigma_bins(max_bins=max_bins)
            
        ordered_bins = [sorted(self.bins[key]) for key in sorted(self.bins.keys())]

        ## Create ND histogram (np.histogramdd doesn't scale down to 1D)
        if self.n_dims == 1:
            self.hist, self.Dedges = np.histogram(self.df.values,bins=ordered_bins[0], normed=False)
        elif self.n_dims > 1:
            self.hist, self.Dedges = np.histogramdd(self.df.values,bins=ordered_bins, normed=False)
        

        ## Empirical Probability Density Function
        if self.hist.sum() == 0:   
            print(self.hist.shape)
            
            with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
                print(self.df.tail(40))

            sys.exit("User-defined histogram is empty. Check bins or increase data points")
        else:
            self.pdf = self.hist/self.hist.sum()
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

class oldAutoBins():
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
                
                I_xy = sum([H for H in HDE.H.values()]) - HDE.H_joint

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
                nk = HDE.hist

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

    def equiprobable_bins(self,max_bins=15):
        """ 
        Returns bins for N-dimensional data, such that each bin should contain equal numbers of
        samples. 
        *** Note that due to SciPy's mquantiles() functional design, the equipartion is not strictly true - 
        it operates independently on the marginals, and so with large bin numbers there are usually 
        significant discrepancies from desired behaviour. Fortunately, for TE we find equipartioning is
        extremely beneficial, so we find good accuracy with small bin counts ***

        Args:
            max_bins        -   (int)       The number of bins in each dimension
        Returns:
            bins            -   (dict)      The calculated bin-edges for pdf estimation
                                            using the histogram method, keyed by df column names
        """
        quantiles = np.array([i/max_bins for i in range(0, max_bins+1)])
        bins = dict(zip(self.axes, mquantiles(a=self.df, prob=quantiles, axis=0).T.tolist()))
        
        ## Remove_duplicates
        bins = {k:sorted(set(bins[k])) for (k,v) in bins.items()} 

        if self.lag is not None:
            bins = self.__extend_bins__(bins)
        return bins

class _kde_(stats.gaussian_kde):
    """
    Subclass of scipy.stats.gaussian_kde. This is to enable the passage of a pre-defined covariance matrix, via the
    `covar` parameter. This is handled internally within TransferEntropy class.
    The matrix is calculated on the overall dataset, before windowing, which allows for consistency between windows,
    and avoiding duplicative computational operations, compared with calculating the covariance each window.

    Functions left as much as possible identical to scipi.stats.gaussian_kde; docs available:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    """
    def __init__(self, dataset, bw_method=None, df=None, covar=None):
        self.dataset = atleast_2d(dataset)
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        self.d, self.n = self.dataset.shape
        self.set_bandwidth(bw_method=bw_method, covar=covar)


    def set_bandwidth(self, bw_method=None, covar=None):
        
        if bw_method is None:
            pass
        elif bw_method == 'scott':
            self.covariance_factor = self.scotts_factor
        elif bw_method == 'silverman':
            self.covariance_factor = self.silverman_factor
        elif np.isscalar(bw_method) and not isinstance(bw_method, string_types):
            self._bw_method = 'use constant'
            self.covariance_factor = lambda: bw_method
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            msg = "`bw_method` should be 'scott', 'silverman', a scalar " \
                  "or a callable."
            raise ValueError(msg)

        self._compute_covariance(covar)

    def _compute_covariance(self, covar):

        if covar is not None:
            self._data_covariance = covar
            self._data_inv_cov = linalg.inv(self._data_covariance)

        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            self._data_covariance = atleast_2d(np.cov(self.dataset, rowvar=1,
                                               bias=False))
            self._data_inv_cov = linalg.inv(self._data_covariance)

        self.covariance = self._data_covariance * self.factor**2
        self.inv_cov = self._data_inv_cov / self.factor**2
        self._norm_factor = sqrt(linalg.det(2*pi*self.covariance)) * self.n


##############################################################################################################
###   U T I L I T Y    F U N C T I O N S 
##############################################################################################################    
    

def get_pdf(df, gridpoints=None, bandwidth=None, estimator=None, bins=None, covar=None):
    """
        Function for non-parametric density estimation

    Args:
        df          -       (DataFrame) Samples over which to estimate density
        gridpoints  -       (int)       Number of gridpoints when integrating KDE over 
                                        the domain. Used if estimator='kernel'
        bandwidth   -       (float)     Bandwidth for KDE (scalar multiple to covariance
                                        matrix). Used if estimator='kernel'
        estimator   -       (string)    'histogram' or 'kernel'
        bins        -       (Dict of lists) Bin edges for NDHistogram. Used if estimator = 'histogram'
        covar       -       (Numpy ndarray) Covariance matrix between dimensions of df. 
                                        Used if estimator = 'kernel'
    Returns:
        pdf         -       (Numpy ndarray) Probability of a sample being in a specific 
                                        bin (technically a probability mass)
    """
    DF = sanitise(df)
    
    if estimator == 'histogram':
        pdf = pdf_histogram(DF, bins)
    else:
        pdf = pdf_kde(DF, gridpoints, bandwidth, covar)
    return pdf

def pdf_kde(df, gridpoints=None, bandwidth=1, covar=None):
    """
        Function for non-parametric density estimation using Kernel Density Estimation

    Args:
        df          -       (DataFrame) Samples over which to estimate density
        gridpoints  -       (int)       Number of gridpoints when integrating KDE over 
                                        the domain. Used if estimator='kernel'
        bandwidth   -       (float)     Bandwidth for KDE (scalar multiple to covariance
                                        matrix).
        covar       -       (Numpy ndarray) Covariance matrix between dimensions of df. 
                                        If None, these are calculated from df during the 
                                        KDE analysis

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
    kernel = _kde_(values, bw_method=bandwidth, covar=covar)
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

def get_entropy(df, gridpoints=15, bandwidth=None, estimator='kernel', bins=None, covar=None):
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
        covar       -       (Numpy ndarray) Covariance matrix between dimensions of df. 
                                        Used if estimator = 'kernel'
    Returns:
        entropy     -       (float)     Shannon entropy in bits

    """
    pdf = get_pdf(df, gridpoints, bandwidth, estimator, bins, covar)
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

def plot_pdf(df,estimator='kernel',gridpoints=None, bandwidth=None, covar=None, bins=None, show=False,
            cmap='inferno', label_fontsize=7):
    """
    Wrapper function to plot the pdf of a pandas dataframe
        
    Args:
        df          -       (DataFrame) Samples over which to estimate density
        estimator   -       (string)    'kernel' or 'histogram'
        gridpoints  -       (int)       Number of gridpoints when integrating KDE over 
                                        the domain. Used if estimator='kernel'
        bandwidth   -       (float)     Bandwidth for KDE (scalar multiple to covariance
                                        matrix). Used if estimator='kernel'
        covar       -       (Numpy ndarray) Covariance matrix between dimensions of df. 
        bins        -       (Dict of lists) Bin edges for NDHistogram. Used if estimator = 'histogram'
        show        -       (Boolean)   whether or not to plot direclty, or simply return axes for later use
        cmap        -       (string)    Colour map (see: https://matplotlib.org/examples/color/colormaps_reference.html)
        label_fontsize -    (float)     Defines the fontsize for the axes labels

    Returns:
        ax          -       AxesSubplot object. Can be added to figures to allow multiple plots.
    """
    
    DF = sanitise(df)
    if len(DF.columns) != 2: 
            print("DataFrame has " + str(len(DF.columns)) + " dimensions. Only 2D or less can be plotted")
            axes = None
    else:
        ## Plot data in Histogram or Kernel form
        if estimator == 'histogram':
            
            if bins is None:
                bins = {axis:np.linspace(DF[axis].min(),
                                        DF[axis].max(),
                                        9) for axis in DF.columns.values}
            fig, axes = plot_pdf_histogram(df,bins,cmap)
        else:
            fig, axes = plot_pdf_kernel(df, gridpoints, bandwidth, covar,cmap)

        ## Format plot
        axes.set_xlabel(DF.columns.values[0],labelpad=20)
        axes.set_ylabel(DF.columns.values[1],labelpad=20)
        for label in axes.xaxis.get_majorticklabels():
            label.set_fontsize(label_fontsize)
        for label in axes.yaxis.get_majorticklabels():
            label.set_fontsize(label_fontsize)
        for label in axes.zaxis.get_majorticklabels():
            label.set_fontsize(label_fontsize)
        axes.view_init(10, 45)
        if show == True:
            plt.show()
        plt.close(fig)
        
        axes.remove()

    return axes

def plot_pdf_histogram(df,bins, cmap='inferno'):
    """
    Function to plot the pdf of a dataset, estimated via histogram.
        
    Args:
        df          -       (DataFrame) Samples over which to estimate density
        bins        -       (Dict of lists) Bin edges for NDHistogram. Used if estimator = 'histogram'

    Returns:
        ax          -       AxesSubplot object, passed back via to plot_pdf() function
    """
    DF = sanitise(df) # in case function called directly


    ## Calculate PDF
    PDF = get_pdf(df=DF,estimator='histogram',bins=bins)

    ## Get x-coords, y-coords for each bar
    (x_edges,y_edges) = bins.values()
    X, Y = np.meshgrid(x_edges[:-1], y_edges[:-1])
    ## Get dx, dy for each bar
    dxs, dys = np.meshgrid(np.diff(x_edges),np.diff(y_edges))

    ## Colourmap
    cmap = cm.get_cmap(cmap) 
    rgba = [cmap((p-PDF.flatten().min())/PDF.flatten().max()) for p in PDF.flatten()] 

    ## Create subplots
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.bar3d(   x = X.flatten(),            #x coordinates of each bar
                y = Y.flatten(),            #y coordinates of each bar
                z = 0,                      #z coordinates of each bar
                dx = dxs.flatten(),         #width of each bar
                dy = dys.flatten(),         #depth of each bar
                dz = PDF.flatten() ,        #height of each bar
                alpha = 1,                  #transparency
                color = rgba
    )
    ax.set_title("Histogram Probability Distribution",fontsize=10)

    return fig, ax

def plot_pdf_kernel(df,gridpoints=None, bandwidth=None, covar=None, cmap='inferno'):
    """
        Function to plot the pdf, calculated by KDE, of a dataset
        
    Args:
        df          -       (DataFrame) Samples over which to estimate density
        gridpoints  -       (int)       Number of gridpoints when integrating KDE over 
                                        the domain. Used if estimator='kernel'
        bandwidth   -       (float)     Bandwidth for KDE (scalar multiple to covariance
                                        matrix). Used if estimator='kernel'
        covar       -       (Numpy ndarray) Covariance matrix between dimensions of df. 
                                        
    Returns:
        ax          -       AxesSubplot object, passed back via to plot_pdf() function
    """
    DF = sanitise(df)
    ## Estimate the PDF from the data
    if gridpoints is None:
        gridpoints = 20

    pdf = get_pdf(DF,gridpoints=gridpoints,bandwidth=bandwidth)    
    N = complex(gridpoints) 
    slices = [slice(dim.min(),dim.max(),N) for dimname, dim in DF.iteritems()]
    X,Y = np.mgrid[slices]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, pdf, cmap=cmap)
    

    ax.set_title("KDE Probability Distribution",fontsize=10)

    return fig, ax

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

