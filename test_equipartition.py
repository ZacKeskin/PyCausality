import numpy as np
import pandas as pd
import os
from copy import deepcopy
import itertools

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
    return df.astype(dtype=np.float32) # Otherwise issue with building histograms


import traceback
from inspect import getouterframes, currentframe
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Custom_Histogram():

    def __init__(self, df, bins=None): 
        """
        
        """
        k=2
        self.DF = sanitise(df) 
        self.bins = self.sanitise_bins(bins)
        self.n_dims = len(df.columns.values)
        self.n_vertices = 2**self.n_dims
        self.n_cubes = 2**k

        if bins is None:
            self.bins = self.auto_bins()
        else:
            self.bins = self.sanitise_bins(bins)

        
        self._hist_ = np.array([  
            np.product(  
                    np.sum(     (self.DF.iloc[:,:] >= self.bins[:,:,0][i][:]) & 
                                (self.DF.iloc[:,:] <  self.bins[:,:,1][i][:])
                           ))/len(self.DF) 
                for i in range(len(self.bins)) ], dtype=np.int32)[:,0]
        

    def sanitise_bins(self,bins):
        """
            Generate a dict of BIN_ID:[[x_min, x_max], [y_min,ymax]] etc. 
        """
        #bin_dict = {i:bin for (i,bin) in enumerate(bins)}
        #return bin_dict
        return np.array(bins)
    
    @property
    def hist(self):
        """
        #    Generate a dict of BIN_ID: Count of items therein
        """
        return self.hist
    
    
    def plot(self):

        fig, axes = plt.subplots(figsize=(4, 3.5))

        if len(self.DF.columns.values) == 1:
            ## 1D Plot
            axes.scatter(self.DF,[1 for i in range(len(self.DF))])
            axes.hist(self.DF.values,bins=100)
            axes.vlines([bin[0] for bin in self.bins], ymin=0,ymax=50)


        elif len(self.DF.columns.values) == 2:
            ## 2D Plot
            plt.scatter(self.DF.iloc[:,0],self.DF.iloc[:,1], 5, 'k')

            # Create a Rectangle patch for each bin and plot
            for i,bin in enumerate(self.bins):
                rect = patches.Rectangle(   (bin[0][0],bin[1][0]),
                                            bin[0][1]-bin[0][0],
                                            bin[1][1]-bin[1][0],
                                            linewidth=1,
                                            edgecolor='r',facecolor='none')
                axes.add_patch(rect)

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
        column = (1 + column) % len(self.DF.columns.values)

        ## For each new cell, recursively call split_cells for each subdivision, splitting along the next axis
        if self.depth < self.max_depth-1:    ## Last loop should be before max_depth is reached
           
            lleft, lright = self.split_cells(cell=left, cellDF=leftdf,location=leftdf.iloc[:,column].median())
            rleft, rright = self.split_cells(cell=right, cellDF=rightdf,location=rightdf.iloc[:,column].median())
        
        if self.depth == self.max_depth-1: 
            
            self.bins.extend(self.split_cells(cell=left, cellDF=leftdf,location=leftdf.iloc[:,column].median()) )
            self.bins.extend(self.split_cells(cell=right, cellDF=rightdf,location=rightdf.iloc[:,column].median()) )
           


        return left, right

    #@property
    #def bins(self)
        # Once we swap the self.bins above for self._bins_

