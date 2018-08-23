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
    return df


import traceback
from inspect import getouterframes, currentframe
import matplotlib.patches as patches

### We actually need a new histogram - from the ground up, with variable bins in N dimensions (based on
class Custom_Histogram():

    def __init__(self, df, bins=None): 
        """
        bins - List of lists of , coordinates for each hypercube
        """
        k=2
        self.DF = sanitise(df) 
        self.bins = self.sanitise_bins(bins)
        self.n_dims = len(df.columns.values)
        self.n_vertices = 2**self.n_dims
        self.n_cubes = 2**k

        cubes = np.zeros(shape=(self.n_cubes, self.n_vertices))
        cubes[0][:] = np.array([0,10,0,10])
        
        
        for i in range(2**k):
            ## Copy current cube
            old_cube = deepcopy(cubes[i-1][:])
            cubes[i][:] = np.vstack(old_cube[:int(self.n_vertices/2)],)

        print(cubes)

    def sanitise_bins(self,bins):
        return bins


class Equipartition():
    """
        Factory class to generate bin edges for custom, contiguous bins in n-dimensions.
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
        #print(self.depth, len(self.DF.columns.values), column)
        ## Calculate the domain of each new cell
        left = deepcopy(cell)
        right = deepcopy(cell)
        left[column][1] =  np.nan_to_num(location)
        right[column][0] = np.nan_to_num(location)
        
        ## Calculate data for each 
        #location = cellDF.iloc[:,column].median()
        leftdf,rightdf = self.split_points(cellDF,column,location)
        
        ## Update column for next location check
        column = (1 + column) % len(self.DF.columns.values)

        ## For each new cell, recursively call split_cells for each subdivision, splitting along the next axis
        if self.depth < self.max_depth-1:    ## Last loop should be before max_depth is reached
           
            lleft, lright = self.split_cells(cell=left, cellDF=leftdf,location=leftdf.iloc[:,column].median())
            rleft, rright = self.split_cells(cell=right, cellDF=rightdf,location=rightdf.iloc[:,column].median())
        
        if self.depth == self.max_depth-1: 
            
            self.bins.extend(self.split_cells(cell=left, cellDF=leftdf,location=leftdf.iloc[:,column].median()) )
            #self.bins.append(self.split_cells(cell=left, cellDF=leftdf,location=leftdf.iloc[:,column].median())[1])
            self.bins.extend(self.split_cells(cell=right, cellDF=rightdf,location=rightdf.iloc[:,column].median()) )
            #self.bins.append(self.split_cells(cell=right, cellDF=rightdf,location=rightdf.iloc[:,column].median())[1])


        return left, right



filepath = os.path.join(os.getcwd(), 'PyCausality','Testing','Test_Utils','test_data.csv')

DF = pd.read_csv(filepath)
DF.set_index('date',inplace=True)
DF = DF[['S1','S2']].diff()[1:]
print(DF)
"""
n_axes = len(DF.columns.values)
"""
#S1 = np.random.normal(5,2,1000)

#DF = pd.DataFrame({'S1':S1})


MAX_DEPTH = 4
bins = Equipartition(DF,MAX_DEPTH).bins
#print(bins)

## Compare that all bins contain (roughly)equal numbers:


import matplotlib.pyplot as plt


fig, axes = plt.subplots(figsize=(4, 3.5))

"""

## 1D Plot
axes.scatter(DF['S1'],[1 for i in range(len(DF))])
axes.hist(DF.values,bins=100)
axes.vlines([bin[0] for bin in bins], ymin=0,ymax=50)

for i,bin in enumerate(bins):
    print(bin)
    print(len(DF.loc[   ( DF['S1']  >= bin[0][0]) &
                        ( DF['S1']   < bin[0][1]) 
                        ].dropna(how='all')))


"""
## 2D Plot
plt.scatter(DF['S1'],DF['S2'], 5, 'k')

# Create a Rectangle patch for each bin and plot
#for spine in axes.spines:
#    spine.set_visible(False)
for i,bin in enumerate(bins):
    print(bin)
    print(len(DF.loc[   ( DF['S1']  >= bin[0][0]) &
                        ( DF['S1']   < bin[0][1]) &
                        ( DF['S2']  >= bin[1][0]) &
                        ( DF['S2']   < bin[1][1]) 
                        ].dropna(how='all')))

    #print('bottom left (x,y) = ',(bin[0][0],bin[1][0]) )
    #print('width = ', bin[0][1]-bin[0][0]  )
    #print('height = ',  bin[1][1]-bin[1][0])

    rect = patches.Rectangle(   (bin[0][0],bin[1][0]),
                                bin[0][1]-bin[0][0],
                                bin[1][1]-bin[1][0],
                                linewidth=1,
                                edgecolor='r',facecolor='none')
    # Add the patch to the Axes
    axes.add_patch(rect)

axes.set_ylim()


plt.show()