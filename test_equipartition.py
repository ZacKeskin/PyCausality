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


filepath = os.path.join(os.getcwd(), 'PyCausality','Testing','Test_Utils','test_data.csv')

DF = pd.read_csv(filepath)
DF.set_index('date',inplace=True)



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
        self.dict = {i:[] for i in range(MAX_DEPTH)}

        ## Calculate bins, starting from x-axis 
        self.split_cells(self.base_cell, self.DF, 0)


    def split_points(self, DF, column, location):
        left = DF[DF.iloc[:,column] < location].dropna(how='all') #.iloc[:,column] 
        right = DF[DF.iloc[:,column] >= location].dropna(how='all') #.iloc[:,column] 
        return left,right


    def split_cells(self, cell, cellDF, location):
        
        self.counter +=1
        self.depth = np.log2(self.counter+1)//1

        ## Increment the column so that each slice in the same direction per layer
        column = int(self.depth % len(cellDF.columns.values))

        ## Calculate the points in each new cell    
        leftdf, rightdf = self.split_points(cellDF,column,location)
        
        ## Calculate the domain of each new cell
        left = deepcopy(cell)
        right = deepcopy(cell)
        left[column][1] = np.nan_to_num(location)
        right[column][0] = np.nan_to_num(location)
        
        
        if self.depth not in self.dict:
            self.dict[self.depth] = []
        
        print(self.depth)
        while self.depth < self.max_depth:   
            ## For each new cell, split cells along next axis

            self.dict[self.depth] =   list(itertools.chain(
                                    self.dict[self.depth], 
                                    self.split_cells(cell=left, cellDF=leftdf, location=leftdf.iloc[:,column].median()),
                                    self.split_cells(cell=right, cellDF=rightdf, location=rightdf.iloc[:,column].median())
                                ))

        return left, right

    @property
    def bins(self):
        return self.dict[self.max_depth]

## Initialise cell:
DF = DF[['S1']].diff()[1:]
n_axes = len(DF.columns.values)



MAX_DEPTH = 3
bins = Equipartition(DF,MAX_DEPTH).bins
print(bins)

## Compare that all bins contain (roughly)equal numbers:
for i,bin in enumerate(bins):
    print(bin)
    print(len(DF.loc[   ( DF['S1']  >= bin[0][0]) &
                        ( DF['S1']   < bin[0][1]) ].dropna(how='all')))


import matplotlib.pyplot as plt


## 1D Plot
fig, axes = plt.subplots(figsize=(4, 3.5))
#axes.scatter(DF['S1'],[1 for i in range(len(DF))])
axes.hist(DF.values,bins=15)
axes.vlines([bin[0] for bin in bins], ymin=0,ymax=200)

## 2D Plot
#plt.scatter(DF['S1'],DF['S2'])

plt.show()