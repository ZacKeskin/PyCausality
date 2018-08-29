import os
import numpy as np
import pandas as pd
from test_equipartition import Custom_Histogram, Equipartition
import matplotlib.pyplot as plt

filepath = os.path.join(os.getcwd(), 'PyCausality','Testing','Test_Utils','test_data.csv')

"""
## Generate Random Data
X = np.random.normal(0.5,0.1,500)
Y = np.random.normal(0.5,0.25,500)
## Populate a Pandas DataFrame
DF = pd.DataFrame({'x':X,'y':Y})
"""

DF = pd.read_csv(filepath)[['S1','S2']]

bins = Equipartition(DF,3).bins

hist = Custom_Histogram(DF, bins)

plt.style.use('seaborn')
axes = hist.plot()
axes.set_xlabel('S1')
axes.set_ylabel('S2')
plt.show()