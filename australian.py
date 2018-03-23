import sys

import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

sys.setrecursionlimit(10000) # Recursion limit for KDTree.

# Australian dataset characteristics
COLUMN_NAMES = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10','A11', 'A12', 'A13', 'A14', 'Y']

# Classes 
class ExploratoryAnalysis(object):
	"""
	Performs exploratory analysis
	"""
	def __init__(self):
		self.data = None
		self.regressors = None
		self.labels = None

	def load_csv_file(self, csv_file, comment = '@', limit = None, header = None, names = COLUMN_NAMES):
		"""
		Loads CSV with australian data
		:param csv_file: CSV file name
		:param comment: str that indicates lines commented
		:param limit: number of rows of the file to read
		"""
		self.data = pd.read_csv(csv_file, nrows = limit, comment = comment, header = header, names = names)

# Exploratory analysis
australian = ExploratoryAnalysis()
australian.load_csv_file('australian.dat')

australian.data.columns # There are 15 columns, 14 variables and 1 value to classify
australian.data.describe() # Y values seems balanced for classification purpose.
pd.isna(australian.data).sum() # There are no missing values

# Category VS continuous variables
for name in COLUMN_NAMES:
	print(name + ' -> ' + str(len(pd.unique(australian.data[name]))))
	# Categorical: A1, A4, A6, A8, A9, A11, A12, Y
	# Not sure, but defined as continuous: A5, A10
	# Continuous: A2, A3, A7, A13, A14

for cat_var in ['A1', 'A4', 'A6', 'A8', 'A9', 'A11', 'A12', 'Y']:
	australian.data[cat_var] = australian.data[cat_var].astype('category')

# Plots


