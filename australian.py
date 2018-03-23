import sys

import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

sys.setrecursionlimit(10000) # Recursion limit for KDTree.

# Australian dataset characteristics
COLUMN_NAMES = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10','A11', 'A12', 'A13', 'A14', 'Y']


# Classes 
class ExploratoryAnalysis(object):
	"""
	Performs exploratory analysis
	"""
	def __init__(self):
		self.data = None
		self.grouped = None
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
		self.grouped = self.data.groupby('Y')

# Exploratory analysis 
australian = ExploratoryAnalysis()
australian.load_csv_file('australian.dat')

australian.data.columns # There are 15 columns, 14 variables and 1 value to classify
australian.data.describe() # Y values seems balanced for classification purpose.
pd.isna(australian.data).sum() # There are no missing values
australian.data.corr() # A08, A09, A10 look like relevant variables

# Category vs continuous variables
for name in COLUMN_NAMES:
	print(name + ' -> ' + str(len(pd.unique(australian.data[name]))))
	# Categorical: A01, A04, A06, A08, A09, A11, A12, Y
	# Not sure, but defined as continuous: A05, A10
	# Continuous: A02, A03, A07, A13, A14

#for cat_var in ['A01', 'A04', 'A06', 'A08', 'A09', 'A11', 'A12', 'Y']:
#	australian.data[cat_var] = australian.data[cat_var].astype('category')
	# Does not seem very useful in pandas, so it remains commented.

# Plots 
# TODO: implement plots in a method for the class
# 	Histogram for A02
fig, ax = plt.subplots(1,1)
for name, group in australian.grouped:
	ax.hist(group.A02, alpha = 0.5, label = 'Y = ' + str(name), bins=20)

ax.legend()
ax.set_title('A02') 
plt.show()

# 	Histogram for all variables
fig, ax_array = plt.subplots(4,4)
[[ax1,ax2,ax3,ax4], [ax5,ax6,ax7,ax8], [ax9,ax10,ax11,ax12], [ax13,ax14,_,_]] = ax_array
for i in np.arange(14):
	var = COLUMN_NAMES[i]
	ax = ax_array.flatten()[i]
	for name, group in australian.grouped:
		ax.hist(group[var], alpha = 0.5, label = 'Y = ' + str(name))

fig.legend()
plt.show()

# Commentaries ----------------------------------------------------------------
#TODO: mirar sns (SEABORN) para análisis multivariables. Libros:
#- Pandas: Pandas Cookbook
#- seaborn: Python for Data Analysis, 2nd Edition
#TODO: implement KNN classification
