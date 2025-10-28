# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 12:19:09 2023

@author: hp
"""
#%% Data imputation
import numpy as np
from sklearn.impute import SimpleImputer
import scipy.sparse as sp
import pandas as pd

imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent') #mean, median, constant
imp.fit([[1, 2], [np.nan, 3], [7, 6]])
print(imp.statistics_)
X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(X)
X=imp.transform(X)
print(X)
#%% scattered matrix
X = sp.csc_matrix([[-1, -1], [0, -1], [8, 4]])
print(X)
imp = SimpleImputer(missing_values=-1, strategy='mean')
imp.fit(X)
X_test = sp.csc_matrix([[-1, 2], [6, -1], [7, 6]])
print(imp.transform(X_test).toarray())
#%% most_frequent
df = pd.DataFrame([["a", "x"],
                   [np.nan, "y"],
                   ["a", np.nan],
                   ["b", "y"]], dtype="category")
print(df)
imp = SimpleImputer(strategy="most_frequent")
df2 = imp.fit_transform(df)
print(df2)
#%%
A = np.array([["Y","N","Y","N"],
              ["Y","N","?","N"],
              ["Y","N","Y","?"],
              ["?","N","Y","Y"]])
print(A)
df = pd.DataFrame(A,columns=["Airconditioning","Garage","Pool","Verandah"])
print(df.head())
imp = SimpleImputer(missing_values="?", strategy='most_frequent')
df2 = pd.DataFrame(imp.fit_transform(df))
df2.head()
#%%
A = np.array([["Y","N","Y","N"],
              ["Y","N","NA","N"],
              ["Y","N","Y","Y"],
              ["NA","N","Y","Y"]])
print(A)
df = pd.DataFrame(A,columns=["Airconditioning","Garage","Pool","Verandah"])
print(df.head())
imp = SimpleImputer(missing_values="NA", strategy='most_frequent')
df2 = pd.DataFrame(imp.fit_transform(df))
df2.head()
#%% K-mean Algorithm
import numpy as np
from sklearn.impute import KNNImputer #KNNImpute machine learning based K-mean algorithm
nan = np.nan
X = [[1, 0, nan], [3, 5, 1], [nan, 6, 5], [8, 8, nan]]
print(X)
imp = KNNImputer(n_neighbors=2, weights="uniform")
X=imp.fit_transform(X)
print(X)