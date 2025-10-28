# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 11:02:42 2023

@author: hp
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

df=pd.read_csv("D:\Click Top\Machine Learning\diabetic_data.csv")
df.head()
col = df.columns
print(col)
n = df.count()
print(n)
print(type(df))
print(df.shape)
print(df.dtypes)
df.describe
#%%
a = pd.DataFrame([[0, 1],[3,2],[0,1],[0,1]],columns =["c1", "c2"])
print(a)
print(a.loc[a.duplicated()])
b=a.drop_duplicates()
print(b.head())
a.head()
#%%

df2 = pd.DataFrame({

    'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],

    'style': ['cup', 'cup', 'cup', 'pack', 'pack'],

    'rating': [4, 4, 3.5, 15, 5]

})
print(df2.head())
df2 = df2.drop_duplicates()
print(df2.head())
df3 = df2.drop_duplicates(subset = ["brand"])
print(df3)
df4 = df2.drop_duplicates(subset = ["style"])
print(df4)
df5 = df2.drop_duplicates(subset = ["brand","style"])
print(df5)
#%%
a = pd.DataFrame([[0, 1],[3,2],[0,np.nan],[0,1]],columns =["c1", "c2"])
print(a)
b=a.dropna()
print(b.head())
a.head()
#%% optimization (gradient descent method)

import numpy as np

def objFunc(theta):
    J = (theta[0]-5)**2 + (theta[1]-5)**2
    dJdtheta = np.array([2*(theta[0]-5), 2*(theta[1]-5)])
    return (J,dJdtheta)

err = 1
tol = 1e-6
theta = np.array([0,1])
alpha = 0.1
Iter = 0

while err > tol:
    J,dJdtheta = objFunc(theta)
    theta_new = theta - alpha*dJdtheta
    err = max(abs(theta_new-theta))
    theta = theta_new
    Iter += 1
    print(Iter)

print("Objective function J = {:.2e} at number of iterations = {:d}".format(J,Iter))
print("Optimal thetas are: theta_1 = {:.2f}, theta_2={:.2f}".format(theta[0],theta[1]))
