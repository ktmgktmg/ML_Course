# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 17:38:01 2023

@author: Admin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plotData(X,y):
    plt.figure
    plt.plot(X[np.where(y==1),0],X[np.where(y==1),1],'k+',linewidth = 2,markersize=7)
    plt.plot(X[np.where(y==0),0],X[np.where(y==0),1],'ko',markerfacecolor = 'y',markersize=7)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend('Admitted', 'Not admitted')
    plt.show()
    
def sigmoid(z):
    g = np.zeros(z.shape)
    g = 1/(1+np.exp(-z))
    return g

def costFunction(theta,X,y):
    m = len(y)
    J = 0
    grad = np.zeros(theta.shape)
    h = sigmoid(X@theta)
    #print(theta)
    #plt.figure()
    #plt.scatter(range(0,100),h)
    J = 1/m*(-y.T@np.log(h)-(1-y).T@np.log(1-h))
    grad[0] = 1/m*(X[:,0].reshape(1,m)@(h-y))
    grad[1:] = 1/m*(X[:,1:].T@(h-y))
    return J, grad

def mapFeature(X1,X2):
    degree = 6
    out = np.ones((len(X1),1))
    for i in np.arange(1,degree+1):
        for j in np.arange(0,i+1):
            out = np.concatenate((out, X1**(i-j)*(X2**j)),axis=1)
    return out

def plotDecisionBoundary(X,y,theta):
    m,n = X.shape
    if n <= 3:
        plot_x = np.array([np.min(X[:,1])-2, np.max(X[:,1])+2])
        plot_y = -1/theta[2]*(theta[1]*plot_x+theta[0])
        plt.figure(2)
        plt.plot(plot_x,plot_y,'b-',linewidth=2) 
    else: 
        u = np.linspace(-1,1.5,50)
        v = np.linspace(-1,1.5,50)
        z = np.zeros((len(u),len(v)))
        for i in np.arange(len(u)):
            for j in np.arange(len(v)):
                z[i,j] =float(mapFeature(np.array([u[i]],ndmin=2),np.array([v[j]],ndmin=2))@theta.reshape(n,1))
        z = z.T
        plt.figure
        plt.plot(X[np.where(y==1),1],X[np.where(y==1),2],'k+',linewidth = 2,markersize=7)
        plt.plot(X[np.where(y==0),1],X[np.where(y==0),2],'ko',markerfacecolor = 'y',markersize=7)
        plt.xlabel('Chip 1')
        plt.ylabel('Chip 2')
        plt.legend('Selected', 'Not selected')  
        plt.contour(u,v,z,1)
        plt.axis([-1,1.3,-1,1.3])

df = pd.read_csv('ex2data2.csv')
df.head()
print(df.head())
X= np.array(df[["x1","x2"]].values)
y = (df["y"].values).reshape(-1).T
plotData(X[:,0:2],y)

m = X.shape[0]
X = mapFeature(X[:,0].reshape(m,1),X[:,1].reshape(m,1))

m,n = X.shape
theta_ini = np.zeros((n))

cost,grad = costFunction(theta_ini, X, y)

err = 1
tol = 1e-6
alpha = 0.004
h = np.zeros((m,1))
ii = 1
while err > tol:
    cost, grad = costFunction(theta_ini,X,y)
    theta_new=theta_ini-alpha*grad
    err = np.max(np.abs(theta_ini-theta_new))
    # print(err)
    ii += 1
    # index =np.ix_([-1],np.arange(0,7))
    # theta_old = np.transpose(theta_new[index])
    theta_ini = theta_new

print('Solution converges after {} iterations. \n'.format(ii))
print(theta_new)

plotDecisionBoundary(X, y, theta_new)

def confusion_matrix(y_true, y_pred):
    # Initialize variables for true positive, true negative, false positive, and false negative
    tp = np.sum((np.array(y_pred) == 1) & (np.array(y_true) == 1))
    tn = np.sum((np.array(y_pred) == 0) & (np.array(y_true) == 0))
    fp = np.sum((np.array(y_pred) == 1) & (np.array(y_true) == 0))
    fn = np.sum((np.array(y_pred) == 0) & (np.array(y_true) == 1))

    # Print confusion matrix
    print("Confusion Matrix:")
    print(f"\t\t\t\t\t| Actual False \t| Actual True")
    print(f"Prediction False\t|     {tn}        |      {fn}")
    print(f"Prediction True\t\t|     {fp}         |      {tp}")
    return (tp,tn,fp,fn)

def f1Score(perVars):
    (tp,tn,fp,fn) = perVars
    # Compute precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print F1 score
    print(f"\nF1 Score: {f1:.2f}")
    return (f1,precision,recall)
    # Compute precision, recall, and F1 score

prediction = sigmoid(X@theta_new)
prediction[np.where(prediction>0.5)] = 1
prediction[np.where(prediction<0.5)] = 0
pervars = confusion_matrix(y, prediction)
F1,Precision,Recall = f1Score(pervars)