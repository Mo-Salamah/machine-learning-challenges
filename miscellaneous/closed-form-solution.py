#%%
import numpy as np
from numpy import linalg
import pandas as pd
# %%
X = np.array([[1,2],
              [1,4],
            #   [1,6],
            #   [1,33],
              [1,5],])

m = X.shape[0]
x = X[:, 1]

X_new = np.zeros(m)
 
for i in range(m):
    x_i = x
    x_i = x_i ** i 
    X_new = np.column_stack((X_new, x_i))
    print(X_new)

X_new = X_new[:, 1:]
# y = np.random.randint(0, 100, m)
y = [2,4,6]

#%%

X = np.array([[175, 75, 1370], 
              [160, 85, 1000],
              [181, 70, 1100],
              [150, 50, 1500],
              [155, 42, 1400]])

def standardize(X):
    m, n = X.shape

    Z = np.zeros(X.shape)

    
    for i in range(n):
        column = X[:, i]
        mean = np.mean(column)
        std = np.std(column)
        
        Z[:, i] = (column - mean) / std
    
    return Z
        








# %%

