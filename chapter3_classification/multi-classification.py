# %%
# load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.datasets import fetch_openml

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

# my model performance visualization file
from performance_eval import performance_vs_thresholds
from performance_eval import plot_digits
from performance_eval import plot_performance_curve
from performance_eval import refit_model_predict
from performance_eval import evaluate_model
from performance_eval import explore_metrics
from performance_eval import explore_metrics

 

#%%
# define constants

MODELS_PATH = './models/'
#%%
# load data

mnist = pd.read_csv('./data/mnist.csv')

#%%
# train test split
X = mnist.drop(columns='target')
y = mnist['target']

X_train = X.iloc[:60000, :]
X_test = X.iloc[60000:, :]

y_train = y[:60000]
y_test = y[60000:]
# %%
