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


#%%
METRICS_NAMES = {
    'tp/p': 'Recall',
    
}


# %%
# load data

# mnist = fetch_openml('mnist_784', version=1)
# mnist.keys()
mnist = pd.read_csv('./data/mnist.csv')

# %%
# split to train and test sets

# X, y = mnist['data'], mnist['target']
X, y = mnist.drop(columns=['target']), mnist.iloc[:, -1]
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000].astype('int'), y[60000:].astype('int')

# %%
def plot_digits(vector):
    """takes a 784 length vector and produces
        a plot of the digit
    Args:
        vector (itterable): 784 length with floats from 0 to 255
    """
    digit = np.array(vector).reshape((28, 28))
    plt.imshow(digit, cmap='binary')
    plt.axis('off')
    plt.show()

# %%
# visualize one digit example
some_digit = X_train.iloc[0]
plot_digits(some_digit)


# %%
# simplify classification problem

# two classes: true if 8, false otherwise
y_train_8 = (y_train == 8)
y_test_8 = (y_test == 8)

# %%
# train a classification model
from sklearn.linear_model import SGDClassifier
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

sgd_clf = SGDClassifier() # no parameters? what cost function is default?
# sgd_clf.fit(X_train, y_train_8)
# y_pred_8 = sgd_clf.predict(X_test)
# cost = 'f1'
y_pred_8 = cross_val_predict(sgd_clf, X_train, y_train_8, cv=3)

# %%
# understand errors of the model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

conf_mtrx_8 = confusion_matrix(y_true=y_train_8, y_pred=y_pred_8)
targets_sums = np.sum(conf_mtrx_8, axis=1)
norm_conf_mtrx_8 = conf_mtrx_8 / targets_sums

# %%
plt.figure(figsize=(4,4))
ConfusionMatrixDisplay.from_predictions(y_true=y_train_8, y_pred=y_pred_8)

# %%
# calculare different scoring metrics
def assess_performance(y_true, y_pred, majority='negative'):
    """calculates the different classification 
    metrics for binary classification

    Args:
        y_true (itterable): array of {0, 1} 
        y_pred (itterable): array of predicted {0, 1}
        majority (str)   : {'negative', 'positive'} designation of majority class
    """
    conf_mtrx = confusion_matrix(y_true, y_pred)
    
    sums = np.sum(conf_mtrx, axis=1)
    majority_index = np.argmax(sums)
    
    if majority == 'positive':
        positive_index = majority_index
        if majority_index == 0:
            negative_index = 1
        else:
            negative_index = 0 
    else:
        negative_index = majority_index
        if majority_index == 0:
            positive_index = 1
        else:
            positive_index = 0 
        
    # marginal sums
    n_positive = np.sum(conf_mtrx[positive_index, :])
    n_negative = np.sum(conf_mtrx[negative_index, :])
    
    n_pred_positive = np.sum(conf_mtrx[:, positive_index])
    n_pred_negative = np.sum(conf_mtrx[:, negative_index])
    
    # positives
    tp = conf_mtrx[positive_index, positive_index]
    fn = conf_mtrx[positive_index, negative_index]
    
    # negatives
    tn = conf_mtrx[negative_index, negative_index]
    fp = conf_mtrx[negative_index, positive_index]
        
    metrics = {}
    
    # recall
    metrics['tp/p'] = tp / n_positive
    # 1 - recall
    metrics['fn/p'] = fn / n_positive
    
    # 1 - percision
    metrics['fp/g'] = fp / n_pred_positive
    # percision
    metrics['tp/g'] = tp / n_pred_positive
    
    # specificity
    metrics['tn/n'] = tn / n_negative
    # 1 - specificity
    metrics['fp/n'] = fp / n_negative
    
    # !!
    metrics['tn/b'] = tn / n_pred_negative
    metrics['fn/b'] = fn / n_pred_negative
    
    return metrics

# %%
metrics_sgd_8 = assess_performance(y_train_8, y_pred_8)


# %%
# calculate error metrics v.s. different threshholds 
from sklearn.metrics import roc_curve

y_score_8 = cross_val_predict(sgd_clf, X_train, y_train_8, 
                              cv=3, method="decision_function")

fpr, tpr, thresholds = roc_curve(y_true=y_train_8, y_score=y_score_8)


#%%

def performance_vs_thresholds(y_true,
                              y_score, 
                              metric1='tp/p', metric2='fp/n'):
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # {'tp/p':[0.93, 0. 84,...]}
    metrics_scores = {'thresholds':thresholds, metric1:[], metric2:[]}
    for threshold in thresholds:
        
        # positive class if y_score[i] > threshold, for each element in y
        y_pred = y_score > threshold
        
        # performance metrics at threshold
        metrics = assess_performance(y_true, y_pred)
        per1, per2 = metrics[metric1], metrics[metric2]
        
        metrics_scores[metric1].append(per1)
        metrics_scores[metric2].append(per2)
        
    return metrics_scores


# %%
# visualize error metrics and threshhold
def plot_performance_curve(metric1, metric2, label=None, line_only=False,
                           use_conventional_names=False):
    """generates a plot similar to the ROC curve based
    on model's performance at different thresholds

    Args:
        metric1 (tuple): 'metric_name', [...metric scores...]
        metric2 (tuple): _description_
        label (str)    : e.g., 'Random Forest'
    """
    
    metric1_name, metric1_score = metric1
    metric2_name, metric2_score = metric2
    
    # e.g., from 'tp/p' to 'Recall'
    if use_conventional_names:
        metric1_name = METRICS_NAMES[metric1_name]
        metric2_name = METRICS_NAMES[metric2_name]
    
    # set color and linestyle randomly to avoid using the same 
    # color and style repeatedly
    # linestyles = ['-', '--', '-.', ':']
    linestyles = ['-']
    
    color = sb.palettes.color_palette()[np.random.randint(0, 10)]
    linestyle = linestyles[np.random.randint(0, len(linestyles))]
    
    
    plt.plot(metric1_score, metric2_score, label=label,
             color=color, linestyle=linestyle)
    
    if not line_only:
        plt.xlabel(metric1_name)
        plt.ylabel(metric2_name)
        plt.title(f"{metric1_name} v.s. {metric2_name}")
        plt.grid()
        plt.plot([0, 1], [0, 1], 'k--')
    plt.legend(loc="lower right")
    plt.tight_layout()
   
    
#%%
# sgd performance

fpr, tpr, thresholds = roc_curve(y_true=y_train_8, y_score=y_score_8)

metrics_scores = performance_vs_thresholds(y_train_8, y_score_8)

# test plotting function with sklearn values
plot_performance_curve(('fpr', fpr), ('tpr', tpr), label="sklearn")

# test plotting function with assess_performance() values
#%%
metric1_test = ('tp/p', metrics_scores['tp/p'])
metric2_test = ('fp/n', metrics_scores['fp/n']) 
plot_performance_curve(metric2_test, metric1_test, label='my implementation')

# %%
