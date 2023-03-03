# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
# ---

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


# how to use
# metrics_score = performance_vs_thresholds(...)
# plot_performance_curve(metrics_score['fp/p'], metrics_score['tp/n'], label='Logestic Regression')
# plot_performance_curve(..., line_only=True)



# %%
METRICS_NAMES = {
    'tp/p': 'Recall',
    'fn/p': '1 - Recall',
    'tp/g': 'Percision',
    'fp/g': '1 - Percision',
    'tn/n': 'Specificity',
    'fp/n': '1 - Specificity',
    'tn/b': 'tn/b', # doesn't have a name
    'fn/b': 'fn/b',    
}


# %%
# load data

# mnist = fetch_openml('mnist_784', version=1)
# mnist.keys()
# mnist = pd.read_csv('./data/mnist.csv')

# # %%
# # split to train and test sets

# # X, y = mnist['data'], mnist['target']
# X, y = mnist.drop(columns=['target']), mnist.iloc[:, -1]
# X_train, X_test = X[:60000], X[60000:]
# y_train, y_test = y[:60000].astype('int'), y[60000:].astype('int')

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
# some_digit = X_train.iloc[0]
# plot_digits(some_digit)


# %%
# simplify classification problem

# two classes: true if 8, false otherwise
# y_train_8 = (y_train == 8)
# y_test_8 = (y_test == 8)

# %%
# train a classification model
from sklearn.linear_model import SGDClassifier
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

sgd_clf = SGDClassifier() # no parameters? what cost function is default?
# sgd_clf.fit(X_train, y_train_8)
# y_pred_8 = sgd_clf.predict(X_test)
# cost = 'f1'
# y_pred_8 = cross_val_predict(sgd_clf, X_train, y_train_8, cv=3)

# %%
# understand errors of the model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# conf_mtrx_8 = confusion_matrix(y_true=y_train_8, y_pred=y_pred_8)
# targets_sums = np.sum(conf_mtrx_8, axis=1)
# norm_conf_mtrx_8 = conf_mtrx_8 / targets_sums

# %%
# plt.figure(figsize=(4,4))
# ConfusionMatrixDisplay.from_predictions(y_true=y_train_8, y_pred=y_pred_8)

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
# metrics_sgd_8 = assess_performance(y_train_8, y_pred_8)


# %%
# calculate error metrics v.s. different threshholds 
from sklearn.metrics import roc_curve

# y_score_8 = cross_val_predict(sgd_clf, X_train, y_train_8, 
#                               cv=3, method="decision_function")

# fpr, tpr, thresholds = roc_curve(y_true=y_train_8, y_score=y_score_8)


# %%

def performance_vs_thresholds(y_true,
                              y_score, 
                              all_metrics=False):
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # {'tp/p':[0.93, 0. 84,...]}
    metrics_scores = {'thresholds':thresholds}
    
    # initialize metric_scores
    for metric in METRICS_NAMES.keys():
        metrics_scores[metric] = []
    
    for threshold in thresholds:
        
        # positive class if y_score[i] > threshold, for each element in y
        y_pred = y_score > threshold
        
        # performance metrics at threshold
        metrics = assess_performance(y_true, y_pred)
        
        for metric, score in metrics.items():
            metrics_scores[metric].append(score)
        
        # per1, per2 = metrics[metric1], metrics[metric2]
        
        # metrics_scores[metric1].append(per1)
        # metrics_scores[metric2].append(per2)
        
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








###############
#### new ######
###############


# %%
# compare simple knn model and optimized linear regression model
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score


def print_accuracy(y_true, y_pred, model_name='', balanced_accuracy=True):
    """Print out accuracy scores

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
        model_name (str, optional): to use when printing. Defaults to ''.
        balanced_accuracy (bool, optional): print balanced accuracy score too?. Defaults to True.
    """
    
    accuracy_score_lr_clf_8 = accuracy_score(y_true, y_pred)
    balanced_accuracy_score_lr_clf_8 =balanced_accuracy_score(y_true, y_pred)
    
    print(f"{model_name} accuracy: ", accuracy_score_lr_clf_8)
    print(f"{model_name} balanced accuracy: ", balanced_accuracy_score_lr_clf_8)
        

#%%
def print_cv_results(cv_results, keys=['mean_test_score', 'params']):
    """print out the performance of the different models tested in GridSearch

    Args:
        cv_results (dict): FittedGridSearch.cv_results_
        keys (list, optional): what information to print? Provide keys inside cv_results. 
                                Defaults to ['mean_test_score', 'params'].
    """   
    
    
    if keys[0] == 'mean_test_score':
        # Extract relevant data and sort by mean_test_score
        results = sorted(zip(*tuple([cv_results[key] for key in keys])), 
                        key=lambda x: x[0], reverse=True)
    else:
        results = zip(*tuple([cv_results[key] for key in keys]))
    
    
    # formatting for table
    lst = ['{:<20}' for i in range(len(keys))]
    formatting_str = ' '.join(lst)
     
    # Print header row
    print(formatting_str.format(*keys))

    for result in results:
        result_str = [f'{round(val, 4)}' if isinstance(val, float) else str(val) for val in result]
        print(formatting_str.format(*result_str))



#%%
def refit_model_predict(estimator, X, y, method, estimator_class=None):
    """_summary_

    Args:
        estimator (_type_): _description_
        X (_type_): _description_
        y (_type_): _description_
        method (str): ["predict", "decision_function", "predict_proba", "predict_log_proba"]
        estimator_class (_type_, optional): _description_. Defaults to None.

    Raises:
        BaseException: _description_

    Returns:
        _type_: _description_
    """
    
    # initialize estimator_fin based on estimator's type
    if isinstance(estimator, dict):
        if estimator_class is not None:
            # unpack keyword arguments inside estimator dictionary
            estimator_fin = estimator_class(**estimator)
        else:
            raise BaseException('parameters input incorrect')
    # estimator should be a sklearn estimator
    else:
        estimator_fin = clone(estimator)
    
    # refit and predict
    y_wanted = cross_val_predict(estimator_fin, X, y, method=method)
    
    return y_wanted
    

# %%

import joblib
import inspect

def save_model(model, path='./models'):
    """Save model at location path."""
    
    for name, value in inspect.currentframe().f_back.f_locals.items():
        if value is model:
            variable_name = name
            joblib.dump(value, f'{path}/{variable_name}.pkl')
            break
        
#%%
def explore_metrics(metrics_scores):
    """generate all pairs of ROC-like curves

    Args:
        metrics_scores (_type_): _description_

    Returns:
        list: list of plots
    """
    
    plots = []

    for name_i, value_i in metrics_scores.items():
        for name_j, value_j in metrics_scores.items():
            
            if not (name_i == 'thresholds' or name_j == 'thresholds'):
            
                metric1 = (name_i, value_i)
                metric2 = (name_j, value_j)
                
                plt.figure()
                plot_performance_curve(metric1, metric2, use_conventional_names=True)
                curr_plot = plt.gcf()

                plots.append(curr_plot)
    
    return plots



#%%
# compiles all relevant data into 1 dictionary to evaluate the performance of a model
# fitst draft produced by ChatGPT

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(X, y, model_b):
    
    # clone the model (to prevent data leakage)
    model = model_b
    
    # Train the model
    # model.fit(X, y)
    
    # Predict on the training data
    y_pred = refit_model_predict(model, X, y, 'predict')
    
    # 
    try:
        y_score = refit_model_predict(model, X, y, 'predict_proba')
    except:
        y_score = refit_model_predict(model, X, y, 'decision_function')
    
    # Calculate accuracy and balanced accuracy
    accuracy = accuracy_score(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    print("Accuracy:", accuracy)
    print("Balanced accuracy:", balanced_accuracy)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred)
    print("Confusion matrix:\n", cm)
    
    # Calculate classification report
    print("Classification report:\n", classification_report(y, y_pred))
    
    try:
        # Calculate ROC curve and AUC score !!!!!!!!! this needs to work using y_score
        fpr, tpr, thresholds = roc_curve(y, model.predict_proba(X)[:,1])
        roc_auc = roc_auc_score(y, model.predict_proba(X)[:,1])
            
        # Plot ROC curve
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()
    
        # Save ROC curve and AUC score in a list
        roc_data = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
    except:
        roc_data = {'results':'run failed!'}

    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'confusion_matrix': cm,
        'classification_report': classification_report(y, y_pred, output_dict=True),
        'roc_data': roc_data
    }
