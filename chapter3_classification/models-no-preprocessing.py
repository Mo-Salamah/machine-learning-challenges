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
import performance_eval 

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

# binary classification 
y_train_8 = y_train == 8

#%%
# logestic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

lr_clf = LogisticRegression()

cv = 5

hyper_params = {
                'penalty':['none', 'l1', 'l2', 'elasticnet'],
                # 'regularization':[],
                'fit_intercept':[True, False],
                'class_weight':['balanced', {0:1, 1:3}, {0:3, 1:1}], 
                # 'l1_ratio':[]
                }

rdm_grid_search_lr_clf = RandomizedSearchCV(lr_clf, hyper_params, cv=cv, return_train_score=True, )




# %%
import joblib
# run grid search
# rdm_grid_search_lr_clf.fit(X_train, y_train_8)
# rdm_grid_serach_lr_clf = LogisticRegression().fit()
# performance_eval.save_model(rdm_grid_search_lr_clf)

grid_search_rf_result = joblib.load(MODELS_PATH + 'rdm_grid_search_lr_clf' + '.pkl')

# %%
# models failed to converge!
# compare different models

lr_clf_results = rdm_grid_search_lr_clf.cv_results_


performances_lr_clf = zip(
    lr_clf_results['params'], 
    lr_clf_results['mean_test_score'], 
    lr_clf_results['std_test_score'], 
    lr_clf_results['rank_test_score'])

# print models performance scores
for model_performance in performances_lr_clf:

    params = model_performance[0]
    mean_test_score = model_performance[1]
    
    print(mean_test_score, '   ', params)
    


#%%
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone

# copy the hyper parameters of the best estimator
lr_clf_best = clone(rdm_grid_search_lr_clf.best_estimator_)

# get the z_hat scores
y_score_8_lr_clf = cross_val_predict(lr_clf_best, X_train, y_train_8, method="decision_function")
# now you can visualize performance using the other file
y_predicted_8_lr_clf = cross_val_predict(lr_clf_best, X_train, y_train_8)
#%%
# visualize the performance of the second best model
# one with 'class_weight': {0: 1, 1: 3}
y_score_8_lr_clf_whts = cross_val_predict(LogisticRegression(penalty= 'l2', fit_intercept= True, class_weight= {0: 1, 1: 3}),
                                     X_train, y_train_8, method="decision_function")


#%%
# compare two models
metrics_scores = performance_eval.performance_vs_thresholds(y_true=y_train_8, y_score=y_score_8_lr_clf)
metrics_scores_whts = performance_eval.performance_vs_thresholds(y_true=y_train_8, y_score=y_score_8_lr_clf)


performance_eval.plot_performance_curve(('fp/n', metrics_scores['fp/n']),
                                        ('tp/p', metrics_scores['tp/p']), label='Logestic Regression')

performance_eval.plot_performance_curve(('fp/n', metrics_scores['fp/n']),
                                        ('tp/p', metrics_scores['tp/p']), label='Logestic Regression Weighted',
                                        line_only=True)

# ROC curves are exactly the same!

#%%
# compare AUC scores
from sklearn.metrics import roc_auc_score
auc_score_lr_clf = roc_auc_score(y_train_8, y_score_8_lr_clf)
auc_score_lr_clf_whts = roc_auc_score(y_train_8, y_score_8_lr_clf_whts)

# auc scores for the second best model (according to grid search) were higher than best model!

################################
############ knn  ##############
################################


#%%
# knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

knn_clf = KNeighborsClassifier()

cv = 3

params_knn = {
    'n_neighbors':np.arange(1, 5, 1),
}

rdm_hyper_params_knn = {
                'n_neighbors':[1, 3, 5], # would take way too long
                'weights':['uniform', 'distance'],
                # 'algorithm':['ball_tree', 'kd_tree' , 'brute', 'auto'],
                'algorithm':['auto'],
                
                }

rdm_grid_search_knn_clf = RandomizedSearchCV(knn_clf, rdm_hyper_params_knn, cv=cv)
grid_search_knn_clf = GridSearchCV(knn_clf, params_knn, cv=cv)


# %%
# run grid search
rdm_grid_search_knn_clf.fit(X_train, y_train_8)
# rdm_grid_search_knn_clf = joblib.load("./models/rdm_grid_serach_knn_clf.pkl")

#%%
grid_search_knn_clf.fit(X_train, y_train_8)
# grid_search_knn_clf = joblib.load("./models/grid_search_knn_clf.pkl")

#%%
# save trained models for future use
import joblib

joblib.dump(grid_search_knn_clf, './models/grid_search_knn_clf.pkl')
# my_model_loaded = joblib.load("/models/rdm_grid_serach_knn_clf.pkl")


#%%
print("Analyze simple knn model (n_neighbors tuned only)")
#%%
# visualie performance for different k values

scores_knn_clf = grid_search_knn_clf.cv_results_

performances_knn_clf = zip(
    # rdm_knn_clf_results['params'], 
    scores_knn_clf['mean_test_score'], 
    scores_knn_clf['std_test_score'], 
    scores_knn_clf['mean_score_time'],
    # scores_knn_clf['mean_train_score'],
    scores_knn_clf['rank_test_score'])

test_scr_knn_clf = scores_knn_clf['mean_test_score']
n_neighbors = scores_knn_clf['param_n_neighbors'].data

#%%
# best result is at k = 1, with small accuracy drop when k increases 

plt.plot(n_neighbors, test_scr_knn_clf, label='KNN Performance')
plt.title('KNN Model Performance at Different K Values')
plt.xlabel('Neighbors')
plt.ylabel('Metric') # what is it?


#%%
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone
# analyze simple model's performance

knn_clf_copy = clone(grid_search_knn_clf.best_estimator_)
y_scores_knn = cross_val_predict(knn_clf_copy, X_train, y_train_8, method='predict_proba')
y_predicted_knn = cross_val_predict(knn_clf_copy, X_train, y_train_8, method='predict')



#%%
# compare simple knn model and optimized linear regression model
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from performance_eval import print_accuracy

print_accuracy(y_train_8, y_predicted_8_lr_clf, 'Linear Regression', balanced_accuracy=True)
print_accuracy(y_train_8, y_predicted_knn, 'KNN', balanced_accuracy=True)

# knn is miles better than linear regression!

#%% 
# create visualizations of knn performance

print("knn doesn't have a decision function, so we can't draw ROC curve")

#%%
# performance_eval module needs to be reloaded
import importlib
importlib.reload(performance_eval)


# %%
# assess model generated through random grid search
import performance_eval
from performance_eval import print_cv_results

rdm_knn_clf_results = rdm_grid_search_knn_clf.cv_results_


header_knn_clf = ['params','mean_test_score','std_test_score',
                  'mean_score_time','mean_train_score','rank_test_score']

# use default keys
print_cv_results(rdm_knn_clf_results)


#%%
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone
from performance_eval import refit_model_predict

best_estimator = rdm_grid_search_knn_clf.best_estimator_

# get output of decision function of the best model
y_score_8_knn_clf = refit_model_predict(best_estimator, X_train, y_train_8, 'decision_function')

# get predicted values from another interesting model

# params_lr_clf_2_best = lr_grid_serach.best_params_
params_lr_clf_2_best = {'penalty': 'l2', 'fit_intercept':True, 'class_weight':{0:1, 1:3}}

# y_predicted_8_lr_clf_2 = refit_model_predict(params_lr_clf_2_best, X_train, y_train_8, method='predict', LogisticRegression)



#%%
# compare two models
metrics_scores = performance_eval.performance_vs_thresholds(y_true=y_train_8, y_score=y_score_8_lr_clf)
metrics_scores_whts = performance_eval.performance_vs_thresholds(y_true=y_train_8, y_score=y_score_8_lr_clf)


performance_eval.plot_performance_curve(('fp/n', metrics_scores['fp/n']),
                                        ('tp/p', metrics_scores['tp/p']), label='Logestic Regression')

performance_eval.plot_performance_curve(('fp/n', metrics_scores['fp/n']),
                                        ('tp/p', metrics_scores['tp/p']), label='Logestic Regression Weighted',
                                        line_only=True)

# ROC curves are exactly the same!

#%%
# compare AUC scores
from sklearn.metrics import roc_auc_score
auc_score_lr_clf = roc_auc_score(y_train_8, y_score_8_lr_clf)
auc_score_lr_clf_whts = roc_auc_score(y_train_8, y_score_8_lr_clf_whts)

# auc scores for the second best model (according to grid search) were higher than best model!






################################
######## Random Forest #########
################################ 


#%%
# find best model
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()

params_rdm_frst = {'n_estimators':[25, 80]}

cv = 3

scoring = None

grid_search_rf_clf = GridSearchCV(rf_clf, params_rdm_frst, scoring=scoring)



# %%
# fit GridSearch
import joblib

grid_search_rf_result = grid_search_rf_clf.fit(X_train, y_train_8)
# grid_search_rf_result = joblib.load(MODELS_PATH + 'grid_search_rf_result')

# %%
# print cv results
import performance_eval
from performance_eval import print_cv_results

cv_results = grid_search_rf_result.cv_results_

print_cv_results(cv_results) 


#%% 
# refit the best model to get "clean predictions"

from performance_eval import refit_model_predict

y_pred_8_rf = refit_model_predict(grid_search_rf_clf.best_estimator_, X_train, y_train_8, method='predict') 
y_score_8_rf = refit_model_predict(grid_search_rf_clf.best_estimator_, X_train, y_train_8, method='proba') 

#%%
# print accuracy
from performance_eval import print_accuracy
from sklearn.metrics import confusion_matrix


print_accuracy(y_train_8, y_pred_8_rf)

print(confusion_matrix(y_train_8, y_pred_8_rf))



# %%
# plot ROC curve for rf
from performance_eval import performance_vs_thresholds
from performance_eval import plot_performance_curve

metrics_scores_8_rf = performance_vs_thresholds(y_train_8, y_score_8_rf[:, 0])



#########################
######### SVM ###########
#########################


#%%
# SVM setup
from sklearn.svm import SVC

svm_clf = SVC()

params_svm_clf = {'C': [0.5, 1, 2],
                  'kernel':['rbf', 'poly']}

cv = 4

scoring = 'f1'

grid_search_svm_clf = GridSearchCV(svm_clf, params_svm_clf, scoring=scoring,
                                   refit=False, cv=cv)


#%%
# run gridsearchcv

import importlib
importlib.reload(performance_eval)
from performance_eval import save_model
# grid_search_svm_clf.fit(X_train, y_train_8)
save_model(grid_search_svm_clf)


#%%



#%%

###########################
##### Compare Models ######
###########################



# tuned models
# refit the models
# calculate y_pred & y_score ! you could implement new cross_val algorithm

# output:
# compare accuracy & balanced accuracy
# compare error types
# compare confusion matrices (for the basic model that produces y_pred & y_score)
# compare metrics_scores (for all thresholds)
# save ROC curves and AUC scores
# compare ...

#%%
# compare models
# importlib.reload(performance_eval)
import joblib
from performance_eval import evaluate_model

grid_search_rf_clf = joblib.load('./models/grid_search_rf_clf.pkl')
grid_search_lr_clf = joblib.load(MODELS_PATH + 'rdm_grid_search_lr_clf' + '.pkl')


rf = evaluate_model(X_train, y_train_8, grid_search_rf_clf.best_estimator_)
linearRegression = evaluate_model(X_train, y_train_8, grid_search_lr_clf.best_estimator_)

%%
# grid_search_rf_result = joblib.load(MODELS_PATH + 'grid_search_rf_result' + '.pkl')







