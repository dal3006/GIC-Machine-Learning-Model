# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 03:23:50 2020

@author: e-sshen
"""
# Compare Algorithms
import pandas
import time
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.svm import SVC
# load dataset
dataset = pd.read_csv('C:/Users/e-sshen/Desktop/GIC Data/ActiveHolders.csv')
x = dataset.iloc[:,1:65].values
y = dataset.iloc[:,67].values
x[:] = np.nan_to_num(x)

# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('LR', LogisticRegression(solver = 'lbfgs')))
#models.append(('LI', LinearRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
#models.append(('CART- Regressor', DecisionTreeRegressor()))
models.append(('RF', RandomForestClassifier(n_estimators = 100)))
#models.append(('RF - Regressor', RandomForestRegressor()))
models.append(('XGB', XGBClassifier()))
#models.append(('XBG - Regressor', XGBRegressor()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))
# evaluate each model in turn
results1 = []
results2 = []
results3 = []
names = []
for name, model in models:
    start_time = time.time()
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv1 = model_selection.cross_val_score(model, x, y, cv=kfold, scoring='accuracy')
    results1.append(cv1)
    cv2 = model_selection.cross_val_score(model, x, y, cv=kfold, scoring = 'neg_log_loss')
    results2.append(cv2)
    cv3 = model_selection.cross_val_score(model, x, y, cv= kfold, scoring = 'roc_auc')
    results3.append(cv3)
    names.append(name)
    msg = "%s: Accuracy [%f (%f)] Log Loss [%f (%f)] Area Under ROC Curve [%f (%f)]  Time: %f" % (name, cv1.mean(), cv1.std(), cv2.mean(), cv2.std(), cv3.mean(), cv3.std(), round(time.time() - start_time, 2))
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison - Classifiers (Accuracy)')
ax = fig.add_subplot(111)
plt.boxplot(results1)
ax.set_xticklabels(names)
plt.show()

fig = plt.figure()
fig.suptitle('Algorithm Comparison - Classifiers (Log Loss)')
ax = fig.add_subplot(111)
plt.boxplot(results2)
ax.set_xticklabels(names)
plt.show()

fig = plt.figure()
fig.suptitle('Algorithm Comparison - Classifiers (Area Under ROC Curve)')
ax = fig.add_subplot(111)
plt.boxplot(results3)
ax.set_xticklabels(names)
plt.show()