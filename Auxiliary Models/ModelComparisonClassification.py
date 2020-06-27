# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 03:23:50 2020

@author: e-sshen
"""
# Compare Algorithms
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

# load dataset
dataset = pd.read_csv('C:/Users/e-sshen/Desktop/GIC Data/ActiveHolders.csv')
x = dataset.iloc[:,2:66].values
y = dataset.iloc[:,68].values
x[:] = np.nan_to_num(x)

# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('LR', LogisticRegression(solver = 'lbfgs')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier(n_estimators = 100)))
models.append(('XGB', XGBClassifier()))
models.append(('NB', GaussianNB()))

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
    
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.2, random_state = seed)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    matrix = confusion_matrix(y_test, predicted)
    
    names.append(name)
    print("%s: Accuracy %f (%f)" % (name, cv1.mean(), cv1.std()))
    print("%s: Log Loss %f (%f)" %(name, cv2.mean(), cv2.std()))
    print("%s: Area Under ROC Curve %f (%f)" %(name, cv3.mean(), cv3.std()))
    print(matrix)
    print("Time: %f seconds" % (round(time.time() - start_time, 2)))
    
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