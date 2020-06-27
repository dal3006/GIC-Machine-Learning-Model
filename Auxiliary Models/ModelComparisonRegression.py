# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 04:49:20 2020

@author: e-sshen
"""

# Compare Algorithms
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# load dataset
dataset = pd.read_csv('C:/Users/e-sshen/Desktop/GIC Data/ActiveHolders.csv')
x = dataset.iloc[:,2:66].values
y = dataset.iloc[:,67].values
x[:] = np.nan_to_num(x)
sc = StandardScaler()
x = sc.fit_transform(x)

# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []

models.append(('LI', LinearRegression()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('RF', RandomForestRegressor(n_estimators = 100)))
models.append(('XBG', XGBRegressor()))
# evaluate each model in turn
results1 = []
results2 = []
results3 = []
names = []
for name, model in models:
    start_time = time.time()
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv1 = model_selection.cross_val_score(model, x, y, cv=kfold, scoring='neg_mean_absolute_error')
    results1.append(cv1)
    cv2 = model_selection.cross_val_score(model, x, y, cv=kfold, scoring = 'neg_mean_squared_error')
    results2.append(cv2)
    cv3 = model_selection.cross_val_score(model, x, y, cv= kfold, scoring = 'r2')
    results3.append(cv3)
    
    names.append(name)
    print(("%s: Mean Absolute Error %f (%f)") % (name, cv1.mean(), cv1.std()))
    print(("%s: Mean Squared Error %f (%f)") % (name, cv2.mean(), cv2.std()))
    print(("%s: R^2 %f (%f)") % (name, cv3.mean(), cv3.std()))
    print(("Time: %f seconds") % (round(time.time() - start_time, 2)))
    
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison - Regression (Mean Absolute Error)')
ax = fig.add_subplot(111)
plt.boxplot(results1)
ax.set_xticklabels(names)
plt.show()

fig = plt.figure()
fig.suptitle('Algorithm Comparison - Regression (Mean Squared Error)')
ax = fig.add_subplot(111)
plt.boxplot(results2)
ax.set_xticklabels(names)
plt.show()

fig = plt.figure()
fig.suptitle('Algorithm Comparison - Regression (R^2)')
ax = fig.add_subplot(111)
plt.boxplot(results3)
ax.set_xticklabels(names)
plt.show()