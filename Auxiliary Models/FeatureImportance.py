# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 02:19:09 2020

@author: e-sshen
"""

import pandas as pd
import numpy as np
import xgboost

dataset = pd.read_csv('C:/Users/e-sshen/Desktop/GIC Data/ActiveHolders.csv')
#x = dataset.iloc[:,2:8].values
#y = dataset.iloc[:,9].values

print(dataset.shape)
x = dataset.iloc[:,2:66].values
py = dataset.iloc[:,67].values
y = dataset.iloc[:,68].values
x[:] = np.nan_to_num(x)


# decision tree for feature importance on a regression problem
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot
# define dataset
x, y = make_regression(n_samples=25289, n_features=60, n_informative=30, random_state=1)
# define the model
model = DecisionTreeRegressor()
# fit the model
model.fit(x, y)
# get importance
importance = model.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()



# xgboost for feature importance on a classification problem
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=25289, n_features=60, n_informative=30, n_redundant=30, random_state=1)
# define the model
model = XGBClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
