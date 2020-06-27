# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 02:22:55 2020

@author: e-sshen
"""

import pandas as pd
import numpy as np

#Load the Dataset
dataset = pd.read_csv('C:/Users/e-sshen/Desktop/GIC Data/ActiveHolders.csv')
x = dataset.iloc[:,2:66].values
py = dataset.iloc[:,67].values
x[:] = np.nan_to_num(x)

#Split Dataset into Trainng and Test Datasets
from sklearn.model_selection import train_test_split
x_train, x_test, py_train, py_test = train_test_split(x, py, test_size = 0.2, random_state = 0, shuffle = True)

#Fit model to Training Set
from sklearn.tree import DecisionTreeRegressor

purchase_classifier = DecisionTreeRegressor()
purchase_classifier.fit(x_train, py_train)

#Predict the Test Results
from sklearn.metrics import accuracy_score

py_pred = purchase_classifier.predict(x_test)
accuracy = accuracy_score(py_test, py_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits = 10, random_state = 7)

results = cross_val_score(purchase_classifier, x, py, cv = kfold)
print("Accuracy: %.2f%% (%.2f%%) " % (results.mean()*100, results.std()*100))


#Save the Model to a Pickle File
##import pickle
#filename = 'C:/Users/e-sshen/Desktop/GIC Data/Purchase.pkl'
#pickle.dump(purchase_classifier, open(filename, 'wb'))