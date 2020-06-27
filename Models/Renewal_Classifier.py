# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:36:51 2020

@author: e-sshen
"""

import pandas as pd
import numpy as np

#Load the Dataset
dataset = pd.read_csv('C:/Users/e-sshen/Desktop/GIC Data/ActiveHolders.csv')
x = dataset.iloc[:,2:66].values
ry = dataset.iloc[:,68].values
x[:] = np.nan_to_num(x)

print(x)
print(ry)

#Split Dataset into Trainng and Test Datasets
from sklearn.model_selection import train_test_split
x_train, x_test, ry_train, ry_test = train_test_split(x, ry, test_size = 0.2, random_state = 0, shuffle = True)

#Fit model to Training Set
from xgboost import XGBClassifier
renewal_classifier = XGBClassifier()
renewal_classifier.fit(x_train, ry_train)

#Predict the Test Results
from sklearn.metrics import accuracy_score

ry_pred = renewal_classifier.predict(x_test)
accuracy = accuracy_score(ry_test, ry_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits = 10, random_state = 7)

results = cross_val_score(renewal_classifier, x, ry, cv = kfold)
print("Accuracy: %.2f%% (%.2f%%) " % (results.mean()*100, results.std()*100))

#Save the Model to a Pickle File
import pickle
filename = 'C:/Users/e-sshen/Desktop/GIC Data/Renewal.pkl'
pickle.dump(renewal_classifier, open(filename, 'wb'))


            
