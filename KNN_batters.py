#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:22:48 2019

@author: EmmetDonahue
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors 
from sklearn.model_selection import train_test_split

df = pd.read_csv('batter_stats.csv', sep=',', header=0)
salaries = np.array(df['Salary'])

sal_range =  np.amax(salaries) - np.amin(salaries)
sal_interval = sal_range/500
targets = salaries/sal_interval # target salaries
targets = targets.astype(int)

df = df.drop(['Rank', 'Player', 'Position', 'Team', 'Season'], axis=1)

features = np.array(df) # player stats

training_features, test_features, training_targets, test_targets = train_test_split(features, targets, test_size = .4)

nbrs = neighbors.KNeighborsClassifier(1)
nbrs.fit(training_features, training_targets)

predictions = []
for elem in test_features: # test model
    elem = np.array([elem])
    predictions.append(nbrs.predict(elem))
    
pred = [] # non numpy array form of predictions
for elem in predictions:
    pred.append(elem[0])

hr = test_features[:,6] # read test stats into arrays for visualization
rbi = test_features[:,7]
so = test_features[:,9]
runs = test_features[:,3]
hits = test_features[:,2]
slg = test_features[:,12]
avg = test_features[:,10]

plt.xlabel('Home Runs')
plt.ylabel('RBI')
plt.title('KNN Model Salary Predictions with afmhot Color Scheme')
plt.scatter(hr, rbi, c=pred, cmap = 'afmhot')
plt.show()

plt.xlabel('Runs')
plt.ylabel('Hits')
plt.title('KNN Model Salary Predictions with afmhot Color Scheme')
plt.scatter(runs, hits, c=pred, cmap = 'afmhot')
plt.show()

correct = 0.0
for i in range(len(predictions)): # assess accuracy of model
    if predictions[i] == test_targets[i]:
        correct += 1
        
print 'Accuracy:',correct/len(predictions)*100.0,'%'
