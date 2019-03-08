#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:22:48 2019

@author: EmmetDonahue
"""

import numpy as np
import pandas as pd
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
    
correct = 0.0
for i in range(len(predictions)): # assess accuracy of model
    if predictions[i] == test_targets[i]:
        correct += 1
        
print 'Accuracy:',correct/len(predictions)*100.0,'%'