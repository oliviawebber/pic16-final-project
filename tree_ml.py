# -*- coding: utf-8 -*-
"""
Created on Tue Mar 05 18:33:53 2019

@author: webberl
"""
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
#from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

df=pd.read_csv('batter_stats.csv', sep=',',header=0)

#data = []
#target = []
#
#for i in range(len(df)/2):
#    row_data = df[i:i+1].values.tolist()[0][5:20]
#    data.append(row_data[:len(row_data)-1])
#    target.append(row_data[len(row_data)-1]/100000.0)
#    
#    
#clf = tree.DecisionTreeRegressor()
#clf = clf.fit(data, target)
## Had to fine tune parameters
#regr = RandomForestRegressor(n_estimators = 100, max_depth=2)
#regr = regr.fit(data, target)
#
#player = 1000
#
#print clf.predict([df[player:player+1].values.tolist()[0][5:19]])[0]
#print regr.predict([df[player:player+1].values.tolist()[0][5:19]])[0]
#print df[player:player+1].values.tolist()[0][19]/100000.0

targets = np.array(df['Salary'])
targets = targets / 100000

df = df.drop(['Rank', 'Player', 'Team', 'Salary'], axis=1)

df = df.drop('Position', axis=1)
df = df.drop('Season', axis=1)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


labels = list(df.columns)
features = np.array(df)

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.5, random_state=42)

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(rf, random_grid, n_iter=100, cv=3, verbose=10)
rf_random.fit(train_features, train_targets)

print '\n', rf_random.best_params_

#'bootstrap': True, 'min_samples_leaf': 4, 'n_estimators': 400, 'max_features': 'sqrt', 'min_samples_split': 2, 'max_depth': 70
#

#rf.fit(train_features, train_targets)
#
#predictions = rf.predict(test_features)
#
#print r2_score(test_targets, predictions, multioutput='raw_values')