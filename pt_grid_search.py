# -*- coding: utf-8 -*-
"""
Created on Thu Mar 07 11:41:36 2019

@author: webberl
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

df=pd.read_csv('batter_stats.csv', sep=',',header=0)

targets = np.array(df['Salary'])
targets = targets / 100000

df = df.drop(['Rank', 'Player', 'Team', 'Salary'], axis=1)

df = df.drop('Position', axis=1)
df = df.drop('Season', axis=1)

labels = list(df.columns)
features = np.array(df)

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.30, random_state=42)


param_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 800, num = 100)],
               'max_features': ['auto'],
               'max_depth': [20,30,40,50,60],
               'min_samples_split': [8,9,10,11,12],
               'min_samples_leaf': [2,3,4,5,6],
               'bootstrap': [True]}


rf = RandomForestRegressor()
rf_random = GridSearchCV(rf, param_grid, cv=5, verbose=10, n_jobs=-1)
rf_random.fit(train_features, train_targets)

#{'bootstrap': True, 'min_samples_leaf': 4, 'n_estimators': 400, 'max_features': 'auto', 'min_samples_split': 10, 'max_depth': 40}
#'bootstrap': True, 'min_samples_leaf': 5, 'n_estimators': 630, 'min_samples_split': 8, 'max_features': 'auto', 'max_depth': 40}
print '\n', rf_random.best_params_
