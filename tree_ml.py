# -*- coding: utf-8 -*-
"""
Created on Tue Mar 05 18:33:53 2019

@author: webberl
"""
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

df=pd.read_csv('batter_stats.csv', sep=',',header=0)


targets = np.array(df['Salary'])
targets = targets / 100000

df = df.drop(['Rank', 'Player', 'Team', 'Salary'], axis=1)

df = df.drop('Position', axis=1)
df = df.drop('Season', axis=1)



labels = list(df.columns)
features = np.array(df)

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.99, random_state=42)

rf = RandomForestRegressor()

rf.fit(train_features, train_targets)

predictions = rf.predict(test_features)

print r2_score(test_targets, predictions, multioutput='raw_values')