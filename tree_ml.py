# -*- coding: utf-8 -*-
"""
Created on Tue Mar 05 18:33:53 2019

@author: webberl
"""
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

df=pd.read_csv('batter_stats.csv', sep=',',header=0)

targets = np.array(df['Salary'])

df = df.drop(['Rank', 'Player', 'Team', 'Salary'], axis=1)


df = df.drop('Position', axis=1)
df = df.drop('Season', axis=1)



labels = list(df.columns)
features = np.array(df)

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.75, random_state=42)

##'bootstrap': True, 'min_samples_leaf': 5, 'n_estimators': 630, 'min_samples_split': 8, 'max_features': 'auto', 'max_depth': 40}
rf = RandomForestRegressor(bootstrap=True, min_samples_leaf=5, n_estimators=650, max_features='auto', min_samples_split=8, max_depth=40)

rf.fit(train_features, train_targets)

predictions = rf.predict(test_features)

print test_targets[1]
print predictions[1]

print r2_score(test_targets, predictions, multioutput='raw_values')

print rf.feature_importances_