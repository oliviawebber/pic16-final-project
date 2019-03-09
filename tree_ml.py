# -*- coding: utf-8 -*-
"""
Created on Tue Mar 05 18:33:53 2019

@author: webberl
"""
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

df=pd.read_csv('batter_stats.csv', sep=',',header=0)

inflation = [0.0531,0.0813,0.0390,0.123,0.0201,0.0110]

years = [2012,2013,2014,2015,2016,2017,2018]

#removed_outliers = df['Salary'] < 25000000
removed_outliers = df['Salary'] < 10000000
df = df[removed_outliers]

for i in range(len(years)):
    year = years[i]
    mask = df['Season']==year
    for adjustment in inflation[i:]:
        df.loc[mask, 'Salary'] = df[mask]['Salary'] * (1+adjustment)

targets = np.array(df['Salary'])
#targets = np.log(targets)
#targets = targets / 100000

df = df.drop(['Rank', 'Player', 'Team', 'Salary'], axis=1)


df = df.drop('Position', axis=1)
df = df.drop('Season', axis=1)

labels = list(df.columns)
features = np.array(df)

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.30, random_state=42)

#{'bootstrap': True, 'min_samples_leaf': 4, 'n_estimators': 400, 'max_features': 'auto', 'min_samples_split': 10, 'max_depth': 40}
rf = RandomForestRegressor(bootstrap=True, min_samples_leaf=4, n_estimators=400, max_features='auto', min_samples_split=10, max_depth=40)
rf.fit(train_features, train_targets)

predictions = rf.predict(test_features)

print test_targets[5]
print predictions[5]

scores = cross_val_score(rf, test_features, test_targets, cv=5)
print scores
#print r2_score(test_targets, predictions)

print rf.feature_importances_