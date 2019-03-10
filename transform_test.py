# -*- coding: utf-8 -*-
"""
Created on Tue Mar 05 18:33:53 2019

@author: webberl
"""
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('batter_stats.csv', sep=',',header=0)

inflation = [0.0531,0.0813,0.0390,0.123,0.0201,0.0110]

years = [2012,2013,2014,2015,2016,2017,2018]

for i in range(len(years)):
    year = years[i]
    mask = df['Season']==year
    for adjustment in inflation[i:]:
        df.loc[mask, 'Salary'] = df[mask]['Salary'] * (1+adjustment)
        
mask = df['Salary'] > 1000000
df = df[mask]

df = df.drop(['Rank', 'Player', 'Team'], axis=1)
df = df.drop('Position', axis=1)
df = df.drop('Season', axis=1)

data = df.values

scaler = QuantileTransformer(n_quantiles=100)
data_scaled = scaler.fit_transform(data)
features = data_scaled[:,:14]
targets = data_scaled[:,14:]

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.30, random_state=42)

##'bootstrap': True, 'min_samples_leaf': 5, 'n_estimators': 630, 'min_samples_split': 8, 'max_features': 'auto', 'max_depth': 40}
rf = RandomForestRegressor(bootstrap=True, min_samples_leaf=5, n_estimators=650, max_features='auto', min_samples_split=8, max_depth=40)


rf.fit(train_features, train_targets)

predictions = rf.predict(test_features)
predictions = predictions.reshape(-1,1)

print r2_score(test_targets, predictions)
print rf.feature_importances_

print test_features.shape
print test_targets.shape
print predictions.shape

actual = np.concatenate((test_features, test_targets), axis=1)
predicted = np.concatenate((test_features, predictions), axis=1)

actual_inverted = scaler.inverse_transform(actual)
predicted_inverted = scaler.inverse_transform(predicted)

plt.plot(actual_inverted[:,14:],predicted_inverted[:,14:], 'o')