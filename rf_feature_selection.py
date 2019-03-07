# -*- coding: utf-8 -*-
"""
Created on Thu Mar 07 12:34:12 2019

@author: webberl
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

df=pd.read_csv('batter_stats.csv', sep=',',header=0)

targets = np.array(df['Salary'])

df = df.drop(['Rank', 'Player', 'Team', 'Salary'], axis=1)


df = df.drop('Position', axis=1)
df = df.drop('Season', axis=1)



labels = list(df.columns)
features = np.array(df)

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.3, random_state=42)

sel = SelectFromModel(RandomForestRegressor(n_estimators = 100))
sel.fit(train_features, train_targets)

selected_feat= df.columns[(sel.get_support())]
print selected_feat