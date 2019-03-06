# -*- coding: utf-8 -*-
"""
Created on Tue Mar 05 18:33:53 2019

@author: webberl
"""

from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

df=pd.read_csv('batter_stats.csv', sep=',',header=0)

data = []
target = []

for i in range(len(df)/2):
    row_data = df[i:i+1].values.tolist()[0][5:20]
    data.append(row_data[:len(row_data)-1])
    target.append(row_data[len(row_data)-1])
    
    
clf = tree.DecisionTreeRegressor()
clf = clf.fit(data, target)
# Had to fine tune parameters
regr = RandomForestRegressor(n_estimators = 500, max_depth=4)
regr = regr.fit(data, target)

player = 1400

print clf.predict([df[player:player+1].values.tolist()[0][5:19]])[0]
print regr.predict([df[player:player+1].values.tolist()[0][5:19]])[0]
print df[player:player+1].values.tolist()[0][19]