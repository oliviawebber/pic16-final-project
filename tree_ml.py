# -*- coding: utf-8 -*-
"""
Created on Tue Mar 05 18:33:53 2019

@author: webberl
"""

from sklearn import tree
import pandas as pd

df=pd.read_csv('batter_stats.csv', sep=',',header=0)

data = []
target = []

for i in range(len(df)/2):
    row_data = df[i:i+1].values.tolist()[0][5:22]
    data.append(row_data[:len(row_data)-1])
    target.append(row_data[len(row_data)-1])
    
    
clf = tree.DecisionTreeRegressor()
clf = clf.fit(data, target)

print clf.predict([df[1599:1600].values[0][5:19]])