# -*- coding: utf-8 -*-
"""
Created on Tue Mar 05 18:33:53 2019

@author: webberl
"""
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('test_bat.csv', sep=',',header=0)
#https://www3.nd.edu/~lawlib/baseball_salary_arbitration/minavgsalaries/Minimum-AverageSalaries.pdf

avg_salary = [371571,412520,412454,438729,497254,597537,851492,1028667,1076089,
              1168263,1110766,1119981,1336609,1398831,1611166,1895630,2138896,
              2295649,2372189,2313535,2476589,2699292,2824751,2925679,2996106,
              3014572,3095183,3210000,3390000,3690000,3840000,4380000,4470000,
              4520000]


inflation = (len(avg_salary)-1) * [0]
for i in range(len(avg_salary)-1):
    inflation[i] = float(avg_salary[i+1])/avg_salary[i]

years = range(1985,2019,1)

removed_outliers = (df.salary < 16269954) & (df.salary > 1569444)
#removed_outliers = df['Salary'] > 500000
df = df[removed_outliers]

for i in range(len(years)):
    year = years[i]
    mask = df['yearID']==year
    for adjustment in inflation[i:]:
        df.loc[mask, 'salary'] = df[mask]['salary'] * (adjustment)

#targets = np.array(df['Salary'])
#targets = np.sqrt(targets)
#targets = (targets-np.mean(targets))/np.std(targets)
#targets = targets / 100000

mask = (df.G < 3) & \
       (df.AB < 3) & \
       (df.R < 3) & \
       (df.H < 3) & \
       (df.B2 < 3) & \
       (df.B3 < 3) & \
       (df.HR < 3) & \
       (df.RBI < 3) & \
       (df.SB < 3) & \
       (df.CS < 3) & \
       (df.BB < 3) & \
       (df.SO < 3) & \
       (df.IBB < 3) & \
       (df.HBP < 3) & \
       (df.SH < 3) & \
       (df.SF < 3) & \
       (df.GIDP < 3) 
       
df = df.drop(df[mask].index)

df = df.drop(['num', 'playerID', 'yearID', 'stint', 'teamID', 'IgID'], axis=1)

labels = list(df.columns)
targets = df['salary']

targets = np.log(np.log(targets))
df = df.drop('salary', axis=1)

features = np.array(df)

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.20, random_state=42)

##'bootstrap': True, 'min_samples_leaf': 5, 'n_estimators': 630, 'min_samples_split': 8, 'max_features': 'auto', 'max_depth': 40}
rf = RandomForestRegressor(bootstrap=True, min_samples_leaf=5, n_estimators=650, max_features='auto', min_samples_split=8, max_depth=40)


rf.fit(train_features, train_targets)

predictions = rf.predict(test_features)

#test_targets = np.exp(test_targets)
#predictions = np.exp(predictions)

print np.sqrt(mean_squared_error(test_targets, predictions))/np.mean(predictions)
print rf.feature_importances_

plt.plot(predictions, test_targets, 'o')
plt.xlabel('predictions')
plt.ylabel('targets')
plt.show()