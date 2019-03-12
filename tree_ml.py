# -*- coding: utf-8 -*-
"""
Created on Tue Mar 05 18:33:53 2019

@author: webberl
"""
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""
Begin Processing Code
"""
df=pd.read_csv('batter_expanded.csv', sep=',',header=0)


#https://www3.nd.edu/~lawlib/baseball_salary_arbitration/minavgsalaries/Minimum-AverageSalaries.pdf
avg_salary = [371571,412520,412454,438729,497254,597537,851492,1028667,1076089,
              1168263,1110766,1119981,1336609,1398831,1611166,1895630,2138896,
              2295649,2372189,2313535,2476589,2699292,2824751,2925679,2996106,
              3014572,3095183,3210000,3390000,3690000,3840000,4380000,4470000,
              4520000]

inflation = (len(avg_salary)-1) * [0]
for i in range(len(avg_salary)-1):
    inflation[i] = float(avg_salary[i+1])/avg_salary[i]

years = range(1985,2017,1)


for i in range(len(years)):
    year = years[i]
    mask = df['yearID']==year
    for adjustment in inflation[i:]:
        df.loc[mask, 'salary'] = df[mask]['salary'] * (adjustment)

removed_outliers = (df.salary < np.percentile(df.salary, 75)*1.5) & (df.salary > np.percentile(df.salary, 25)/1.5)
df = df[removed_outliers]

cutoff = 5
 
mask = (df.AB < cutoff) & \
       (df.R < cutoff) & \
       (df.H < cutoff) & \
       (df.B2 < cutoff) & \
       (df.B3 < cutoff) & \
       (df.HR < cutoff) & \
       (df.RBI < cutoff) & \
       (df.SB < cutoff) & \
       (df.CS < cutoff) & \
       (df.BB < cutoff) & \
       (df.SO < cutoff) & \
       (df.IBB < cutoff) & \
       (df.HBP < cutoff) & \
       (df.SH < cutoff) & \
       (df.SF < cutoff) & \
       (df.GIDP < cutoff) 
df = df.drop(df[mask].index)
"""
End Processing Code
"""



df = df.drop(['num', 'playerID', 'yearID', 'stint', 'teamID', 'IgID'], axis=1)

labels = list(df.columns)
targets = np.array(df['salary'])

#targets = np.log(targets)
df = df.drop('salary', axis=1)
print df
features = np.array(df)

biner = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
biner.fit(targets.reshape(-1,1))
targets_transform = biner.transform(targets.reshape(-1,1))
targets_transform = targets_transform.astype(int)
targets_transform = np.ravel(targets_transform)
X_train, X_test, y_train, y_test = train_test_split(features, targets_transform, test_size=0.2)
rfc = RandomForestClassifier(n_estimators = 50)

print X_train
print y_train
print X_train.shape
print y_train.shape
rfc.fit(X_train, y_train)


predictions = rfc.predict(X_test)
plt.plot(predictions, y_test, 'o')
plt.xlabel('predictions')
plt.ylabel('targets')
plt.show()

""" Regression"""
#train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.20, random_state=42)
#
###'bootstrap': True, 'min_samples_leaf': 5, 'n_estimators': 630, 'min_samples_split': 8, 'max_features': 'auto', 'max_depth': 40}
#rf = RandomForestRegressor(bootstrap=True, min_samples_leaf=5, n_estimators=650, max_features='auto', min_samples_split=8, max_depth=40)
#
#
#rf.fit(train_features, train_targets)
#
#predictions = rf.predict(test_features)
#
##test_targets = np.exp(test_targets)
##predictions = np.exp(predictions)
#
#print np.sqrt(mean_squared_error(test_targets, predictions))/np.mean(predictions)
#print rf.feature_importances_
#
#plt.plot(predictions, test_targets, 'o')
#plt.xlabel('predictions')
#plt.ylabel('targets')
#plt.show()

