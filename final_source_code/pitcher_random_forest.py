# -*- coding: utf-8 -*-
"""
@author: webberl
"""
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in the clean pitcher data and drop categorical columns that are not useful for
# predicting salary
df=pd.read_csv('clean_pitcher.csv', sep=',',header=0)
df = df.drop(['num', 'playerID', 'yearID', 'stint', 'teamID', 'IgID'], axis=1)

# Capture the headers for plotting later
labels = list(df.columns)

# Create the array of target variables
targets = np.array(df['salary'])

# Drop the target variable and turn everything remaining into a feature matrix
df = df.drop('salary', axis=1)
features = np.array(df)

# Split the data into training and testing sets
train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.20, random_state=42)

# Initialize the random forest regressor, train it on the data, and make predictions
rf = RandomForestRegressor(bootstrap=True, min_samples_leaf=5, n_estimators=650, max_features='auto', min_samples_split=8, max_depth=40)
rf.fit(train_features, train_targets)
predictions = rf.predict(test_features)

# Metrics to understand variance between predictions vs. actual and importance
# of different variables
print np.sqrt(mean_squared_error(test_targets, predictions))
print rf.feature_importances_

# Create plots to summarize findings
plt.plot(predictions, test_targets, 'o')
plt.xlabel('predictions')
plt.ylabel('targets')
plt.title('Pitcher Salary Predictions vs. Actual Salaries')
plt.show()

plt.bar(np.linspace(0,25,num=26), rf.feature_importances_, tick_label=labels[:26])
plt.setp(plt.gca().get_xticklabels(), rotation=90)
plt.ylabel('importance')
plt.xlabel('stat')
plt.title('Pitcher Feature Importance w/ prev_salary')
plt.show()

plt.bar(np.linspace(0,25,num=25), rf.feature_importances_[:-1], tick_label=labels[:25])
plt.setp(plt.gca().get_xticklabels(), rotation=90)
plt.ylabel('importance')
plt.xlabel('stat')
plt.title('Pitcher Feature Importance w/o prev_salary')
plt.show()
