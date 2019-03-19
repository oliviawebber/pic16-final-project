import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors 
from sklearn.model_selection import train_test_split

df = pd.read_csv('clean_pitcher.csv', sep=',', header=0)

df = df.drop(['num', 'playerID', 'yearID', 'stint', 'teamID', 'IgID', 'prev_salary'], axis=1)

salaries = np.array(df['salary'])
sal_range =  np.amax(salaries) - np.amin(salaries)
sal_interval = sal_range/500
targets = salaries/sal_interval # target salaries
targets = targets.astype(int)

features = np.array(df) # player stats

training_features, test_features, training_targets, test_targets = train_test_split(features, targets, test_size = .4)

nbrs = neighbors.KNeighborsClassifier(1)
nbrs.fit(training_features, training_targets)

predictions = []
for elem in test_features: # test model
    elem = np.array([elem])
    predictions.append(nbrs.predict(elem))
    
pred = [] # non numpy array form of predictions
for elem in predictions:
    pred.append(elem[0])

so = test_features[:,7]
wins = test_features[:,8]

plt.xlabel('Strikeouts')
plt.ylabel('Wins')
plt.title('KNN Model Salary Predictions with Hot Color Scheme')
plt.scatter(so, wins, c=pred, cmap = 'hot')
plt.show()

correct = 0.0
for i in range(len(predictions)): # assess accuracy of model
    if predictions[i] == test_targets[i]:
        correct += 1
        
print 'Accuracy:',correct/len(predictions)*100.0,'%'