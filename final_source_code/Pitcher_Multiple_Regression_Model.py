import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor   
from patsy import dmatrices
import statsmodels.api as sm
from scipy.stats import linregress


pitchersDF = pd.read_csv('clean_pitcher.csv') #Read in data for Pitcher

predictor_list = ['H', 'R','ER','BB','SO','W', 'L', 'SV','HR','ERA','prev_salary'] #List that stores variables that will be used in predicting salaries


#The following for loop shows the significance value for each variable (look at each p-value)
for x in predictor_list:
	print 'Variable: {}'.format(x),linregress(pitchersDF['salary'], pitchersDF['{}'.format(x)]) 


#The following for loop plots each predictor against the salary variable to get a sense of what we can expect in our coefficents later on
for x in predictor_list:
	pitchersDF.plot(kind='scatter',x='{}'.format(x),y='salary',color='red')
	plt.show()

#We want to adjust the salarys because we want our predictor variables and response variable to be in the same range. 
pitchersDF['prev_salary'] = pitchersDF['prev_salary']/100000 
pitchersDF['salary'] = pitchersDF['salary']/100000

pitchersDF = pitchersDF[['H', 'R','ER','BB','SO','W', 'L', 'SV','HR','ERA','prev_salary','salary']] #Subset the dataframe to only include the predictors and response variable 

#The function below was found from this URL: https://planspace.org/20150423-forward_selection_with_statsmodels/
def forward_selected(data, response):
	remaining = set(data.columns)
	remaining.remove(response)
	selected = []
	current_score, best_new_score = 0.0, 0.0
	while remaining and current_score == best_new_score:
		scores_with_candidates = []
		for candidate in remaining:
			formula = "{} ~ {}".format(response,
										   ' + '.join(selected + [candidate]))
			score = smf.ols(formula, data).fit().rsquared_adj
			scores_with_candidates.append((score, candidate))
		scores_with_candidates.sort()
		best_new_score, best_candidate = scores_with_candidates.pop()
		if current_score < best_new_score:
			remaining.remove(best_candidate)
			selected.append(best_candidate)
			current_score = best_new_score
	formula = "{} ~ {} ".format(response,
								   ' + '.join(selected))
	model = smf.ols(formula, data).fit()
	return model

model = forward_selected(pitchersDF,'salary') #Initial Model selected based on adjusted R^2


y, X = dmatrices('salary ~ prev_salary + SO + SV + W + H + ER + ERA ', data=pitchersDF, return_type="dataframe") #Exract the formula as y and the data as X
vif1 = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])] #Calculate the varaince inflation factor for each predictor, including the intercept
result = smf.OLS(y, X).fit() #Resulting regression object
print model.model.formula #Print the forumla
print vif1 #Prints the variance inflation factor as a list, starting ith intercept and then going in the order of the predictors in the formula.
print result.summary() #Summary table for resulting regression model

print 'After VIF Removal (H)' #We removed H as a predictor because it's variance inflation factor was too high (20.867)
y, X = dmatrices('salary ~ prev_salary + SO + SV + W + ER + ERA', data=pitchersDF, return_type="dataframe") #Exract the formula as y and the data as X
vif1 = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])] #Calculate the varaince inflation factor for each predictor, including the intercept
result = smf.OLS(y, X).fit()  #Resulting regression object
print 'salary ~ prev_salary + SO + SV + W + ER + ERA' #Print the forumla
print vif1 #Prints the variance inflation factor as a list, starting ith intercept and then going in the order of the predictors in the formula.
print result.summary() #Summary table for resulting regression model

print 'After Insignificant Variable Removal (ERA)' #We removed ERA as a predictor because it's p-value was larger than the 0.05 significance level we were testing (0.65)
y, X = dmatrices('salary ~ prev_salary + SO + SV + W + ER', data=pitchersDF, return_type="dataframe")  #Exract the formula as y and the data as X
vif1 = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])] #Calculate the varaince inflation factor for each predictor, including the intercept
result = smf.OLS(y, X).fit() #Resulting regression object
print 'salary ~ prev_salary + SO + SV + W + ER' #Print the forumla
print vif1 #Prints the variance inflation factor as a list, starting ith intercept and then going in the order of the predictors in the formula.
print result.summary() #Summary table for resulting regression model

print 'After Insignificant Variable Removal (ER)'  #We removed ER as a predictor because it's p-value was larger than the 0.05 significance level we were testing (0.354)
y, X = dmatrices('salary ~ prev_salary + SO + SV + W', data=pitchersDF, return_type="dataframe")   #Exract the formula as y and the data as X
vif1 = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])] #Calculate the varaince inflation factor for each predictor, including the intercept
result = smf.OLS(y, X).fit() #Resulting regression object
print 'salary ~ prev_salary + SO + SV + W' #Print the forumla
print vif1 #Prints the variance inflation factor as a list, starting ith intercept and then going in the order of the predictors in the formula.
print result.summary() #Summary table for resulting regression model


pitchersDF['predicted_salary'] = result.predict() #We add a new column in our data frame that holds the predicted salaries from our final regression model

#We multiply both original salary variables by the factor we divided them by in the beginning so our graphs make sense
pitchersDF['salary'] = pitchersDF['salary']*100000 
pitchersDF['predicted_salary'] = pitchersDF['predicted_salary']*100000


sm.graphics.plot_partregress_grid(result) #This plot shows the slopes of the coefficients from each predictor variable in the model
pitchersDF.plot(x='predicted_salary',y='salary',kind="scatter",c='salary',colormap='viridis') #This plot shows the predicted_salary variable plotted against the salary variable usign a colormap
plt.title('predicted_salary vs salary') #Title for pitchersDF plot
plt.show() #Show the plots