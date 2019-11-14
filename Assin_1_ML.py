# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:22:15 2019

@author: prasa
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style('white')

# Importing the dataset
dataset = pd.read_csv('energydata_complete.csv')
dataset=dataset.drop(['date','lights'],axis=1)
X = dataset.iloc[:,1:].values
y = dataset.iloc[:, 0].values



#part 1
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train_Linear_Auto, X_test_Linear_Auto, y_train_Linear_Auto, y_test_Linear_Auto = train_test_split(X, y, test_size = 0.3, random_state = 123)



#part2

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train_Linear_Auto, y_train_Linear_Auto)

# Predicting the Test set results
y_pred_Linear_Auto = regressor.predict(X_test_Linear_Auto)

import sklearn.metrics as metrics
r2=metrics.r2_score(y_test_Linear_Auto, y_pred_Linear_Auto)
r2

#the R-Square value we are getting is only 14.86%

#cost_Linear=regressor.coef_
#cost_Linear=np.asarray(cost_Linear)
#cost_Linear_diff=np.diff(cost_Linear)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)



#part3
def hypothesis(theta, X, n):
    h = np.ones((X.shape[0],1))
    theta = theta.reshape(1,n+1)
    for i in range(0,X.shape[0]):
        h[i] = float(np.matmul(theta, X[i]))
    h = h.reshape(X.shape[0])
    return h

def BGD(theta, alpha, num_iters, h, X, y, n):
    cost = np.ones(num_iters)
    for i in range(0,num_iters):
        theta[0] = theta[0] - (alpha/X.shape[0]) * sum(h - y)
        for j in range(1,n+1):
            theta[j] = theta[j] - (alpha/X.shape[0])*sum((h-y)* X.transpose()[j])
        h = hypothesis(theta, X, n)
        cost[i] = (1/X.shape[0]) * 0.5 * sum(np.square(h - y))
    theta = theta.reshape(1,n+1)
    return theta, cost
  
def linear_regression(X, y, alpha, num_iters):
    n = X.shape[1]
    one_column = np.ones((X.shape[0],1))
    X = np.concatenate((one_column, X), axis = 1)
    # initializing the parameter vector...
    theta = np.zeros(n+1)
    # hypothesis calculation....
    h = hypothesis(theta, X, n)
    # returning the optimized parameters by Gradient Descent...
    theta, cost = BGD(theta,alpha,num_iters,h,X,y,n)
    return theta, cost
  
mean = np.ones(X_train.shape[1])
std = np.ones(X_train.shape[1])
for i in range(0, X_train.shape[1]):
    mean[i] = np.mean(X_train.transpose()[i])
    std[i] = np.std(X_train.transpose()[i])
    for j in range(0, X_train.shape[0]):
        X_train[j][i] = (X_train[j][i] - mean[i])/std[i]

mean_train = np.ones(X_test.shape[1])
std_train = np.ones(X_test.shape[1])
for i in range(0, X_test.shape[1]):
    mean_train[i] = np.mean(X_test.transpose()[i])
    std_train[i] = np.std(X_test.transpose()[i])
    for j in range(0, X_test.shape[0]):
        X_test[j][i] = (X_test[j][i] - mean_train[i])/std_train[i]
# calling the principal function with learning_rate = 0.0001 and 
# num_iters = 100
#theta, cost = linear_regression(X_train, y_train,
#                                               0.0001, 100)
#cost = list(cost)
#n_iterations = [x for x in range(1,101)]
#plt.plot(n_iterations, cost)
#plt.xlabel('No. of iterations')
#plt.ylabel('Cost')



#part4
#we will create a new column as consumption level to signal high consumption or low consumption
dataset["consumption_level"]=[1 if x>=dataset["Appliances"].mean() else 0 for x in dataset["Appliances"]]

X_Logistic = dataset.iloc[:,1:27].values #X = dataset.iloc[:,1:].values

y_Logistic = dataset.iloc[:,27].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train_Logistic, X_test_Logistic, y_train_Logistic, y_test_Logistic = train_test_split(X_Logistic, y_Logistic, test_size = 0.3, random_state = 123)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_Logistic = sc.fit_transform(X_train_Logistic)
X_test_Logistic = sc.transform(X_test_Logistic)

# Fitting classifier to the Training set
from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier(loss="log",alpha=0.0001, average=False, class_weight=None,
       early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
       l1_ratio=0.15, learning_rate='optimal', max_iter=1000,
       n_iter_no_change=5, n_jobs=None, penalty='l2', power_t=0.5,
       random_state=None, shuffle=True, tol=0.001, validation_fraction=0.1,
       verbose=0, warm_start=False)
classifier.fit(X_train_Logistic, y_train_Logistic)

# Predicting the Test set results
y_pred_Logistic = classifier.predict(X_test_Logistic)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confuse_mat = confusion_matrix(y_test_Logistic, y_pred_Logistic)

from sklearn.metrics import accuracy_score
accuracy_Logistic_1=(accuracy_score(y_test_Logistic, y_pred_Logistic)*100)
print(accuracy_Logistic_1)



#Experiment 1

#Linear Regression

#Training data
#1st Experiment
alpha=0.01
theta_Linear_Ex1, cost_Linear_Ex1 = linear_regression(X_train, y_train,
                                               alpha, 1000)
cost_Linear_Ex1 = list(cost_Linear_Ex1)
n_iterations = [x for x in range(1,1001)]
plt.plot(n_iterations, cost_Linear_Ex1)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
cost_Linear_Ex1.pop()
#cost=np.asarray(cost)
#cost_diff=np.diff(cost)
#cost_diff



#2nd Experiment
alpha=0.04
theta_Linear_Ex2, cost_Linear_Ex2 = linear_regression(X_train, y_train,
                                               alpha, 1000)
cost_Linear_Ex2= list(cost_Linear_Ex2)
n_iterations = [x for x in range(1,1001)]
plt.plot(n_iterations, cost_Linear_Ex2)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
cost_Linear_Ex2.pop()


#3rd Experiment
alpha=0.005
theta_Linear_Ex3, cost_Linear_Ex3 = linear_regression(X_train, y_train,
                                               alpha, 1000)
cost_Linear_Ex3 = list(cost_Linear_Ex3)
n_iterations = [x for x in range(1,1001)]
plt.plot(n_iterations, cost_Linear_Ex3)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
cost_Linear_Ex3.pop()


#4th Experiment
alpha=0.1
theta_Linear_Ex4, cost_Linear_Ex4 = linear_regression(X_train, y_train,
                                               alpha, 1000)
cost_Linear_Ex4 = list(cost_Linear_Ex4)
n_iterations = [x for x in range(1,1001)]
plt.plot(n_iterations, cost_Linear_Ex4)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
cost_Linear_Ex4.pop()

#5th Experiment
alpha=0.2
theta_Linear_Ex5, cost_Linear_Ex5 = linear_regression(X_train, y_train,
                                               alpha, 1000)
cost_Linear_Ex5 = list(cost_Linear_Ex5)
n_iterations = [x for x in range(1,1001)]
plt.plot(n_iterations, cost_Linear_Ex5)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
cost_Linear_Ex5.pop()

#Testing data
#1st Experiment
alpha=0.01
theta_Linear_Ex1_test, cost_Linear_Ex1_test = linear_regression(X_test, y_test,
                                               alpha, 1000)
cost_Linear_Ex1_test = list(cost_Linear_Ex1_test)
n_iterations = [x for x in range(1,1001)]
plt.plot(n_iterations, cost_Linear_Ex1_test)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
cost_Linear_Ex1_test.pop()

#2nd Experiment
alpha=0.1
theta_Linear_Ex2_test, cost_Linear_Ex2_test = linear_regression(X_test, y_test,
                                               alpha, 1000)
cost_Linear_Ex2_test = list(cost_Linear_Ex2_test)
n_iterations = [x for x in range(1,1001)]
plt.plot(n_iterations, cost_Linear_Ex2_test)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
cost_Linear_Ex2_test.pop()

#3rd Experiment
alpha=0.2
theta_Linear_Ex3_test, cost_Linear_Ex3_test = linear_regression(X_test, y_test,
                                               alpha, 1000)
cost_Linear_Ex3_test = list(cost_Linear_Ex3_test)
n_iterations = [x for x in range(1,1001)]
plt.plot(n_iterations, cost_Linear_Ex3_test)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
cost_Linear_Ex3_test.pop()


#Logistic Regression
#Training data
# Fitting classifier to the Training set
from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier(loss="log",alpha=0.25, average=False, class_weight=None,
       early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
       l1_ratio=0.15, learning_rate='optimal', max_iter=1000,
       n_iter_no_change=5, n_jobs=None, penalty='l2', power_t=0.5,
       random_state=None, shuffle=True, tol=0.001, validation_fraction=0.1,
       verbose=0, warm_start=False)
classifier.fit(X_train_Logistic, y_train_Logistic)

# Predicting the Test set results
y_pred_Logistic = classifier.predict(X_test_Logistic)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confuse_mat = confusion_matrix(y_test_Logistic, y_pred_Logistic)

from sklearn.metrics import accuracy_score
accuracy_Logistic_1=(accuracy_score(y_test_Logistic, y_pred_Logistic)*100)
print(accuracy_Logistic_1)

# Fitting classifier to the Training set
from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier(loss="log",alpha=0.75, average=False, class_weight=None,
       early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
       l1_ratio=0.15, learning_rate='optimal', max_iter=1000,
       n_iter_no_change=5, n_jobs=None, penalty='l2', power_t=0.5,
       random_state=None, shuffle=True, tol=0.001, validation_fraction=0.1,
       verbose=0, warm_start=False)
classifier.fit(X_train_Logistic, y_train_Logistic)

# Predicting the Test set results
y_pred_Logistic = classifier.predict(X_test_Logistic)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confuse_mat = confusion_matrix(y_test_Logistic, y_pred_Logistic)

from sklearn.metrics import accuracy_score
accuracy_Logistic_1=(accuracy_score(y_test_Logistic, y_pred_Logistic)*100)
print(accuracy_Logistic_1)


####################################################################################

#Experiment-2

#Training data
#1st thereshold
alpha=0.2
iterations=2000
theta_Linear_Exp2, cost_Linear_Exp2 = linear_regression(X_train, y_train,
                                               alpha, iterations)
cost_Linear_Exp2 = list(cost_Linear_Exp2)
n_iterations = [x for x in range(1,2001)]
plt.plot(n_iterations, cost_Linear_Exp2)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
cost_Linear_Exp2.pop()
thereshold_1=cost_Linear_Exp2[len(cost_Linear_Exp2)-1]-cost_Linear_Exp2[len(cost_Linear_Exp2)-2]
thereshold_1

#2nd thereshold
alpha=0.2
iterations=3000
theta_Linear_Exp3, cost_Linear_Exp3 = linear_regression(X_train, y_train,
                                               alpha, iterations)
cost_Linear_Exp3 = list(cost_Linear_Exp3)
n_iterations = [x for x in range(1,3001)]
plt.plot(n_iterations, cost_Linear_Exp3)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
cost_Linear_Exp3.pop()
thereshold_2=cost_Linear_Exp3[len(cost_Linear_Exp3)-1]-cost_Linear_Exp3[len(cost_Linear_Exp3)-2]
thereshold_2


#3rd thereshold
alpha=0.2
iterations=4000
theta_Linear_Exp4, cost_Linear_Exp4 = linear_regression(X_train, y_train,
                                               alpha, iterations)
cost_Linear_Exp4 = list(cost_Linear_Exp4)
n_iterations = [x for x in range(1,4001)]
plt.plot(n_iterations, cost_Linear_Exp4)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
cost_Linear_Exp4.pop()
thereshold_3=cost_Linear_Exp4[len(cost_Linear_Exp4)-1]-cost_Linear_Exp4[len(cost_Linear_Exp4)-2]
thereshold_3

#testing data
alpha=0.2
iterations=2000

theta_Linear_Exp2_test, cost_Linear_Exp2_test = linear_regression(X_test, y_test,
                                               alpha, iterations)
cost_Linear_Exp2_test = list(cost_Linear_Exp2_test)
n_iterations = [x for x in range(1,2001)]
plt.plot(n_iterations, cost_Linear_Exp2_test)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
cost_Linear_Exp2_test.pop()
thereshold_4=cost_Linear_Exp2_test[len(cost_Linear_Exp2_test)-1]-cost_Linear_Exp2_test[len(cost_Linear_Exp2_test)-2]
thereshold_4


#testing data
alpha=0.2
iterations=3000

theta_Linear_Exp3_test, cost_Linear_Exp3_test = linear_regression(X_test, y_test,
                                               alpha, iterations)
cost_Linear_Exp3_test = list(cost_Linear_Exp3_test)
n_iterations = [x for x in range(1,3001)]
plt.plot(n_iterations, cost_Linear_Exp3_test)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
cost_Linear_Exp3_test.pop()
thereshold_5=cost_Linear_Exp3_test[len(cost_Linear_Exp3_test)-1]-cost_Linear_Exp3_test[len(cost_Linear_Exp3_test)-2]
thereshold_5


#testing data
alpha=0.2
iterations=4000

theta_Linear_Exp4_test, cost_Linear_Exp4_test = linear_regression(X_test, y_test,
                                               alpha, iterations)
cost_Linear_Exp4_test = list(cost_Linear_Exp4_test)
n_iterations = [x for x in range(1,4001)]
plt.plot(n_iterations, cost_Linear_Exp4_test)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
cost_Linear_Exp4_test.pop()
thereshold_6=cost_Linear_Exp4_test[len(cost_Linear_Exp4_test)-1]-cost_Linear_Exp4_test[len(cost_Linear_Exp4_test)-2]
thereshold_6



##################################################################################################
#Experiment 3

#Picking out 10 random features
X_random = dataset.iloc[:,1:11].values
y_random = dataset.iloc[:, 0].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train_Linear_Auto_Random, X_test_Linear_Auto_Random, y_train_Linear_Auto_Random, y_test_Linear_Auto_Random = train_test_split(X_random, y_random, test_size = 0.3, random_state = 123)

#sklearn Linear Regression
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor_random = LinearRegression()
regressor_random.fit(X_train_Linear_Auto_Random, y_train_Linear_Auto_Random)

# Predicting the Test set results
y_pred_Linear_Auto_Random = regressor_random.predict(X_test_Linear_Auto_Random)

import sklearn.metrics as metrics
r2_random=metrics.r2_score(y_test_Linear_Auto_Random, y_pred_Linear_Auto_Random)
r2_random

#we will create a new column as consumption level to signal high consumption or low consumption

X_Logistic_random = dataset.iloc[:,1:11].values #X = dataset.iloc[:,1:].values

y_Logistic_random = dataset.iloc[:,27].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train_Logistic_random, X_test_Logistic_random, y_train_Logistic_random, y_test_Logistic_random = train_test_split(X_Logistic_random, y_Logistic_random, test_size = 0.3, random_state = 123)


from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier(loss="log",alpha=0.9, average=False, class_weight=None,
       early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
       l1_ratio=0.15, learning_rate='optimal', max_iter=1000,
       n_iter_no_change=5, n_jobs=None, penalty='l2', power_t=0.5,
       random_state=None, shuffle=True, tol=0.001, validation_fraction=0.1,
       verbose=0, warm_start=False)
classifier.fit(X_train_Logistic_random, y_train_Logistic_random)

# Predicting the Test set results
y_pred_random = classifier.predict(X_test_Logistic_random)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confuse_mat_random = confusion_matrix(y_test_Logistic_random, y_pred_random)

from sklearn.metrics import accuracy_score
accuracy_Logistic_random=(accuracy_score(y_test_Logistic_random, y_pred_random)*100)
print(accuracy_Logistic_random)



#Grdaient Desccent for Random dataset 

#scaling the values
mean_random = np.ones(X_train_Linear_Auto_Random.shape[1])
std = np.ones(X_train_Linear_Auto_Random.shape[1])
for i in range(0, X_train_Linear_Auto_Random.shape[1]):
    mean_random[i] = np.mean(X_train_Linear_Auto_Random.transpose()[i])
    std[i] = np.std(X_train_Linear_Auto_Random.transpose()[i])
    for j in range(0, X_train_Linear_Auto_Random.shape[0]):
        X_train_Linear_Auto_Random[j][i] = (X_train_Linear_Auto_Random[j][i] - mean_random[i])/std[i]

mean_train = np.ones(X_test_Linear_Auto_Random.shape[1])
std_train = np.ones(X_test_Linear_Auto_Random.shape[1])
for i in range(0, X_test_Linear_Auto_Random.shape[1]):
    mean_train[i] = np.mean(X_test_Linear_Auto_Random.transpose()[i])
    std_train[i] = np.std(X_test_Linear_Auto_Random.transpose()[i])
    for j in range(0, X_test_Linear_Auto_Random.shape[0]):
        X_test_Linear_Auto_Random[j][i] = (X_test_Linear_Auto_Random[j][i] - mean_train[i])/std_train[i]
alpha=0.2
theta_Linear_Exp_random, cost_Linear_Exp_random = linear_regression(X_train_Linear_Auto_Random, y_train_Linear_Auto_Random,
                                               alpha, 1000)
cost_Linear_Exp_random = list(cost_Linear_Exp_random)
n_iterations = [x for x in range(1,1001)]
plt.plot(n_iterations, cost_Linear_Exp_random)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
cost_Linear_Exp_random.pop()

alpha=0.2
theta_Linear_Exp_test_random, cost_Linear_Exp_test_random = linear_regression(X_test_Linear_Auto_Random, y_test_Linear_Auto_Random,
                                               alpha, 1000)
cost_Linear_Exp_test_random = list(cost_Linear_Exp_test_random)
n_iterations = [x for x in range(1,1001)]
plt.plot(n_iterations, cost_Linear_Exp_test_random)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
cost_Linear_Exp_test_random.pop()


##############################################################################################

#Experiment 4

corr = dataset.corr()
# Mask the repeated values
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
  
f, ax = plt.subplots(figsize=(16, 14))
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, annot=True, fmt=".2f" , mask=mask,)
    #Apply xticks
plt.xticks(range(len(corr.columns)), corr.columns);
    #Apply yticks
plt.yticks(range(len(corr.columns)), corr.columns)
    #show plot
plt.show()

#Picking out 10 random features
X_Picked = dataset.iloc[:,[1,11,19,21,2,5,3,7,15,6]].values 

y_Picked = dataset.loc[:, [0]].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train_Linear_Auto_Picked, X_test_Linear_Auto_Picked, y_train_Linear_Auto_Picked, y_test_Linear_Auto_Picked = train_test_split(X_Picked, y_Picked, test_size = 0.3, random_state = 123)


#sklearn Linear Regression
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor_Picked = LinearRegression()
regressor_Picked.fit(X_train_Linear_Auto_Picked, y_train_Linear_Auto_Picked)

# Predicting the Test set results
y_pred_Linear_Auto_Picked = regressor_Picked.predict(X_test_Linear_Auto_Picked)

import sklearn.metrics as metrics
r2_Picked=metrics.r2_score(y_test_Linear_Auto_Picked, y_pred_Linear_Auto_Picked)
r2_Picked

#we will create a new column as consumption level to signal high consumption or low consumption

X_Logistic_Picked = dataset.loc[:,['T2','T6','T_out','Windspeed','RH_1','T3','T1','T4','T8','RH_3']].values

y_Logistic_Picked = dataset.iloc[:,27].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train_Logistic_Picked, X_test_Logistic_Picked, y_train_Logistic_Picked, y_test_Logistic_Picked = train_test_split(X_Logistic_Picked, y_Logistic_Picked, test_size = 0.3, random_state = 123)


from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier(loss="log",alpha=0.9, average=False, class_weight=None,
       early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
       l1_ratio=0.15, learning_rate='optimal', max_iter=1000,
       n_iter_no_change=5, n_jobs=None, penalty='l2', power_t=0.5,
       random_state=None, shuffle=True, tol=0.001, validation_fraction=0.1,
       verbose=0, warm_start=False)
classifier.fit(X_train_Logistic_Picked, y_train_Logistic_Picked)

# Predicting the Test set results
y_pred_Picked = classifier.predict(X_test_Logistic_Picked)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confuse_mat_Picked = confusion_matrix(y_test_Logistic_Picked, y_pred_Picked)

from sklearn.metrics import accuracy_score
accuracy_Logistic_Picked=(accuracy_score(y_test_Logistic_Picked, y_pred_Picked)*100)
print(accuracy_Logistic_Picked)


#Grdaient Desccent for Picked dataset 

#alpha=0.2
#theta_Linear_Exp_Picked, cost_Linear_Exp_Picked = linear_regression(X_train_Linear_Auto_Picked, y_train_Linear_Auto_Picked,
#                                               alpha, 1000)
#cost_Linear_Exp_Picked = list(cost_Linear_Exp_Picked)
#n_iterations = [x for x in range(1,1001)]
#plt.plot(n_iterations, cost_Linear_Exp_Picked)
#plt.xlabel('No. of iterations')
#plt.ylabel('Cost')
#cost_Linear_Exp_Picked.pop()
#
#alpha=0.2
#theta_Linear_Exp_test_Picked, cost_Linear_Exp_test_Picked = linear_regression(X_test_Linear_Auto_Picked, y_test_Linear_Auto_Picked,
#                                               alpha, 1000)
#cost_Linear_Exp_test_Picked = list(cost_Linear_Exp_test_Picked)
#n_iterations = [x for x in range(1,1001)]
#plt.plot(n_iterations, cost_Linear_Exp_test_Picked)
#plt.xlabel('No. of iterations')
#plt.ylabel('Cost')
#cost_Linear_Exp_test_Picked.pop()


#Picking out 10 random features
X_Picked_Gradient = dataset.loc[:,['T2','T6','T_out','Windspeed','RH_1','T3','T1','T4','T8','RH_3']].values 

y_Picked_Gradient = dataset.loc[:, ['Appliances']].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train_Linear_Auto_Picked, X_test_Linear_Auto_Picked, y_train_Linear_Auto_Picked, y_test_Linear_Auto_Picked = train_test_split(X_Picked, y_Picked, test_size = 0.3, random_state = 123)



#scaling the values
mean_Picked = np.ones(X_train_Linear_Auto_Picked.shape[1])
std = np.ones(X_train_Linear_Auto_Picked.shape[1])
for i in range(0, X_train_Linear_Auto_Picked.shape[1]):
    mean_Picked[i] = np.mean(X_train_Linear_Auto_Picked.transpose()[i])
    std[i] = np.std(X_train_Linear_Auto_Picked.transpose()[i])
    for j in range(0, X_train_Linear_Auto_Picked.shape[0]):
        X_train_Linear_Auto_Picked[j][i] = (X_train_Linear_Auto_Picked[j][i] - mean_Picked[i])/std[i]

mean_train = np.ones(X_test_Linear_Auto_Picked.shape[1])
std_train = np.ones(X_test_Linear_Auto_Picked.shape[1])
for i in range(0, X_test_Linear_Auto_Picked.shape[1]):
    mean_train[i] = np.mean(X_test_Linear_Auto_Picked.transpose()[i])
    std_train[i] = np.std(X_test_Linear_Auto_Picked.transpose()[i])
    for j in range(0, X_test_Linear_Auto_Picked.shape[0]):
        X_test_Linear_Auto_Picked[j][i] = (X_test_Linear_Auto_Picked[j][i] - mean_train[i])/std_train[i]
