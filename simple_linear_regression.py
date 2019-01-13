# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 21:52:12 2018

@author: Sayantan
"""
#Simple Linear Regression Model on Salary Statics



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#reading the datasets
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values


"""#Handling the missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding catagorical Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)"""

#Splitting the dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/3,random_state = 0)

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Fitting Simple Linear regeression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
accuracy = regressor.score(X_test,Y_test)

print(accuracy*100,'%')

#Prediction the test set Results
Y_pred = regressor.predict(X_test)
print(regressor.predict(10.5))









#Visualising the training set Results
plt.scatter(X_train, Y_train,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.title('Salary vs Experince(Traning Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising the test set Results
plt.scatter(X_test, Y_test,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.title('Salary vs Experince(Test Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()







