import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import datasets
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

#splitting data into dataset and training set
from sklearn.model_selection import train_test_split 
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=1/3, random_state = 0)

#Fitting simple linear resgreesion to training set
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
regressor = LinearRegression()
regressor.fit(X_train,Y_train) 


#predicting testset results
y_pred = regressor.predict(X_test)
print(r2_score(Y_test,y_pred))


#Visualising training set results
plt.scatter(X_train,Y_train, color = 'red')
plt.plot(X_train , regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#Visualising test set results
plt.scatter(X_test,Y_test, color = 'red')
plt.plot(X_train , regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


