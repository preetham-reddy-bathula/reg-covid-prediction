import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import pickle

dataset = pd.read_csv('cases.csv') 

#Filtering the required features
dataset = dataset[['Date','Total Confirmed cases']]

#Grouping the similar dates and summing up count
dataset = dataset.groupby('Date',as_index = False).sum()

#Converting the date to readable format 
dataset['Date'] = pd.to_datetime(dataset['Date'])

#Since the date can't be trained, converting to Gregorian number
import datetime as dt 
dataset['Date'] = dataset['Date'].map(dt.datetime.toordinal)

#Splitting train and test sets
#Since the dataset is small, we will train all available dataset
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
regressor = LinearRegression()

#fitting model with train data
regressor.fit(X_poly, y)

#Saving model to disk
pickle.dump(regressor, open('model.pkl', 'wb'))

#loading model to compare results
#model = pickle.load(open('model.pkl','rb'))




