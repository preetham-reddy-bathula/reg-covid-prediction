import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import pickle

dataset = pd.read_csv('data_7JULY.csv') 
dataset.rename(columns={'/0/date':'Date',
                   '/0/totalconfirmed':'Cases'}, inplace=1)

#Filtering the required features
dataset = dataset[['Date','Cases']]

#Changing date format
d=[]
for date in dataset['Date']:
    date = date.split()
    for n,i in enumerate(date):
        if i=='January':
            date[n]='-01-2020'
        if i=='February':
            date[n]='-02-2020'
        if i=='March':
            date[n]='-03-2020'
        if i=='April':
            date[n]='-04-2020'
        if i=='May':
            date[n]='-05-2020'
        if i=='June':
            date[n]='-06-2020'
        if i=='July':
            date[n]='-07-2020'
        if i=='August':
            date[n]='-08-2020'    
        date[0:1] = [''.join(date[0:1])]
    
    date1 = (''.join(date))
    d.append(date1)
    
dataset['Date'] = d



#Converting the date to readable format 
dataset['Date'] = pd.to_datetime(dataset['Date'],format='%d-%m-%Y')

#Since the date can't be trained, converting to Gregorian number
import datetime as dt 
dataset['Date'] = dataset['Date'].map(dt.datetime.toordinal)

dataset = dataset[dataset['Date'] > 737540]

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




