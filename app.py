import numpy as np 
from flask import Flask,request, jsonify, render_template
import pickle
from sklearn.preprocessing import PolynomialFeatures
import datetime as dt
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    int_features =   request.form['Date']
    x=np.array([[(dt.datetime.toordinal(pd.to_datetime(int_features)))+1]]) 
    poly_reg = PolynomialFeatures(degree = 4)
    prediction = model.predict(poly_reg.fit_transform(x))

    output = int(round(prediction[0]))

    return render_template('index.html', prediction_text='Cases = {}'.format(output))

if __name__=="__main__":
    app.run(debug=True)    
