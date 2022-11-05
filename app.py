import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''Testing the api With postman'''
    data=request.json['data'] #getting the data sent to this post
    unscaled_data = np.array(list(data.values())).reshape(1, -1) #reshaping to 2 dimensions
    scaled_data = scaler.transform(unscaled_data) #scaling for our ML prediction
    output = regmodel.predict(scaled_data) #predicting
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    scaled_input = scaler.transform(np.array(data).reshape(1,-1))
    print(scaled_input)
    output = regmodel.predict(scaled_input)[0] #predicting
    return render_template('home.html', prediction_text=f'The predicted price: {output}')

if __name__=="__main__":
    app.run(debug=True)
