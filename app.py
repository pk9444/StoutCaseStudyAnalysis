from flask import Flask, render_template, request, redirect, url_for, app
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import pickle
import base64


# create an instance of the app as a Flask Object
app = Flask(__name__)

# load the pickle file containing the Logistic Regression classifier object into a variable
LR_pipeline = pickle.load(open('model_LR.pkl','rb'))

# Route the home page of the web app
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/description')
def description():
    return render_template('description.html')

@app.route('/prediction')
def prediction():
    #return "<xmp>"  "\n" + "Prediction" + "\n" + " </xmp> "
    return render_template('prediction.html')

@app.route('/prediction/detectFraud', methods=['POST','GET'])
def detect():

    # amount = request.form['amount']
    # oldBalanceOrg = request.form['oldBalanceOrg']
    # newBalanceOrg = request.form['newBalanceOrg']
    # oldBalanceDest = request.form['oldBalanceDest']
    # newBalanceDest = request.form['newBalanceDest']
    # isFraud = request.form['isFraud']

    form_data = [int(index) for index in request.form.values()]
    feature_set = [np.array(form_data)]

    predict_fraud = LR_pipeline.predict(feature_set)

    if predict_fraud == 0:
        return "<xmp>"  "\n" + "not fraudulent! " + "\n" + " </xmp> "
    elif predict_fraud == 1:
        return "<xmp>"  "\n" + "fraudulent!" + "\n" + " </xmp> "
       #return "<xmp>"  "\n" + "fraudulent" + "\n" + " </xmp> "

@app.route('/visualizations')
def visualization():
    return render_template('visualizations.html')

@app.route('/evaluation')
def evaluation():
    return render_template('evaluation.html')

# Driver Function
if __name__ == '__main__':
    app.run(debug=True)