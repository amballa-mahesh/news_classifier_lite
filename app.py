from flask import Flask, redirect, url_for,render_template,request
import numpy as np
import joblib 
from joblib import dump, load
import pandas as pd
import logging
import os
from src.logger import logging
from pickle import load
from src.utils import text_pipeline,model_predict_nn,model_predict_rf
from keras.models import load_model



# import mysql.connector
# import cassandra
# from datetime import datetime
# from cassandra.cluster import Cluster
# from cassandra.auth import PlainTextAuthProvider
# print(os.getcwd())




# logging.info('libraries loaded...')
tokenizer = load(open("artifacts/data/tokenizer.p","rb"))
print('tokenizer loaded...')
     
model_rf = joblib.load('artifacts/models/model_rf.pkl')
print('Model Loaded..')



app = Flask(__name__)



@app.route('/')
def welcome_user():
    return render_template('index.html')


@app.route('/submit', methods = ['POST','GET'])
def submit():
    back = request.referrer
    if request.method == 'POST':
        data  = request.form['entered_text']
        X =  text_pipeline(data,tokenizer)
        prediction = model_predict_rf(X,model_rf)
        print(prediction[0])
        return render_template('index.html',result = prediction[0].upper())
    return redirect(back)
    
    
    
if __name__== '__main__':
    app.run(debug=True)