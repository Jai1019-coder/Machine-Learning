import pickle
import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## import models
ridgecv_model = pickle.load(open('models/ridgecv.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predictdata',methods = ['GET','POST'])
def predict_data():
    if request.method == 'POST':
        temperature = float(request.form['Temperature'])
        rh = float(request.form['RH'])
        ws = float(request.form['Ws'])
        rain = float(request.form['Rain'])
        ffmc = float(request.form['FFMC'])
        dmc = float(request.form['DMC'])
        isi = float(request.form['ISI'])
        classes = float(request.form['Classes'])
        region = float(request.form['Region'])

        data = [[temperature,rh,ws,rain,ffmc,dmc,isi,classes,region]]
        new_scaled_data = standard_scaler.transform(data)
        result = ridgecv_model.predict(new_scaled_data)
        return render_template('collect.html',results = result[0][0])
    else:
        return render_template('collect.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0")