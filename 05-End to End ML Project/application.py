import pickle
import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## import models
ridgecv_model = pickle.load(open('Models/ridgecv.pkl','rb'))
scale = pickle.load(open('Models/scaler.pkl','rb'))
@app.route('/')
def index():
    return render_template("index.html")
if __name__ == '__main__':
    app.run(host="0.0.0.0")