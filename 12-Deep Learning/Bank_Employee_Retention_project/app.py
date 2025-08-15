import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
## Load the model
model = load_model('model.h5')
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# streamlit app
st.title("Customer Churn Prediction App")

# User Input
geography = st.selectbox('Geography', ('France', 'Spain', 'Germany'))
gender = st.selectbox('Gender', ('Male', 'Female'))
age = st.number_input('Age', min_value=18, max_value=100, value=30)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input(
    'Credit Score',
    min_value=300,
    max_value=850,
    value=600
)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
tenure = st.slider('Tenure', min_value=0, max_value=10, value=5)
num_of_products = st.slider('Number of Products', min_value=1, max_value=4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = {
    'CreditScore': credit_score,
    'Gender': label_encoder_gender.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary,
}
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_df = pd.DataFrame([input_data])
input_df = pd.concat([input_df.drop(['Geography'], axis=1), geo_encoded_df], axis=1)
input_data_scaled = scaler.transform(input_df)

# Make prediction

prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

if prediction_probability > 0.5:
    st.write("The employ is likely to churn.")
else:
    st.write("The employ is not likely to churn.")