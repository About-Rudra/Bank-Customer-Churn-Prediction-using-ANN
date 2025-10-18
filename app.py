import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle

# Load the trained model
model = tf.keras.models.load_model('churn_model.h5')

# Load the scaler and encoders
with open('scaler.pkl', 'rb') as f: 
    scaler = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('lablel_encoder_gender.pkl', 'rb') as f:
    lablel_encoder_gender = pickle.load(f)

#streamlit app
st.title("Rudra's Customer Churn Prediction: Accuracy 86%")

st.write("Enter customer details to predict churn probability.")
# Input fields
credit_score = st.number_input("Credit Score", min_value=300, max_value=850
, value=600)
geography = st.selectbox("Geography", options=["France", "Spain", "Germany"])
gender = st.selectbox("Gender", options =["Male", "Female"])
age = st.slider("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Balance")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure (years)", min_value=0, max_value=10, value=5)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card", options=[0, 1])
is_active_member = st.selectbox("Is Active Member", options=[0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [lablel_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

#one hot encode geography
geo_encoded = onehot_encoder_geo.transform([[geography]])
geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data, geo_df], axis=1)

# Scale numerical features
input_data_scaled = scaler.transform(input_data)

# Predict churn probability
prediction = model.predict(input_data_scaled)
churn_probability = prediction[0][0]

if churn_probability >= 0.5:
    st.error(f"Churn Probability: {churn_probability:.2f} - High risk of churn")    
else:
    st.success(f"Churn Probability: {churn_probability:.2f} - Low risk of churn")

