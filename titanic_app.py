# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# titanic_app.py
# Streamlit web app for Titanic Survival Prediction

import streamlit as st
import pandas as pd
import pickle

# ===========================
# Load the trained model
# ===========================
model_path = r'C:\Users\tusha\OneDrive\Desktop\Project Deployment\E__Titanic_model.pkl'  # ⚠️ Ensure this file exists at this path

with open(model_path, 'rb') as file:
    titanic_logr_model = pickle.load(file)


# ===========================
# Define prediction function
# ===========================
def prediction(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    # The model expects input as a DataFrame
    features = pd.DataFrame([{
        'Pclass': Pclass,
        'Sex': Sex,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
        'Embarked': Embarked
    }])
    preds = titanic_logr_model.predict(features)[0]
    return preds


# ===========================
# Streamlit UI
# ===========================
st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details below and predict whether they survived the Titanic disaster.")

# User input fields
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 100, 25)
SibSp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
Parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
Fare = st.number_input("Fare Paid", min_value=0.0, value=50.0)
Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Predict button
if st.button("🔍 Predict Now"):
    pred = prediction(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
    if pred == 1:
        st.success("😀 The passenger **Survived**!")
    else:
        st.error("💀 The passenger **Did Not Survive**.")
