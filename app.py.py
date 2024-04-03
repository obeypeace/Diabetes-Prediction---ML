# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:43:21 2024

@author: hp
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('C:/Users/hp/Desktop/Data analysis/Python/Project/diabetes/diabetes_model.sav', 'rb'))

def predict_diabetes(input_data):
    # Convert input data type to numpy array
    input_arr = np.asarray(input_data)
    
    # Reshape the array because we are predicting for 1 instance
    input_arr2 = input_arr.reshape(1, -1)
    
    # Standardize data
    #stdz_input = scaler.transform(input_arr2)
    
    # Make prediction
    prediction = loaded_model.predict(input_arr2)
    
    if (prediction[0] == '0'):
        return 'The patient is non-diabetic'
    else:
        return 'The patient is diabetic'



def main():
    
    # giving a title to the web page
    st.title('Diabetes Prediction Web App')
    
    
    # getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age')
    
    # code for prediction
    diagnosis = ''
    
    # creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = predict_diabetes([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)
    
    

if __name__ == '__main__':
    main()