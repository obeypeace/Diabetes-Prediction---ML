# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:42:41 2024

@author: hp
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMidddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers["*"]
    )

class model_input(BaseModel):
    Pregnancies : int
    Glucose : int
    BloodPressure : int
    SkinThickness : int
    Insulin : int
    BMI : float
    DiabetesPedigreeFunction : float
    Age : int
    

# loading the saved model
diabetes_model = pickle.load(open('C:/Users/hp/Desktop/Data analysis/Python/Project/diabetes/diabetes_model.sav', 'rb'))

#
@app.post('/diabetes_prediction')
def diabetes_pred(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    preg = input_dictionary['Pregnancies']
    glu = input_dictionary['Glucose']
    bp = input_dictionary['BloodPressure']
    skin = input_dictionary['SkinThickness']
    insulin = input_dictionary['Insulin']
    bmi = input_dictionary['BMI']
    dpf = input_dictionary['DiabetesPedigreeFunction']
    age = input_dictionary['Age']
    
    input_list = [preg, glu, bp, skin, insulin, bmi, dpf, age]
    
    
    # Make prediction
    prediction = diabetes_model.predict([input_list])
    
    if prediction[0] == '0':
        return 'The patient is non-diabetic'
    else:
        return 'The patient is diabetic'