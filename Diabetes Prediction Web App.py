# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 13:02:54 2023

@author: Dell
"""

import numpy as np
import joblib as jb
import streamlit as st

model = jb.load('C:/Users/Dell/Desktop/ML/Diabetes_test_nonscaler.joblib')

def diabetes_prediction(input_data) :
    

    new_data = np.asarray(input_data)

    new_data_rs = new_data.reshape(1,-1)

    # new_data_rs_std = scaler.transform(new_data_rs)

    #print(new_data_rs_std)

    prediction = model.predict(new_data_rs)

    if ( prediction[0]== 0):
        return 'Person is Non-Diabetic'
    else :
        return 'Person is Diabetic'
    

def main():
    
    st.title('Diabetes Prediction BTP-Project')
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    Age = st.text_input('Age of the Person')
    
    diagnosis = ''
    
    if st.button('Diabetes Test Result') :
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
        
        
    st.success(diagnosis)
    
if __name__ =='__main__':
    main()
    