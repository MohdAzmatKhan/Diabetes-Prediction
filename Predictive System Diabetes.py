# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import joblib as jb 

model = jb.load('C:/Users/Dell/Desktop/ML/Diabetes_test_nonscaler.joblib')

input_data = (5,166,72,19,175,25.8,0.587,51)

new_data = np.asarray(input_data)

new_data_rs = new_data.reshape(1,-1)

# new_data_rs_std = scaler.transform(new_data_rs)

#print(new_data_rs_std)

prediction = model.predict(new_data_rs)

if ( prediction[0]== 0):
    print('Person is Non-Diabetic')
else :
    print('Person is Diabetic')