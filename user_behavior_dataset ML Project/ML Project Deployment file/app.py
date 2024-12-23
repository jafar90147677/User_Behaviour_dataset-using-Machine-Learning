import pandas as pd,numpy as np
import pickle
import streamlit as st

st.title("Mobile Data User Behaviour")

# st.title("User Behavior Using Mobile Prediction")
# model_2 = pickle.load(open(r"\rfc.pkl","rb")) #pickle file path

# if st.button("Submit"):
#     result = model_2.predict([[]])
#     st.write(f"The predicted price of the rental house is {result}")
    

# age = st.number_input("Enter age",min_value=0,max_value=1000,step=1,format="%d" )
# gender = st.radio("Enter gender",['Male','Female'])
# chestpain = st.selectbox("chestpain",['non-anginal_pain', 'typical_angina', 'atypical_angina','asymptomatic'])
# restingBP =  st.number_input("Enter BP",min_value=0,max_value=1000,step=1,format="%d")
# serum_cholesterol = st.number_input("Enter serum_cholesterol",min_value=0,max_value=10000,step=1,format="%d")
# fasting_blood_sugar = st.radio("Enter fasting_blood_sugar",['yes','no'])
# restingrelectro= st.selectbox("Enter resting_electro",['ST-T_wave_abnormality', 'normal', 'left_ventricular_hypertrophy'])
# maxheartrate = st.number_input("Enter max_heart_rate",min_value=0,max_value=1000,step=1,format="%d")
# exerciseangia =  st.radio("Enter exercise_angia",['yes','no'])
# oldpeak= st.number_input("Enter oldpeak")
# slope = st.selectbox("Enter slope",['downsloping', 'upsloping', 'flat'])
# noofmajorvessels = st.selectbox("enter number of major vessels",['Three', 'One', 'Zero', 'Two'])