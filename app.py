import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Streamlit Page Setup

st.set_page_config(
   page_title= "Indian Car Price Predictor",
   layout='wide'
)

# Load model
@st.cache_resource
def load_model():
    try:
       with open('rf_model.pkl', 'rb') as file:
            model = pickle.load(file)
       return model
    except Exception as e:
         st.error(f'Failed to load model: {e}')
         st.stop()

model = load_model()

#----------------------

st.title(
 "CARS SELLING PRICE APP IN INDIAN MARKET")
st.markdown("### Provide the information below to get a car price estimate.")

# Input Section

#-----

# Input fields for the features
car_name = st.text_input('Car Name'),
vehicle_age = st.number_input('Vehicle Age', min_value=0, max_value=9),
km_driven = st.number_input('Kilometers Driven', min_value= 100, max_value = 3800000),
seller_type = st.selectbox('Seller Type', options=['Individual', 'Dealer', 'Trustmark Dealer']),
fuel_type = st.selectbox('Fuel Type', options=['CNG', 'Diesel', 'Electric', 'LPG', 'Petrol']),
transmision_type = st.selectbox('Transmission Type', options=['Automatic', 'Manual']),
mileage = st.number_input("Mileage", min_value= 4.0, max_value = 33.54),
engine = st.number_input("Engine", min_value= 793, max_value = 6592),
max_power = st.number_input("Maximum Power", min_value= 38.4, max_value = 626.5),
seats = st.number_input("Seats", min_value= 0, max_value = 9)
      

# Feature Engineering

# -----------------

current_year = 2025
#year = current_year - vehicle_age

# Button to make prediction
if st.button("Predict Price"):
    # Prepare the input data for prediction
    input_data = pd.DataFrame([{
              'vehicle_age': vehicle_age,
              'km_driven': km_driven,
              'seller_type': seller_type,
              'fuel_type': fuel_type,
              'transmision_type': transmision_type,
              'car_name': car_name,        
              'mileage': mileage,
              'engine': engine,
              'max_power': max_power,
              'seats': seats
}])

#Encode the categorical Variables
    input_data['seller_type'] = input_data['seller_type'].map({'Individual': 0, 'Dealer': 1, 'Trustmark Dealer': 2}),
    input_data['fuel_type'] = input_data['fuel_type'].map({'CNG':0, 'Diesel':1, 'Electric':2, 'LPG':3, 'Petrol':4}),
    input_data['transmision_type'] = input_data['transmision_type'].map({'Automatic':0, 'Manual':1})
   

 # Make preiction
#if st.button('Predict Price'):
    predicted_price = model.predict(input_data)[0]

 # Display the result
    st.success(f"The predicted price of the car is: {predicted_price:.f}")
 
        
      