import streamlit as st
import pandas as pd
import numpy as np
import pickle


#-------------------------------------------------
# page Configuration
# -------------------------------------------------
st.set_page_config(
   page_title= "Indian Car Price Predictor",
   layout='wide'
)

#--------------------------------------------------
# Load Artifacts
#-------------------------------------------------
with open('rf_model.pkl', 'rb') as file:
     rf_model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
     scaler = pickle.load(file)
      
with open('feature_columns.pkl', 'rb') as file:
     feature_columns = pickle.load(file)

with open('model_metrics.pkl', 'rb') as file:
     metrics = pickle.load(file)

# Extract Metrics
r2 = metrics['r2']
mae = metrics['mae']
rmse = metrics['rmse']

# Sidebar
with st.sidebar:
      st.title("About")
      st.info("""This app predicts used car prices in Indian using a  **Random Forest** model trained on 154111 car.""" )
      st.subheader("Model Performance")

      st.metric("R2", round(r2, 3))
      st.metric("MAE", f"{int(mae):,}")
      st.metric("RMSE", f"{int(rmse):,}")

      st.markdown("---")
      st.markdown("### Information")
#---------------------------------------------------
# Title & Discription
# ------------------------------------------------
st.title(
 "CARS SELLING PRICE PREDICTION - INDIAN MARKET")
st.markdown("""This application Predicts the **selling price of used cars in India** using a **Random Forest Regression Model** trained on the
   CarDekho dataset.""")

#------------------------------------------------
# User Inputs
#------------------------------------------------
car_name = st.selectbox('Car Name', options=['Audi A4', 'Audi A6', 'Audi A8', 'Audi Q7', 'Bentley Continental', 'BMW 3',
                                            'BMW 5', 'BMW 6', 'BMW 7', 'BMW X1','BMW X3', 'BMW X4', 'BMW X5', 'BMW Z4',
                                            'Datsun GO','Datsun RediGO', 'Datsun redi-GO', 'Ferrari GTC4Lusso', 'Force Gurkha', 'Ford Aspire',
                                            'Ford Ecosport', 'Ford Endeavour', 'Ford Figo','Ford Freestyle', 'Honda Amaze', 'Honda City', 
                                             'Honda Civic', 'Honda CR','Honda CR-V','Honda Jazz','Honda WR-V', 'Hyundai Aura', 'Hyundai Creta',
                                             'Hyundai Elantra', 'Hyundai Grand','Hyundai i10', 'Hyundai i20', 'Hyundai Santro', 'Hyundai Tucson',
                                             'Hyundai Venue', 'Hyundai Verne', 'Isuzu D-Max', 'ISUZU MUX', 'Jaguar F-PACE', 'Jaguar XE', 
                                             'Jaguar XF', 'Jeep Compass', 'Jeep Wrangler', 'Kia Carnival', 'Kia Seltos', 'Land Rover Rover', 
                                             'Lexus ES', 'Lexus NX', 'Lexus RX', 'Mahindra Alturas', 'Mahindra Bolero', 'Mahindra KUV', 
                                             'Mahindra KUV100', 'Mahindra Murazzo','Mahindra Scorpio','Mahindra Thar','Mahindra XUV300',
                                             'Mahindra XUV500', 'Maruti Alto', 'Maruti Baleno', 'Maruti Celerio', 'Maruti Ciaz', 
                                             'Maruti Dzire LXI', 'Maruti  Dzire VXI','Maruti Dzire ZXI', 'Maruti Eeco','Maruti Ertiga',
                                             'Maruti Ignis','Maruti S-Presso','Maruti Swift', 'Maruti Swift Dzire','Maruti Vitara',      
                                             'Maruti Wagon R', 'Maruti XL6', 'Maserati Ghibli', 'Maserati Quattroporte', 'Mercedes-AMG C',
                                             'Mercedes-Benz C-Class', 'Mercedes-Benz GL-Class', 'Mercedes-Benz GLS', 'Mercedes-Benz S-Class',
                                             'MG Hector', 'Mini Cooper', 'Nissan Kicks', 'Nissan X-Trail', 'Porsche Cayenne', 'Porsche Macan',
                                             'Porsche Panamera', 'Renault Duster', 'Renault KWID','Renault Triber', 'Rolls-Royce Ghost', 
                                             'Skoda Octavia', 'Skoda Rapid','Skoda Superb', 'Tata Altroz', 'Tata Harrier', 'Tata Hexa', 
                                             'Tata Nexon', 'Tata Safari', 'Tata Tiago', 'Tata Tigor', 'Toyota Camry', 'Toyota Fortuner', 
                                             'Toyota Glanza', 'Toyota Innova', 'Toyota Yaris', 'Volkswagen Polo', 'Volkswagen Vento', 'Vol S90', 
                                             'Volo XC', 'Volo XC60', 'Volo XC90'], index=None, placeholder='Select the Car...'),
vehicle_age = st.number_input('Vehicle Age (in years)', min_value=0, max_value=9),
km_driven = st.number_input('Kilometers Driven', min_value= 100, max_value = 3800000),
seller_type = st.selectbox('Seller Type', options=['Individual', 'Dealer', 'Trustmark Dealer'], index=None, placeholder='Select the seller type..'),
fuel_type = st.selectbox('Fuel Type', options=['CNG', 'Diesel', 'Electric', 'LPG', 'Petrol'], index=None, placeholder='Select the fuel type...'),
transmision_type = st.selectbox('Transmission Type', options=['Automatic', 'Manual'], index=None, placeholder='Select the transmission type...'),
mileage = st.number_input("Mileage (km/l)", min_value= 4.0, max_value = 33.54),
engine = st.number_input("Engine (in cc)", min_value= 793, max_value = 6592),
max_power = st.number_input("Maximum Power (in bhp)", min_value= 38.4, max_value = 626.5),
seats = st.number_input("Number of Seats", min_value= 0, max_value = 9)


# ----------------------------------------------
# Create Input DataFrame
# --------------------------------------------
input_data = {
              'vehicle_age': vehicle_age,
              'km_driven': km_driven,
              'seller_type': seller_type,
              'fuel_type': fuel_type,
              'transmision_type': transmision_type,
              'car_name': car_name,        
              'mileage': mileage,
              'engine': engine,
              'max_power':max_power,
              'seats': seats
}
df_input = pd.DataFrame([input_data])

#Add categorical columns
df_input['seller_type'] = seller_type
df_input['fuel_type'] = fuel_type
df_input['transmision_type'] = transmision_type
df_input['car_name'] = car_name


#One-Hot Encoding
df_input = pd.get_dummies(df_input)

#Align with training columns
df_input = df_input.reindex(columns=feature_columns, fill_value=0)

#-----------------------------------------------------------
# Scaling
#-----------------------------------------------
df_input_scaled = scaler.transform(df_input)

#------------------------------------------------------
# Prediction
#---------------------------------------------------------
if st.button('Predict Selling Price'):
   Prediction = rf_model.predict(df_input_scaled)[0]

   if Prediction <= 0:
      st.error("Invalid Prediction. Please review inputs.")
   else:
       #price_rupees = int(Prediction)
       price_lakh = Prediction / 100000

       col1, col2 = st.columns(2)
       with col1:  
            st.metric(label='Price', value= f'{int(Prediction):,}') 
 
       with col2:
            st.metric(label='Price (Lakh)', value=f'{price_lakh:.2f} L')
            
            st.success(f'The predicted price of the car is {Prediction:,.2f}')


 # Footer
st.markdown("---")
st.markdown(""" <div style= "text-align: center; color: grey; font-size: 0.9rem;">
            <p>. Indian Car Price Prediction System</b><br> Built using <b> Random Forest Regression</b> trained on
            the <b> CarDekho Dataset</b><br><br><i>This prediction is an estimate based on historical data and should not be considered as a final              market price.</i><br><br>
            2026 | Developed by <b> Isaac</b> | Machine Learning project
            </div>
            """, unsafe_allow_html=True)
  
