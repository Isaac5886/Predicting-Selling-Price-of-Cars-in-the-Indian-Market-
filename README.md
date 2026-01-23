🚗 Predicting Selling Price of Cars in the Indian Market
📌 Project Overview
The objective of this project is to build a machine learning model that predicts the selling price of used cars in the Indian market based on vehicle specifications and seller-related features.
This solution can help buyers, sellers, and online automobile platforms make informed pricing decisions.
🏢 Business Problem
In the Indian used-car market, pricing vehicles accurately is challenging due to factors such as:
Vehicle age and usage
Fuel type and transmission
Seller type (dealer vs individual)
Vehicle performance attributes
Incorrect pricing can lead to:
Overpricing → reduced sales
Underpricing → financial loss
This project aims to automate price estimation using machine learning.
📊 Dataset Description
The dataset contains 15,411 records with the following features:
Feature
Description
Brand
Manufacturer of the car
Model
Car model
Seller Type
Individual / Dealer
Fuel Type
Petrol / Diesel / CNG
Transmission
Manual / Automatic
Km Driven
Total distance driven
Mileage
Fuel efficiency
Seats
Seating capacity
Vehicle Age
Age of the car
Engine
Engine capacity
Max Power
Engine power
Selling Price
Target variable
🎯 Target Variable
Selling Price (Continuous numerical value)
🧹 Data Preprocessing
The following preprocessing steps were performed:
Dropped Car Name to avoid high-cardinality noise
Created new features:
Year (derived from vehicle age)
Energy Consumption (derived from engine and mileage)
Removed original features after transformation:
Vehicle Age
Engine
Max Power
Label encoding for categorical variables
Feature scaling using StandardScaler
📈 Exploratory Data Analysis (EDA)
Key visualizations include:
Brand vs Selling Price
Fuel Type vs Selling Price
Transmission vs Selling Price
Seller Type vs Selling Price
Actual vs Predicted Price (Scatter Plot)
🤖 Model Development
Multiple regression models were trained and evaluated:
Linear Regression
Lasso Regression
ElasticNet
Gradient Boosting Regressor
Random Forest Regressor
All models were evaluated using a held-out test set.
📐 Model Evaluation Metrics
The following metrics were used:
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
Mean Squared Error (MSE)
R² Score
🏆 Model Selection
After comparing overall performance across models, the best-performing model was selected based on:
Highest R² score
Lowest RMSE
Only this final model was used for deployment.
📊 Model Performance Summary
A comparison table was created to evaluate overall performance across models.
(You can attach a screenshot or CSV of the results table here)
🌐 Web Application (Streamlit)
A Streamlit web application was developed to allow users to:
Enter car details
Predict the selling price instantly
App Features:
User-friendly UI
Real-time prediction
Uses only the final trained model
☁️ Deployment
The Streamlit application is designed to be deployed on AWS EC2
The trained model is loaded using joblib
No model training occurs during deployment
📁 Project Structure
Copy code

car-price-prediction/
│
├── data/
│   └── car_data.csv
│
├── notebooks/
│   ├── 01_model_experiments.ipynb
│   └── 02_best_model.ipynb
│
├── models/
│   └── best_model.pkl
│
├── app.py
├── requirements.txt
└── README.md
🛠️ Technologies Used
Python
Pandas, NumPy
Scikit-learn
Matplotlib, Seaborn
Streamlit
AWS EC2 (optional)
🚀 Future Improvements
Hyperparameter tuning
Advanced feature engineering
Model explainability (SHAP)
CI/CD pipeline for deployment
Docker containerization
👤 Author
Your Name
Aspiring Data Scientist / Machine Learning Engineer
