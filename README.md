# 🚗 Predicting Selling Price of Cars in the Indian Market

# 📌 Project Overview
The objective of this project is to build a machine learning model that predicts the selling price of used cars in the Indian market based on vehicle specifications and seller-related features.
This solution can help buyers, sellers, and online automobile platforms make informed pricing decisions.


🚗 Predicting Selling Price of Cars in the Indian Market
📌 Project Overview
The objective of this project is to build a machine learning model that predicts the selling price of used cars in the Indian market based on vehicle specifications and seller-related attributes.
This solution can assist:
Buyers in evaluating fair prices
Sellers in pricing vehicles competitively
Online automobile platforms in automating price estimation
📊 Dataset Description
The dataset used in this project was obtained from Kaggle (CarDekho).
Dataset size: 15,411 records
Number of features: 14
Features
car_name
brand
model
vehicle_age
km_driven
seller_type
fuel_type
transmission_type
mileage
engine
max_power
seats
selling_price (Target variable)
Numerical features: 7
Categorical features: 6
🔍 Exploratory Data Analysis (EDA)
Several visualizations were performed to understand the data distribution and relationships:
Key Insights
Engine size and max power show strong positive correlation with selling price
Vehicle age and mileage are negatively correlated with selling price
Luxury brands (Ferrari, Rolls-Royce, Bentley) have the highest average prices
Electric vehicles have higher average selling prices compared to petrol and diesel
Dealer-listed cars tend to have higher prices than individual sellers
EDA techniques used:
Correlation heatmap
Brand-wise price comparison
Model-wise price analysis
Fuel type, transmission type, and seller type analysis
⚙️ Methodology
1. Data Preprocessing
Removal of irrelevant and redundant columns
Handling missing values
Conversion of categorical variables
2. Feature Engineering
Dropped non-essential features such as:
brand
model
car_name
One-hot encoding applied using pd.get_dummies()
Feature scaling using StandardScaler
🤖 Machine Learning Models Used
The following regression models were trained and evaluated:
ElasticNet Regression
Lasso Regression
XGBoost Regressor
Gradient Boosting Regressor
Random Forest Regressor
📈 Model Evaluation
Evaluation metrics used:
MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
R² Score
Performance Summary
Model
R² Score
Random Forest Regressor
⭐ Highest
Gradient Boosting Regressor
Second Best
XGBoost Regressor
Competitive
ElasticNet & Lasso
Baseline
🚀 Model Deployment Decision
Although Random Forest Regressor achieved the highest performance, it was not used for deployment because:
Model size ≈ 95MB
GitHub deployment limit ≈ 25MB
✅ Gradient Boosting Regressor was selected for deployment because:
High predictive performance
Smaller model size
Suitable for Streamlit deployment
🛠️ Tech Stack
Python
Pandas & NumPy
Matplotlib & Seaborn
Scikit-learn
XGBoost
Streamlit
Jupyter Notebook
🧪 How to Run the Project
Copy code
Bash
pip install -r requirements.txt
streamlit run app.py
📌 Future Improvements
Hyperparameter tuning for improved accuracy
Model compression techniques for Random Forest deployment
Integration with live car listing APIs
Support for multiple currencies
Addition of confidence intervals for predictions
👤 Author
Agboola Isaacoluwatomiwa
Machine Learning / Data Science Enthusiast
