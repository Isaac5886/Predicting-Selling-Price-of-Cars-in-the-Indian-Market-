# 🚗 Predicting Selling Price of Cars in the Indian Market
A machine learning regression project for predicting used car prices in the Indian market.

## Table of Contents
- Project Overview
- Dataset Description
- Data Preprocessing
- Exploratory Data Analysis
- Machine Learning Models Used
- Model Evaluation
- Model Deployment Decision
- Streamlit Web Application
- Tech Stack
- How to Run the Project
- Future Improvements
- Author


# 📌 Project Overview
The objective of this project is to build a machine learning model that predicts the selling price of used cars in the Indian market based on vehicle specifications and seller-related features. This solution can assist buyers in evaluating fair prices, sellers in pricing vehicles competitively, and online automobile platforms in automating price estimation.

# 📊 Dataset Description
* Source: https://www.kaggle.com/datasets/manishkr1754/cardekho-used-car-data

* The dataset contains 15,411 records, 13 input features, and 1 target variable.

* Categorical Features: *car_name*, *brand*, *model*, *seller_type*, *fuel_type*, *transmission_type*

 * Numerical Features:  *mileage*, *engine*, *max_power*, *seats*, *vehicle_age*, *km_driven*

* Target Variables: *selling_price*

# 🧹 Data Preprocessing

1. Missing Value Treatment:
- Checked for missing values across all features.
- No missing values were found; therefore, no imputation was required.

2.  Outlier Analysis:
- Performed descriptive statistical analysis to understand feature distributions.

3. Encoding:
- One-Hot Encoding applied to categorical features using pd.get_dummies().
- Encoded features were converted to integer format for model compatibility.

4. Feature Engineering:
- Removed non-informative index column (Column1).
- Dropped redundant features (brand, model) to reduce multicollinearity.
- Retained relevant numerical and categorical predictors.

5. Feature Scaling:
- Applied StandardScaler to numerical features.
- Scaling was performed after train-test split to prevent data leakage.

6. Train-Test Split:
- 80% training and 20% testing split.
- Fixed random state used for reproducibility.

# 🔍 Exploratory Data Analysis (EDA)
* Engine size and max power show strong positive correlation with selling price.
* Vehicle age and mileage are negatively correlated with selling price.
* Luxury brands (Ferrari, Rolls-Royce, Bentley) command the highest prices.
* Electric vehicles show higher average prices compared to petrol and diesel.
* Dealer-listed cars tend to have higher prices than individual sellers.

# EDA techniques used:
* Correlation heatmap.
* Brand-wise and Model-wise price comparison.
* Fuel type, transmission type, and seller type analysis.

# 🤖 Machine Learning Models Used
* ElasticNet Regression
* Lasso Regression
* XGBoost Regressor
* Gradient Boosting Regressor
* Random Forest Regressor

# 📈 Model Evaluation

### Model Performance Comparison
  
| Model	| MAE	| RMSE	| R2 |
|-------|-----|-------|----|
| Elasticnet	| 190564.622050	| 448233.364855 | 	0.733106 |
|	Lasso	| 180678.127175 |	405365.979320	| 0.781714 |
|	XGBRegressor |	100771.484375	| 249650.981460	| 0.917206 |
|	Gradient Boosting	| 129058.187362	| 247909.754467 | 	0.918357
|	**Random Forest Regressor**|	**100074.031143**	| **221269.461390**	|**0.934961**⭐|

### Final model
* Selected Model: Random Forest Regressor
* Reason: Highest R2 and lowest MAE/RMSE across all evaluations

# 🚀 Model Deployment Decision
Although Random Forest Regressor achieved the best performance, it was not deployed due to large model size (~ 95MB), which exceeds GitHub hosting limits.

# ✅ Gradient Boosting Regressor was selected for deployment because:
* High predictive performance
* Smaller model size
* Suitable for Streamlit deployment

  ## 🌐 Web Application (Streamlit)

The trained Gradient Boosting Regressor was deployed as an interactive web application using **Streamlit**.

### 🔗 Live App
👉 https://indian-app.streamlit.app

### 📌 Application Features
- User-friendly interface for entering car specifications
- Real-time selling price prediction
- Supports multiple car attributes such as fuel type, transmission, mileage, engine power, and vehicle age
- Lightweight and fast inference suitable for web deployment

# 🛠️ Tech Stack
* Python
* Pandas, NumPy, Matplotlib & Seaborn
* Scikit-learn
* XGBoost
* Streamlit (for web app deployment)
* Jupyter Notebook
* Pickle (model saving)

# 🧪 How to Run the Project

1. Install required packages:
```Bash
pip install -r requirements.txt
```
2. Run the Streamlit app:

 ````Bash
streamlit run app.py
````
3. Access the Web Application:

 ````Bash
http://localhost:8501
````

## 📌 Future Improvements
- Perform hyperparameter tuning to improve regression performance (R², RMSE, MAE).
- Apply model compression techniques to enable Gradient Boosting Regressor deployment.
- Integrate live car listing APIs for real-time price estimation.
- Add support for multiple currencies.
- Provide confidence intervals for predicted prices.
- Deploy the Streamlit application on AWS EC2 for improved scalability and reliability.
- Store trained models in AWS S3 for version control and persistent storage.

# 👤 Author
Agboola Isaacoluwatomiwa
Machine Learning / Data Science Enthusiast
