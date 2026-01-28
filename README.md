# 🚗 Predicting Selling Price of Used Cars in the Indian Market
A machine learning regression project for predicting used car prices in the Indian market.

## Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Description](#-dataset-description)
- [Data Preprocessing](#-data-preprocessing)
- [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [Machine Learning Models Used](#-machine-learning-models-used)
- [Model Evaluation](#-model-evaluation)
- [Model Deployment Decision](#-model-deployment-decision)
- [Streamlit Web Application](#-streamlit-web-application)
- [How to Run the Project](#-how-to-run-the-project)
- [Live App](#-live-app)
- [Application Features](#-application-features)
- [Tech Stack](#-tech-stack)
- [Future Improvements](#-future-improvements)
- [Author](#-author)


# 📌 Project Overview
The objective of this project is to build an end-to-end machine learning solution that predicts the selling price of used cars in the Indian market based on vehicle specifications and seller-related features. In addition to model development and evaluation, a Streamlit web application was built to deploy the trained model and provide an interactive, user-friendly interface for real-time price prediction. This solution can assist:
- Buyers in evaluating fair market prices.
- Sellers in pricing vehicles competitively.
- Online automobile platforms in automating price estimation workflows.

The project demonstrates the complete machine learning lifecycle, including data preprocessing, exploratory data analysis, model training, evaluation, and deployment.

# 📊 Dataset Description
* Source: https://www.kaggle.com/datasets/manishkr1754/cardekho-used-car-data

* The dataset contains 15,411 records, 13 input features, and 1 target variable.

* Categorical Features: *car_name*, *brand*, *model*, *seller_type*, *fuel_type*, *transmission_type*

 * Numerical Features:  *mileage*, *engine*, *max_power*, *seats*, *vehicle_age*, *km_driven*

* Target Variable: *selling_price*

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
### EDA Approach
- **Descriptive Statistics:** Summarized central tendency and dispersion for numerical features.
- **Data Visualization:** Used histograms, box plots, scatter plots, and bar charts to identify trends and anomalies.

### Key Insights
- Engine size and max power show strong positive correlation with selling price.
- Vehicle age and mileage are negatively correlated with selling price.
- Luxury brands (Ferrari, Rolls-Royce, Bentley) command the highest prices.
- Electric vehicles show higher average prices compared to petrol and diesel.
- Dealer-listed cars tend to have higher prices than individual sellers.

### EDA Techniques Used
- Correlation heatmap.
- Fuel type, transmission type, and seller type analysis.

  ![image](https://github.com/Isaac5886/Predicting-Selling-Price-of-Cars-in-the-Indian-Market-
/blog/main/app1.png)

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

### Final Model
- **Best Model:** Random Forest Regressor

- **R2 Score:** 0.934961

- **RMSE:** 221269.461390

- **MAE:** 100074.031143

  **Note:** Random Forest achieved the best offline performance, while Gradient Boosting was deployed for production.

# 🚀 Model Deployment Decision
Although Random Forest Regressor achieved the best performance, it was not deployed due to large model size (~ 95MB), which exceeds GitHub hosting limits. Gradient Boosting Regressor was selected due to its smaller model size and suitability for deployment.

# 🌐 Streamlit Web Application

The trained Gradient Boosting Regressor was deployed as an interactive web application using **Streamlit**.

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

### 🔗 Live App
👉 https://indian-app.streamlit.app

### 📌 Application Features
- User-friendly interface for entering car specifications.
- Real-time selling price prediction.
- Supports multiple car attributes such as fuel type, transmission, mileage, engine power, and vehicle age.
- Lightweight and fast inference suitable for web deployment.

# 🛠️ Tech Stack
* Python
* Pandas, NumPy, Matplotlib & Seaborn
* Scikit-learn
* XGBoost
* Streamlit (for web app deployment)
* Jupyter Notebook
* Pickle (model saving)


## 📌 Future Improvements
- Perform hyperparameter tuning to improve regression performance (R², RMSE, MAE).
- Apply model compression techniques to enable Random Forest Regressor deployment.
- Integrate live car listing APIs for real-time price estimation.
- Add support for multiple currencies.
- Provide confidence intervals for predicted prices.
- Deploy the Streamlit application on AWS EC2 for improved scalability and reliability.
- Store trained models in AWS S3 for version control and persistent storage.

# 👤 Author
Agboola Isaacoluwatomiwa
Machine Learning / Data Science Enthusiast
