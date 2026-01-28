import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Indian Car Price Predictor | ML-Powered Valuation",
    page_icon="🚗",
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #145a8d;
    }
    .info-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load Model Artifacts with Error Handling
# -------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    """Load all model artifacts with error handling"""
    try:
        artifacts = {}
        artifact_files = {
            'model': 'gb_model.pkl',
            'scaler': 'scaler.pkl',
            'features': 'feature_columns.pkl',
            'metrics': 'model_metrics.pkl'
        }
        
        for key, filename in artifact_files.items():
            file_path = Path(filename)
            if not file_path.exists():
                st.error(f"❌ Required file not found: {filename}")
                st.stop()
            
            with open(file_path, 'rb') as file:
                artifacts[key] = pickle.load(file)
        
        logger.info("All model artifacts loaded successfully")
        return artifacts
    
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

# Load artifacts
artifacts = load_model_artifacts()
gb_model = artifacts['model']
scaler = artifacts['scaler']
feature_columns = artifacts['features']
metrics = artifacts['metrics']

# Extract metrics
r2 = metrics.get('r2', 0)
mae = metrics.get('mae', 0)
rmse = metrics.get('rmse', 0)

# -------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/car--v1.png", width=80)
    st.title("📊 Model Information")
    
    st.markdown("### About This App")
    st.info(
        """
        This application uses a **Gradient Boosting Regressor** model 
        trained on over **15,000 vehicles** records from the CarDekho dataset to 
        predict used car prices in the Indian market.
        """
    )
    
    st.markdown("### 🎯 Model Performance")
    st.metric("R² Score", f"{r2:,.3f}")
    st.metric("MAE", f"₹{mae:,.0f}TND")
    st.metric("RMSE", f"₹{rmse:,.0f}TND")
    
    st.markdown("---")
    
    st.markdown("### 📈 Key Features")
    st.markdown("""
    - Real-time price prediction
    - 100+ car models supported
    - High accuracy (R² > 0.90)
    - Instant results
    """)
    
    st.markdown("---")
    st.markdown("### 🔒 Data Privacy")
    st.caption("Your input data is not stored and is only used for prediction purposes.")

# -------------------------------------------------
# Main Application Header
# -------------------------------------------------
st.markdown('<h1 class="main-header">🚗 Indian Car Price Predictor</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Get instant, AI-powered valuations for used cars in the Indian market</p>', 
    unsafe_allow_html=True
)

# -------------------------------------------------
# Information Banner
# -------------------------------------------------
st.markdown("""
    <div class="info-card">
        <strong>ℹ️ How it works:</strong> Enter your car's details below, and our machine learning model 
        will predict its market value based on 15,411 historical sales data points.
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------
# User Input Section
# -------------------------------------------------
st.subheader("📝 Enter Vehicle Details")

# Create organized input layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Basic Information**")
    car_name = st.selectbox(
        'Car Model:',
        options=[
            'Audi A4', 'Audi A6', 'Audi A8', 'Audi Q7', 'Bentley Continental', 
            'BMW 3', 'BMW 5', 'BMW 6', 'BMW 7', 'BMW X1', 'BMW X3', 'BMW X4', 'BMW X5', 'BMW Z4',
            'Datsun GO', 'Datsun RediGO', 'Datsun redi-GO', 'Ferrari GTC4Lusso', 
            'Force Gurkha', 'Ford Aspire', 'Ford Ecosport', 'Ford Endeavour', 'Ford Figo', 'Ford Freestyle',
            'Honda Amaze', 'Honda City', 'Honda Civic', 'Honda CR', 'Honda CR-V', 'Honda Jazz', 'Honda WR-V',
            'Hyundai Aura', 'Hyundai Creta', 'Hyundai Elantra', 'Hyundai Grand', 'Hyundai i10', 
            'Hyundai i20', 'Hyundai Santro', 'Hyundai Tucson', 'Hyundai Venue', 'Hyundai Verne',
            'Isuzu D-Max', 'ISUZU MUX', 'Jaguar F-PACE', 'Jaguar XE', 'Jaguar XF',
            'Jeep Compass', 'Jeep Wrangler', 'Kia Carnival', 'Kia Seltos',
            'Land Rover Rover', 'Lexus ES', 'Lexus NX', 'Lexus RX',
            'Mahindra Alturas', 'Mahindra Bolero', 'Mahindra KUV', 'Mahindra KUV100', 
            'Mahindra Murazzo', 'Mahindra Scorpio', 'Mahindra Thar', 'Mahindra XUV300', 'Mahindra XUV500',
            'Maruti Alto', 'Maruti Baleno', 'Maruti Celerio', 'Maruti Ciaz', 
            'Maruti Dzire LXI', 'Maruti  Dzire VXI', 'Maruti Dzire ZXI', 'Maruti Eeco', 'Maruti Ertiga',
            'Maruti Ignis', 'Maruti S-Presso', 'Maruti Swift', 'Maruti Swift Dzire', 'Maruti Vitara',
            'Maruti Wagon R', 'Maruti XL6', 'Maserati Ghibli', 'Maserati Quattroporte',
            'Mercedes-AMG C', 'Mercedes-Benz C-Class', 'Mercedes-Benz GL-Class', 
            'Mercedes-Benz GLS', 'Mercedes-Benz S-Class',
            'MG Hector', 'Mini Cooper', 'Nissan Kicks', 'Nissan X-Trail',
            'Porsche Cayenne', 'Porsche Macan', 'Porsche Panamera',
            'Renault Duster', 'Renault KWID', 'Renault Triber', 'Rolls-Royce Ghost',
            'Skoda Octavia', 'Skoda Rapid', 'Skoda Superb',
            'Tata Altroz', 'Tata Harrier', 'Tata Hexa', 'Tata Nexon', 'Tata Safari', 'Tata Tiago', 'Tata Tigor',
            'Toyota Camry', 'Toyota Fortuner', 'Toyota Glanza', 'Toyota Innova', 'Toyota Yaris',
            'Volkswagen Polo', 'Volkswagen Vento', 'Vol S90', 'Volo XC', 'Volo XC60', 'Volo XC90'
        ],
        index=None,
        placeholder='Select car model...',
    )
    
    vehicle_age = st.number_input(
        'Vehicle Age (years):',
        min_value=0,
        max_value=25,
        value=3,
    )
    
    km_driven = st.number_input(
        'Kilometers Driven:',
        min_value=100,
        max_value=3800000,
        value=50000,
        step=1000,
    )
    
    seller_type = st.selectbox(
        'Seller Type:',
        options=['Individual', 'Dealer', 'Trustmark Dealer'],
        index=None,
        placeholder='Select seller type...',
    )

with col2:
    st.markdown("**Technical Specifications**")
    fuel_type = st.selectbox(
        'Fuel Type:',
        options=['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'],
        index=None,
        placeholder='Select fuel type...',
    )
    
    transmision_type = st.selectbox(
        'Transmission Type:',
        options=['Manual', 'Automatic'],
        index=None,
        placeholder='Select transmission...',
    )
    
    mileage = st.number_input(
        "Mileage (km/l):",
        min_value=4.0,
        max_value=35.0,
        value=15.0,
        step=0.5,
    )
    
    engine = st.number_input(
        "Engine Capacity (cc):",
        min_value=600,
        max_value=7000,
        value=1200,
        step=100,
    )

with col3:
    st.markdown("**Performance & Comfort**")
    max_power = st.number_input(
        "Maximum Power (bhp):",
        min_value=30.0,
        max_value=650.0,
        value=80.0,
        step=5.0,
    )
    
    seats = st.number_input(
        "Number of Seats:",
        min_value=2,
        max_value=10,
        value=5,
    )

st.markdown("---")

# -------------------------------------------------
# Prediction Logic
# -------------------------------------------------
def validate_inputs():
    """Validate all required inputs"""
    missing_fields = []
    if car_name is None:
        missing_fields.append("Car Model")
    if seller_type is None:
        missing_fields.append("Seller Type")
    if fuel_type is None:
        missing_fields.append("Fuel Type")
    if transmision_type is None:
        missing_fields.append("Transmission Type")
    
    return missing_fields

def prepare_input_data():
    """Prepare input data for prediction"""
    try:
        input_data = {
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
        }
        
        df_input = pd.DataFrame([input_data])
        df_input = pd.get_dummies(df_input)
        df_input = df_input.reindex(columns=feature_columns, fill_value=0)
        
        return df_input
    
    except Exception as e:
        logger.error(f"Error preparing input data: {str(e)}")
        raise

# Prediction Button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button('🔮 Predict Selling Price', type="primary")

if predict_button:
    # Validate inputs
    missing_fields = validate_inputs()
    
    if missing_fields:
        st.error(f"⚠️ Please fill in the following required fields: {', '.join(missing_fields)}")
    else:
        try:
            with st.spinner('🔄 Analyzing vehicle data...'):
                # Prepare and scale input
                df_input = prepare_input_data()
                df_input_scaled = scaler.transform(df_input)
                
                # Make prediction
                prediction = gb_model.predict(df_input_scaled)[0]
                
                if prediction <= 0:
                    st.error("❌ Invalid prediction. Please review your inputs and try again.")
                else:
                    price_lakh = prediction / 100000
                    
                    st.success("✅ Prediction completed successfully!")
                    st.markdown("---")
                    
                    # Display results in attractive cards
                    st.subheader("💰 Predicted Price")
                    
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        st.markdown(f"""
                            <div class="metric-container">
                                <h4 style="margin:0;">Price (INR)</h4>
                                <h2 style="margin:0.5rem 0;">₹{int(prediction):,}</h2>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with result_col2:
                        st.markdown(f"""
                            <div class="metric-container">
                                <h4 style="margin:0;">Price (Lakhs)</h4>
                                <h2 style="margin:0.5rem 0;">₹{price_lakh:.2f} L</h2>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with result_col3:
                        # Calculate price range (±10%)
                        lower_bound = prediction * 0.9
                        upper_bound = prediction * 1.1
                        st.markdown(f"""
                            <div class="metric-container">
                                <h4 style="margin:0;">Price Range</h4>
                                <h2 style="margin:0.5rem 0; font-size:1.3rem;">₹{int(lower_bound/100000):.1f}L - ₹{int(upper_bound/100000):.1f}L</h2>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Additional insights
                    st.markdown("### 📊 Insights")
                    insight_col1, insight_col2 = st.columns(2)
                    
                    with insight_col1:
                        st.info(f"""
                        **Price per kilometer:** ₹{(prediction/km_driven):.2f}  
                        **Price per year:** ₹{(prediction/(vehicle_age if vehicle_age > 0 else 1)/100000):.2f} Lakhs
                        """)
                    
                    with insight_col2:
                        st.warning("""
                        **Disclaimer:** This is an estimated price based on historical data. 
                        Actual market prices may vary based on condition, location, and demand.
                        """)
                    
                    # Log prediction
                    logger.info(f"Prediction made: {car_name} - ₹{int(prediction):,}")
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            st.error(f"❌ An error occurred during prediction: {str(e)}")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <h4 style="color: #1f77b4;">Indian Car Price Prediction System</h4>
        <p>Powered by <strong>Gradient Boosting Regressor</strong> | Trained on <strong>CarDekho Dataset</strong></p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">
            <em>This estimate is intended for informational purposes and should not be used as a sole basis for financial decisions.</em>
        </p>
        <p style="margin-top: 1.5rem; font-size: 0.85rem;">
            © {datetime.now().year} | Developed by <strong>Agboola Isaac</strong> | Machine Learning Project.
        </p>
    </div>
""", unsafe_allow_html=True)
