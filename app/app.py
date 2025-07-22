# Linear Regression App for House Price Prediction
import streamlit as st
import pandas as pd
import joblib

# Load pipeline
model_pipeline = joblib.load('D:/ML Projects/ML Portfolio/House-Price-Prediction/models/linreg_model.pkl')

# Title
st.title("House Price Prediction")

st.write("""
Provide a few details about the house to estimate its **Sale Price**.
""")

# User inputs
st.header("House Details")

OverallQual = st.slider("Overall Quality (1 = Low, 10 = High)", 1, 10, 5)
GrLivArea = st.number_input("Living Area (in sq ft)", min_value=300, max_value=5000, value=1500)
GarageCars = st.slider("Garage Capacity (Number of Cars)", 0, 4, 1)
TotalBsmtSF = st.number_input("Basement Area (in sq ft)", min_value=0, max_value=3000, value=800)
FullBath = st.slider("Number of Full Bathrooms", 0, 4, 1)
YearBuilt = st.number_input("Year Built", min_value=1900, max_value=2025, value=2000)

Neighborhood = st.selectbox("Neighborhood", 
    ['Names', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt', 'Sawyer', 'NWAmes', 'BrkSide', 'Other'])

# Button to predict
if st.button("Predict Sale Price"):
    # Construct input with selected features
    input_dict = {
        'OverallQual': OverallQual,
        'GrLivArea': GrLivArea,
        'GarageCars': GarageCars,
        'TotalBsmtSF': TotalBsmtSF,
        'FullBath': FullBath,
        'YearBuilt': YearBuilt,
        'Neighborhood': Neighborhood
    }

    input_df = pd.DataFrame([input_dict])

    # Fill in remaining features with median/mode during pipeline transformation
    predicted_price = model_pipeline.predict(input_df)[0]

    st.success(f"**Estimated Sale Price: ${predicted_price:,.0f}**")
