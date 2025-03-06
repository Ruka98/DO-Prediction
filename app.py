import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import os

# Define the custom loss function
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# Load the trained model with custom objects handling
@st.cache_resource
def load_model():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the .h5 file
    model_path = os.path.join(script_dir, "trained_lstm_model.h5")
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={"mse": mse})
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Feature preprocessing objects (must match training pipeline)
@st.cache_resource
def load_preprocessing():
    time_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    scaler_X_num = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Dummy fit (Replace with actual fitted objects)
    time_categories = np.array(["Early_morning", "Morning", "Evening", "Night"]).reshape(-1, 1)
    time_encoder.fit(time_categories)
    scaler_X_num.fit(np.random.rand(10, 6))  # Replace with actual training data shape
    scaler_y.fit(np.random.rand(10, 1))     # Replace with actual training data shape
    return time_encoder, scaler_X_num, scaler_y

time_encoder, scaler_X_num, scaler_y = load_preprocessing()

# Month mapping
month_mapping = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
}

# Streamlit UI
st.title("DO Prediction Model")

# Input widgets
time = st.selectbox(
    "Time of Day",
    options=["Early_morning", "Morning", "Evening", "Night"]
)

month = st.selectbox(
    "Month",
    options=list(month_mapping.keys())
)

temp = st.number_input("Temperature (°C)", format="%.2f")
ec = st.number_input("EC (µS/cm)", format="%.2f")
tds = st.number_input("TDS (mg/L)", format="%.2f")
ph = st.number_input("pH", format="%.2f")
salinity = st.number_input("Salinity (psu)", format="%.2f")

# Prediction button
if st.button("Predict DO"):
    try:
        # Process inputs
        month_num = month_mapping[month]
        
        # One-hot encode time
        time_encoded = time_encoder.transform([[time]])
        
        # Scale numerical features
        numerical_values = np.array([[month_num, temp, ec, tds, ph, salinity]])
        numerical_scaled = scaler_X_num.transform(numerical_values)
        
        # Combine features
        X_processed = np.concatenate([time_encoded, numerical_scaled], axis=1)
        X_reshaped = X_processed.reshape((1, 1, X_processed.shape[1]))
        
        # Make prediction
        y_pred_scaled = model.predict(X_reshaped)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
        # Display result
        st.success(f"Predicted DO: {y_pred[0][0]:.4f} ppm")
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")