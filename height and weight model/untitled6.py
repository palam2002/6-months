import pickle
import numpy as np
import streamlit as st

# Load the saved model from the file
filename = 'final_model.pkl'
try:
    with open(filename, 'rb') as file:
        loaded_model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Custom CSS
st.markdown(
    """
    <style>
    .title { color: #FF5733; text-align: center; font-size: 32px; }
    .text { color: #7D3C98; text-align: center; font-size: 18px; }
    .prediction { color: #6C3483; text-align: center; font-size: 24px; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True
)

# UI Title and Instruction
st.markdown('<p class="title">Weight Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="text">Enter your height in feet to predict your weight.</p>', unsafe_allow_html=True)

# Default height
default_height = 5.8
height_input = st.number_input("Enter the height in feet:", value=default_height, min_value=0.0)

# Predict button
if st.button('Predict'):
    # Reshape input for model
    height_input_2d = np.array([[height_input]])
    predicted_weight = loaded_model.predict(height_input_2d)
    predicted_value = predicted_weight.flatten()[0]

    # Show prediction
    st.markdown(f'<p class="prediction">Predicted weight: {predicted_value:.2f} kg</p>', unsafe_allow_html=True)

