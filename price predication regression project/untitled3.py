import streamlit as st
import pandas as pd
import datetime

# --- App Title and Description ---
st.set_page_config(page_title="Avocado Price Predictor", layout="centered")

st.title("🥑 Avocado Price Predictor")
st.markdown("""
    Enter the details below to get a simulated prediction for the average price of an avocado.
    This app demonstrates how you might build a frontend for a machine learning model.
""")

# --- Input Widgets ---

st.header("Input Features")

# Date Input
date_input = st.date_input("Date", datetime.date(2018, 1, 1))

# Type Selectbox
type_options = ["conventional", "organic"]
selected_type = st.selectbox("Type", type_options)

# Year Number Input (restricted to relevant range from dataset)
year_input = st.number_input("Year", min_value=2015, max_value=2020, value=2018, step=1)

# Region Selectbox (populated with example regions from your notebook's dataset)
region_options = [
    "Albany", "Atlanta", "BaltimoreWashington", "Boise", "Boston", "California",
    "Charlotte", "Chicago", "CincinnatiDayton", "Columbus", "DallasFtWorth",
    "Denver", "Detroit", "GrandRapids", "GreatLakes", "HarrisburgScranton",
    "HartfordSpringfield", "Houston", "Indianapolis", "Jacksonville", "LasVegas",
    "LosAngeles", "Louisville", "MiamiFtLauderdale", "Midsouth", "Nashville",
    "NewOrleansMobile", "NewYork", "Northeast", "NorthernNewEngland", "Orlando",
    "Philadelphia", "PhoenixTucson", "Pittsburgh", "Plains", "Portland",
    "RaleighGreensboro", "RichmondNorfolk", "Roanoke", "Sacramento", "SanDiego",
    "SanFrancisco", "Seattle", "SouthCarolina", "SouthCentral", "Southeast",
    "Spokane", "StLouis", "Syracuse", "Tampa", "TotalUS", "West", "WestTexNewMexico"
]
selected_region = st.selectbox("Region", region_options)

# Volume Inputs
st.subheader("Volume Distribution (in units)")
total_volume = st.number_input("Total Volume", min_value=0.0, value=100000.0, step=1000.0)
plu_4046 = st.number_input("PLU 4046 Volume", min_value=0.0, value=50000.0, step=1000.0)
plu_4225 = st.number_input("PLU 4225 Volume", min_value=0.0, value=30000.0, step=1000.0)
plu_4770 = st.number_input("PLU 4770 Volume", min_value=0.0, value=20000.0, step=1000.0)

# --- Prediction Button ---
if st.button("Predict Price"):
    # Basic input validation (Streamlit's number_input handles some of this)
    if total_volume < (plu_4046 + plu_4225 + plu_4770):
        st.warning("Warning: Total Volume should ideally be greater than or equal to the sum of PLU volumes.")

    # In a real application, you would pass these inputs to your trained model
    # and get a real prediction. For this demo, we'll simulate it.

    # Prepare input data (this would be feature engineering for your model)
    # For a real model, you'd likely one-hot encode 'type' and 'region',
    # and extract features like 'month', 'day_of_week' from 'date'.
    input_data = {
        "Date": date_input.strftime("%Y-%m-%d"),
        "Type": selected_type,
        "Year": year_input,
        "Region": selected_region,
        "Total Volume": total_volume,
        "PLU 4046": plu_4046,
        "PLU 4225": plu_4225,
        "PLU 4770": plu_4770,
    }

    st.subheader("Input Data for Prediction (Simulated)")
    st.json(input_data) # Display the collected input data as JSON

    # --- Simulate Prediction ---
    # This is where your actual model prediction logic would go.
    # For demonstration, we'll generate a random price based on type.
    import random
    if selected_type == "organic":
        # Organic avocados are generally more expensive
        predicted_price = round(random.uniform(1.50, 3.50), 2)
    else:
        # Conventional avocados
        predicted_price = round(random.uniform(0.80, 2.50), 2)

    st.success(f"**Predicted Average Price: ${predicted_price:.2f}**")
    st.info("This is a simulated prediction. In a real app, this would come from your trained machine learning model.")

st.markdown("---")
st.markdown("Built with Streamlit and inspired by your avocado price prediction analysis.")
