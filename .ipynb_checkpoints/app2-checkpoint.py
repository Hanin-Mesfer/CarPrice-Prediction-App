import streamlit as st
import pickle
import numpy as np

# Load the trained model and encoders
model = pickle.load(open('best_model.pkl', 'rb'))
car_maker_encoder = pickle.load(open('car_maker_encoder.pkl', 'rb'))
model_encoder = pickle.load(open('model_encoder.pkl', 'rb'))

st.title("🚗 Car Price Prediction App")

# Dropdown options from encoders
car_maker_options = list(car_maker_encoder.classes_)
model_options = list(model_encoder.classes_)

# User input fields
car_maker = st.selectbox("Select Car Maker", car_maker_options)
model_name = st.selectbox("Select Model Name", model_options)
year = st.number_input("Select Year", min_value=1990, max_value=2025, value=2020)
condition = st.selectbox("Select Condition", ["New", "Used"])
kilometers = st.number_input("Enter Kilometers Driven", min_value=0, max_value=500000, value=50000)
transmission = st.selectbox("Select Transmission Type", ["Automatic", "Manual"])

# Prediction button
if st.button("Predict Price"):
    try:
        # Encode categorical inputs
        maker_encoded = car_maker_encoder.transform([car_maker])[0]
        model_encoded = model_encoder.transform([model_name])[0]
        condition_encoded = 1 if condition == "Used" else 0
        transmission_encoded = 1 if transmission == "Automatic" else 0

        # Prepare input array
        inputs = np.array([[maker_encoded, model_encoded, year, condition_encoded, kilometers, transmission_encoded]])

        # Make prediction
        predicted_price = model.predict(inputs)[0]

        #st.success(f"💰 Car Price: {predicted_price:.2f} ريال ")
        st.markdown(
                f"""
                <style>
                    @keyframes fadeIn {{
                        0% {{opacity: 0;}}
                        100% {{opacity: 1;}}
                    }}s
                    .fade-in {{
                        animation: fadeIn 2s ease-in;
                    }}
                </style>
                <div class="fade-in">
                    <h2>💰 Car Price: {predicted_price:.2f} ريال</h2>
                </div>
                """, unsafe_allow_html=True)
    
    

    except Exception as e:
        st.error("❌ Something went wrong with the prediction.")
