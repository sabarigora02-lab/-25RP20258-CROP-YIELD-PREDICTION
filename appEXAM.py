import streamlit as st
import pickle
import numpy as np
import os

# Path to the saved model
model_path = "25RP20258.pkl"

st.title("🌱 Crop Yield Prediction Web App")
st.write("Predict crop yield based on temperature")

# Check if the model file exists
if os.path.exists(model_path):
    # Load the best-performing model
    model = pickle.load(open(model_path, 'rb'))

    # User input for temperature
    temperature = st.number_input(
        "Enter Average Temperature (°C)",
        min_value=0.0,
        step=0.1,
        value=25.0
    )

    if st.button("Predict Crop Yield"):
        # Make prediction
        predicted_yield = model.predict(np.array([[temperature]]))
        st.success(f"Predicted Crop Yield: {predicted_yield[0]:.2f} tons/hectare")

        # Optional: Visualize regression line if Linear Regression is the best model
        if type(model).__name__ == "LinearRegression":
            import matplotlib.pyplot as plt

            # Prepare data for plotting regression line
            import pandas as pd
            Data = pd.read_csv("25RP20258.csv")
            X = Data[['Temperature']]
            y = Data['Crop_Yield']
            X_sorted = X.sort_values(by='Temperature')
            y_pred_sorted = model.predict(X_sorted)

            # Plot regression line and actual points
            plt.figure()
            plt.scatter(X, y, label="Actual Data Points")
            plt.plot(X_sorted, y_pred_sorted, color='red', label="Regression Line")
            plt.xlabel("Temperature (°C)")
            plt.ylabel("Crop Yield (tons/hectare)")
            plt.title("Regression Line vs Actual Data")
            plt.legend()
            st.pyplot(plt)
else:
    st.error(f"Model file '{model_path}' not found. Please ensure the model is in the app folder.")
