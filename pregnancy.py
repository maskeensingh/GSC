from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the pre-trained XGBoost model
with open('xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Predefined scaler (adjust if specific values or steps are used in your notebook)
scaler = StandardScaler()

# Input columns (from the CSV)
INPUT_COLUMNS = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']

@app.route('/')
def home():
    # Render a simple form for user input
    return render_template('index.html', columns=INPUT_COLUMNS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        input_data = [float(request.form[col]) for col in INPUT_COLUMNS]
        input_df = pd.DataFrame([input_data], columns=INPUT_COLUMNS)

        # Example preprocessing: scaling the inputs
        input_scaled = scaler.fit_transform(input_df)  # Replace fit with transform if scaler is pre-fitted

        # Make predictions
        prediction = model.predict(input_scaled)

        # Return prediction result
        risk_mapping = {0: 'Low Risk', 1: 'Mid Risk', 2: 'High Risk'}
        risk_level = risk_mapping.get(prediction[0], 'Unknown Risk Level')
        return render_template('result.html', prediction=risk_level)
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
