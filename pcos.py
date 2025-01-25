from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model
try:
    with open('pcos_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise Exception("The 'pcos_rf_model.pkl' file was not found. Please ensure it exists in the same directory as this script.")

@app.route('/')
def home():
    return "Welcome to the PCOS Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract JSON data from the request
        data = request.get_json()

        # Define required features
        required_features = ["Age", "BMI", "Cycle length (days)", "Follicle no. (L)", "Follicle no. (R)", "Endometrium (mm)"]

        # Check if all required features are provided
        if not all(feature in data for feature in required_features):
            missing_features = [feature for feature in required_features if feature not in data]
            return jsonify({"error": f"Missing required features: {', '.join(missing_features)}"}), 400

        # Prepare input features for prediction
        input_features = np.array([data[feature] for feature in required_features]).reshape(1, -1)

        # Make prediction using the model
        prediction = model.predict(input_features)

        # Return the prediction as JSON
        result = "PCOS Detected" if prediction[0] == 1 else "No PCOS Detected"
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
