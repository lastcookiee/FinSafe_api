import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("fraud.pkl")
scaler = joblib.load("scaler.pkl")  # Save and load the same scaler used during training

# Define transaction types from training
transaction_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']  # Adjust based on your dataset

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Ensure features are provided
        features = data.get("features", None)
        if features is None:
            return jsonify({"error": "Missing 'features' key in request body"}), 400
        
        # Extract numerical features
        numerical_features = [
            features["step"],
            features["amount"],
            features["oldbalanceOrg"],
            features["newbalanceOrig"],
            features["oldbalanceDest"],
            features["newbalanceDest"]
        ]

        # One-hot encode the 'type' feature
        type_vector = [1 if features["type"] == t else 0 for t in transaction_types[1:]]  # Drop first category

        # Combine numerical and encoded categorical features
        input_data = np.array(numerical_features + type_vector).reshape(1, -1)

        # Apply scaling
        input_data_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_data_scaled)[0]

        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
