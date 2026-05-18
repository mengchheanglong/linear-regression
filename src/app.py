import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define the absolute or relative path to the trained model
MODEL_PATH = os.path.join("models", "house_price_model.pkl")

# Load the model at startup
model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("--- Model loaded successfully from models/house_price_model.pkl ---")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
else:
    print(f"Warning: Model file not found at {MODEL_PATH}. Prediction endpoint will fail until it exists.")


@app.route("/", methods=["GET"])
def health_check():
    """
    Health check endpoint to ensure the API is up and running.
    Useful for Render deployment liveness checks.
    """
    return jsonify({
        "status": "healthy",
        "message": "House Price Prediction API is running.",
        "model_loaded": model is not None
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint. Expects a JSON object containing feature names and values.
    """
    if not model:
        return jsonify({"error": "Model is not loaded on the server. Contact Admin."}), 500

    # 1. Check if content type is JSON
    if not request.is_json:
        return jsonify({"error": "Invalid Content-Type. Must be application/json"}), 400

    data = request.get_json()

    # 2. Validate input is not empty
    if not data:
        return jsonify({"error": "Request body cannot be empty"}), 400

    try:
        # If a single dictionary is sent, wrap it in a list to convert cleanly to a DataFrame
        if isinstance(data, dict):
            input_data = [data]
        elif isinstance(data, list):
            input_data = data
        else:
            return jsonify({"error": "Input data must be a JSON Object {} or an Array of Objects []"}), 400

        # 3. Convert input to Pandas DataFrame to preserve feature column names for the model
        df = pd.DataFrame(input_data)

        # 4. Generate Predictions
        # Linear Regression returns a numpy array. We grab the first element if they sent a single object.
        predictions = model.predict(df)
        
        # Format output safely
        predicted_price = float(predictions[0])

        return jsonify({
            "predicted_price": round(predicted_price, 2)
        }), 200

    except KeyError as ke:
        return jsonify({"error": f"Missing or incorrect feature column name: {str(ke)}"}), 400
    except ValueError as ve:
        return jsonify({"error": f"Value error during prediction (check data types): {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    # Host 0.0.0.0 allows external access (essential for Docker/Render deployment later)
    app.run(host="0.0.0.0", port=5000, debug=True)