import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Models")  

models = {}


# -------------------------------
# Load Models Safely
# -------------------------------
def load_models():
    loaded_models = {}

    print(f"BASE_DIR: {BASE_DIR}")
    print(f"MODEL_DIR: {MODEL_DIR}")

    if not os.path.exists(MODEL_DIR):
        print(f"WARNING: Model directory not found: {MODEL_DIR}")
        return loaded_models

    for oil_type in os.listdir(MODEL_DIR):
        oil_path = os.path.join(MODEL_DIR, oil_type)

        if os.path.isdir(oil_path):
            for file in os.listdir(oil_path):
                if file.endswith(".joblib"):
                    model_key = f"{oil_type}_{file.replace('.joblib','')}"
                    model_path = os.path.join(oil_path, file)

                    try:
                        loaded_models[model_key] = joblib.load(model_path)
                        print(f"Loaded: {model_key}")
                    except Exception as e:
                        print(f"Failed to load {model_key}: {e}")

    print(f"Total models loaded: {len(loaded_models)}")
    return loaded_models


# Load models at startup
models = load_models()


# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return "Oil Prediction Models API Running"


@app.route("/models")
def list_models():
    return jsonify({
        "available_models": list(models.keys())
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Get values safely
        model_name = data.get("model")
        raw_features = data.get("features", [])

        # Validate model
        if model_name not in models:
            return jsonify({"error": "Model not found"}), 400

        # -------------------------------
        # Clean & Normalize Input
        # -------------------------------
        features = []

        for x in raw_features:
            try:
                val = float(x)

                # Replace NaN / Inf with 0
                if np.isnan(val) or np.isinf(val):
                    val = 0.0

            except:
                val = 0.0

            features.append(val)

        # Ensure exactly 16 features
        if len(features) < 16:
            features += [0.0] * (16 - len(features))
        elif len(features) > 16:
            features = features[:16]

        features = np.array(features).reshape(1, -1)

        # -------------------------------
        # Prediction
        # -------------------------------
        prediction = models[model_name].predict(features)

        # Round to 4 decimal places
        rounded_prediction = [round(float(p), 4) for p in prediction]

        return jsonify({
            "model": model_name,
            "prediction": rounded_prediction
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# -------------------------------
# Debug Route (VERY USEFUL)
# -------------------------------
@app.route("/debug")
def debug():
    return {
        "model_dir_exists": os.path.exists(MODEL_DIR),
        "model_dir_path": MODEL_DIR,
        "folders": os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else [],
        "loaded_models": list(models.keys())
    }


# -------------------------------
# Run App (local only)
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
