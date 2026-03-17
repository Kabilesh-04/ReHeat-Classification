import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

models = {}

# Load all models from subfolders
for oil_type in os.listdir(MODEL_DIR):
    oil_path = os.path.join(MODEL_DIR, oil_type)

    if os.path.isdir(oil_path):
        for file in os.listdir(oil_path):
            if file.endswith(".joblib"):
                model_key = f"{oil_type}_{file.replace('.joblib','')}"
                model_path = os.path.join(oil_path, file)

                try:
                    models[model_key] = joblib.load(model_path)
                except Exception as e:
                    print(f"Failed to load {model_key}: {e}")

print(f"Loaded {len(models)} models")


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

        # Validate request
        if not data or "model" not in data or "features" not in data:
            return jsonify({"error": "Missing model or features"}), 400

        model_name = data["model"]
        features = np.array(data["features"], dtype=float)

        # Validate model
        if model_name not in models:
            return jsonify({"error": "Model not found"}), 400

        # Validate feature length
        if features.shape[0] != 16:
            return jsonify({"error": "Expected 16 features"}), 400

        # Check for NaN or Inf
        if np.isnan(features).any() or np.isinf(features).any():
            return jsonify({"error": "Invalid values (NaN or Inf)"}), 400

        # Prevent all-zero input
        if np.all(features == 0):
            return jsonify({"error": "Invalid input: all features are zero"}), 400

        # Handle zero variance (all values same)
        if np.std(features) == 0:
            features = features + 1e-6

        features = features.reshape(1, -1)

        # Prediction
        prediction = models[model_name].predict(features)

        return jsonify({
            "model": model_name,
            "prediction": prediction.tolist()
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
