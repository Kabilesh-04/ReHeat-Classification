import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

models = {}

# Automatically load all models from subfolders
for oil_type in os.listdir(MODEL_DIR):
    oil_path = os.path.join(MODEL_DIR, oil_type)

    if os.path.isdir(oil_path):
        for file in os.listdir(oil_path):
            if file.endswith(".joblib"):
                model_key = f"{oil_type}_{file.replace('.joblib','')}"
                model_path = os.path.join(oil_path, file)

                models[model_key] = joblib.load(model_path)

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

        model_name = data["model"]
        features = np.array(data["features"]).reshape(1, -1)

        if model_name not in models:
            return jsonify({"error": "Model not found"}), 400

        prediction = models[model_name].predict(features)

        return jsonify({
            "model": model_name,
            "prediction": prediction.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
