import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load models once at startup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

models = {
    "elastic_sunflower": joblib.load(os.path.join(MODEL_DIR, "ElasticNet_Sunflower.joblib")),
    "ridge_sunflower": joblib.load(os.path.join(MODEL_DIR, "Ridge_Sunflower.joblib")),
    "poly_sunflower": joblib.load(os.path.join(MODEL_DIR, "Polynomial_Regression_Degree_2_Sunflower.joblib")),
    "rf_gingelly": joblib.load(os.path.join(MODEL_DIR, "Random_Forest_Gingelly.joblib")),
    "extra_gingelly": joblib.load(os.path.join(MODEL_DIR, "Extra_Trees_Gingelly.joblib")),
    "mlp_gingelly": joblib.load(os.path.join(MODEL_DIR, "Neural_Network_MLP_Gingelly.joblib")),
}

@app.route("/")
def home():
    return "Heating Models API Running"

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
