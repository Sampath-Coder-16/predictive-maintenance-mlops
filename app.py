from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("artifacts/model.pkl")

@app.route("/")
def home():
    return "Predictive Maintenance API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = model.predict_proba(df)[0][1]
    return jsonify({"failure_probability": float(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
