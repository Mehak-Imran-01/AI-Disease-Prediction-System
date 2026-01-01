from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import pandas as pd
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load model & label encoder (Windows-safe paths using forward slashes)
rf_model = joblib.load("C:/Users/TOSHIBA/Desktop/ADBMS Project/model/disease_rf_model.pkl")
label_encoder = joblib.load("C:/Users/TOSHIBA/Desktop/ADBMS Project/model/label_encoder.pkl")

# Load symptom list
df = pd.read_csv("C:/Users/TOSHIBA/Desktop/ADBMS Project/dataset/Training.csv")
if 'Unnamed: 133' in df.columns:
    df = df.drop(columns=['Unnamed: 133'])
ALL_SYMPTOMS = list(df.drop(columns=['prognosis']).columns)

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["disease_db"]
collection = db["patients"]

# Helper function to convert symptoms to binary vector
def symptoms_to_vector(user_symptoms):
    return np.array([
        1 if symptom in user_symptoms else 0
        for symptom in ALL_SYMPTOMS
    ])

# Prediction API
@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_name = data.get("name", "Anonymous")
    user_symptoms = data.get("symptoms", [])

    user_vector = symptoms_to_vector(user_symptoms)

    prediction = rf_model.predict(user_vector.reshape(1, -1))
    probability = rf_model.predict_proba(user_vector.reshape(1, -1))

    disease = label_encoder.inverse_transform(prediction)[0]
    confidence = float(np.max(probability))

    record = {
        "name": user_name,
        "symptoms": user_symptoms,
        "predicted_disease": disease,
        "confidence": confidence,
        "timestamp": datetime.now()
    }


    # Insert into MongoDB and get the inserted_id
    inserted = collection.insert_one(record)

    # Convert MongoDB ObjectId to string so jsonify works
    record["_id"] = str(inserted.inserted_id)

    return jsonify(record)


if __name__ == "__main__":
    app.run(debug=True)
