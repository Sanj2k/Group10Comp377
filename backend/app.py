from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import joblib
import numpy as np
import hashlib
import os
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)
CORS(app)


mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client['price_predictor']
users_collection = db['users']


model = joblib.load('model.pkl')

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data['username']
    password = hash_password(data['password'])

    if users_collection.find_one({'username': username}):
        return jsonify({'message': 'Username already exists'}), 409

    users_collection.insert_one({'username': username, 'password': password})
    return jsonify({'message': 'Account created successfully!'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data['username']
    password = hash_password(data['password'])

    user = users_collection.find_one({'username': username, 'password': password})
    if user:
        return jsonify({'message': 'Login successful'}), 200
    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[data['bedrooms'], data['sqft']]])
    prediction = model.predict(features)[0]
    return jsonify({'predicted_price': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)
