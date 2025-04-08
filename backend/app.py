from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[data['bedrooms'], data['sqft']]])
    prediction = model.predict(features)[0]
    return jsonify({'predicted_price': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)
