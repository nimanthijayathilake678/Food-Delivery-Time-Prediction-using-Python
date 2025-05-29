from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np

app = Flask(__name__)
model = load_model("saved_model.keras")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    age = data['age']
    rating = data['rating']
    distance = data['distance']

    features = np.array([[age, rating, distance]])
    prediction = model.predict(features)
    return jsonify({'predicted_time': float(prediction[0][0])})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
