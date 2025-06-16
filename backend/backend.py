from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # React will send JSON
        features = [
            float(data['studytime']),
            float(data['failures']),
            float(data['absences']),
            float(data['Medu']),
            float(data['Fedu']),
            float(data['G1']),
            float(data['G2']),
            float(data['traveltime']),
            float(data['health']),
            float(data['Dalc']),
        ]
        prediction = model.predict([np.array(features)])
        label_map = {0: 'Fail', 2: 'Pass'}
        result = label_map.get(prediction[0], 'Unknown')
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
