from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import joblib

app = Flask(__name__)

# Load the trained model and scaler
with open('model/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('model/scaler.pkl', 'rb') as scaler_file:
    scaler = joblib.load(scaler_file)

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/predict', methods=['GET'])
def predict_form():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(data)
        
        # print(data)
        print([key for key in data])

        # Get form data and convert to floats
        input_data = [float(data[key]) for key in data]
        
        # Reshape and scale input
        input_array = np.array(input_data).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
              
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        result = "Positive (Likely Diabetic)" if prediction == 1 else "Negative (Unlikely Diabetic)"

        print(f"Input Data: {input_data}")
        print(f"Scaled Input: {scaled_input}")
        print(f"Prediction: {prediction}")
        print(f"Result: {result}")
        
        return jsonify(result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
