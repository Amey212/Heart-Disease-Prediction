from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and scaler
rf_model = joblib.load('rf_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        # Handle API requests (JSON)
        data = request.get_json()
        input_data = pd.DataFrame(data, index=[0])
        final_features_scaled = scaler.transform(input_data)
    else:
        # Handle form submissions (HTML form)
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        final_features_scaled = scaler.transform(final_features)

    # Make predictions
    prediction = rf_model.predict(final_features_scaled)[0]
    risk_percentage = rf_model.predict_proba(final_features_scaled)[0][1] * 100

    # Determine risk level
    if risk_percentage >= 80:
        result = 'The patient has a high risk of heart attack.'
    else:
        result = 'The patient has a low risk of heart attack.'

    risk = f'Risk percentage: {risk_percentage:.2f}%'

    # Return response based on request type
    if request.is_json:
        return jsonify({
            'prediction': int(prediction),
            'risk_percentage': float(risk_percentage),
            'result': result
        })
    else:
        return render_template('index.html', prediction_text=result, risk_percentage=risk)

if __name__ == "__main__":
    app.run(debug=True)