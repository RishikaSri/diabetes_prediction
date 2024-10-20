from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model using joblib
loaded_model = joblib.load(r"C:\Users\gadda\OneDrive\Desktop\diabetes project\mod.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    age = int(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    bmi = float(request.form['bmi'])
    hba1c_level = float(request.form['HbA1c_level'])
    blood_glucose_level = float(request.form['blood_glucose_level'])
    
    # Create input array for the model (adjust order if necessary)
    features = np.array([[age, hypertension, heart_disease, bmi, hba1c_level, blood_glucose_level]])
    
    # Make a prediction using the model
    prediction = loaded_model.predict(features)[0]

    # Determine the output message based on the prediction result
    if prediction == 1:
        prediction_text = "You might have diabetes."
    else:
        prediction_text = "You are less likely to have diabetes."

    # Pass the prediction to the template
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
