from flask import Flask, request, render_template, jsonify, send_file
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model/pcos_svm_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')  # This serves your index.html file

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        cycle = request.form['Cycle']
        # Convert R/I to numerical values (e.g., R=0, I=1 or whatever your model expects)
        cycle_numeric = 0 if cycle.upper() == 'R' else 1

        data = [
            int(request.form['Age']),
            float(request.form['Weight']),
            float(request.form['Height']),
            float(request.form['BMI']),
            int(request.form['Blood_Group']),
            cycle_numeric,
            int(request.form['Cycle_length']),
            int(request.form['Marriage_Status']),
            int(request.form['Pregnant']),
            int(request.form['No_of_abortions']),
            int(request.form['Weight_gain']),
            int(request.form['Hair_growth']),
            int(request.form['Skin_darkening']),
            int(request.form['Hair_loss']),
            int(request.form['Pimples']),
            int(request.form['Fast_food']),
            int(request.form['Reg_Exercise'])
        ]

        input_array = np.array(data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)

        result = {
            "prediction": "PCOS" if prediction == 1 else "No PCOS",
            "probability_PCOS": f"{prediction_proba[0][1] * 100:.2f}%",
            "probability_No_PCOS": f"{prediction_proba[0][0] * 100:.2f}%"
        }

        return render_template('result.html', result=result)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/predict_from_server', methods=['GET'])
def predict_from_server():
    try:
        # Path to the dataset on the server
        dataset_path = os.path.join('data', 'PCOD-10.csv')  # Adjust the path as needed

        # Load the dataset
        if not os.path.exists(dataset_path):
            return jsonify({"error": f"Dataset not found at {dataset_path}"})

        dataset = pd.read_csv(dataset_path)

        # Ensure the dataset has all required columns
        required_columns = ['Age', 'Weight', 'Height', 'BMI', 'Blood_Group', 'Cycle', 'Cycle_length',
                            'Marriage_Status', 'Pregnant', 'No_of_abortions', 'Weight_gain',
                            'Hair_growth', 'Skin_darkening', 'Hair_loss', 'Pimples',
                            'Fast_food', 'Reg_Exercise']
        if not all(col in dataset.columns for col in required_columns):
            return jsonify({"error": "Dataset is missing one or more required columns"})

        # Preprocess the dataset
        dataset['Cycle'] = dataset['Cycle'].apply(lambda x: 0 if x.upper() == 'R' else 1)
        input_data = dataset[required_columns]
        input_scaled = scaler.transform(input_data)

        # Make predictions
        predictions = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)

        # Add predictions to the dataset
        dataset['prediction'] = ['PCOS' if pred == 1 else 'No PCOS' for pred in predictions]
        dataset['probability_PCOS'] = [f"{prob[1] * 100:.2f}%" for prob in probabilities]
        dataset['probability_No_PCOS'] = [f"{prob[0] * 100:.2f}%" for prob in probabilities]

        # Save the results to a new CSV file
        output_file = os.path.join('data', 'prediction_results.csv')
        dataset.to_csv(output_file, index=False)

        return jsonify({"message": "Predictions completed", "result_file": output_file})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='127.0.0.1')
