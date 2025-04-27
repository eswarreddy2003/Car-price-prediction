from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the model and data
try:
    model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
    car_data = pd.read_csv('Cleaned_Car_data.csv')
except Exception as e:
    print(f"Error loading model or data: {e}")
    raise e

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        companies = sorted(car_data['company'].unique())
        car_models = sorted(car_data['name'].unique())
        years = sorted(car_data['year'].unique(), reverse=True)
        fuel_types = car_data['fuel_type'].unique()
        
        return render_template('index.html', 
                             companies=companies, 
                             car_models=car_models, 
                             years=years, 
                             fuel_types=fuel_types)
    except Exception as e:
        print(f"Error in index route: {e}")
        return render_template('index.html', 
                             companies=[], 
                             car_models=[], 
                             years=[], 
                             fuel_types=[])

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        data = request.form
        
        # Validate input data
        required_fields = ['company', 'car_models', 'year', 'fuel_type', 'kilo_driven']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare input for prediction
        input_data = pd.DataFrame([[data['car_models'], 
                                 data['company'], 
                                 int(data['year']), 
                                 int(data['kilo_driven']), 
                                 data['fuel_type']]],
                               columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
        
        # Make prediction
        prediction = model.predict(input_data)
        predicted_price = np.round(prediction[0], 2)
        
        return jsonify({'predicted_price': predicted_price})
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

@app.route('/get_models', methods=['POST'])
@cross_origin()
def get_models():
    try:
        company = request.form.get('company')
        if not company:
            return jsonify({'error': 'Company not provided'}), 400
            
        models = sorted(car_data[car_data['company'] == company]['name'].unique())
        return jsonify({'models': models})
    
    except Exception as e:
        print(f"Error getting models: {e}")
        return jsonify({'error': 'Failed to get models'}), 500

if __name__ == '__main__':
    app.run(debug=True)