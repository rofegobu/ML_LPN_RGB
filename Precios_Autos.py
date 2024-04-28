## Disponibilizar modelo con Flask

import os

from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import joblib
import numpy as np
import pandas as pd
import json
import random

from unique_values import STATES, MAKES, MODELS

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='API de Predicción de Precios - Automóviles usados',
    description='API para predecir precios de automóviles usados en EE.UU.')


# Espacio de nombres para la API
ns = api.namespace('predict', description='Predicciones')


# Modelo de datos de entrada de la API
model = api.model('PredictionInput', {
    'Year': fields.Integer(required=True, description='Año del vehículo', example=2016),
    'Mileage': fields.Float(required=True, description='Kilometraje del vehículo', example=50000),
    'State': fields.String(required=True, description='Estado donde se encuentra el vehículo', enum=STATES),
    'Make': fields.String(required=True, description='Marca del vehículo', enum=MAKES),
    'Model': fields.String(required=True, description='Modelo del vehículo', enum=MODELS)
})

# Cargar el modelo y preprocesador entrenados
model_auto_price = joblib.load(os.path.dirname(__file__) + '/price_predictor_xgb.pkl')
preprocessor = joblib.load(os.path.dirname(__file__) + '/preprocessor.pkl')

# Ruta del recurso de predicción   
@ns.route('/')
class Predict(Resource):
    @api.expect(model)
    def post(self):
        # Parsea los datos de entrada JSON a un DataFrame
        data = request.json
        input_df = pd.DataFrame([data])
        
        # Realiza el preprocesamiento
        preprocessed_data = preprocessor.transform(input_df)
        
        # Realiza la predicción
        prediction = model_auto_price.predict(preprocessed_data)
        prediction = float(prediction[0])
        
        # Devuelve la predicción en formato JSON
        return jsonify({'predicted_price': prediction})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)