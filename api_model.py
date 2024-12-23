from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

class WeatherPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def prepare_data(self, data):
        #Convertendo dados meteorologicos para features
        features = pd.DataFrame(data)
        #Extraindo caracteristicas temporais
        data['hour'] = features.index.hour
        data['day'] = features.index.day
        data['month'] = features.index.month
        return features
    def train_model(self, features, target):
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        #Normalizando dados
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        #Salvando o modelo
        joblib.dump(self.model, 'weather_model.joblib')
    def predict(self, features):
        if self.model is None:
            self.model = joblib.load('weather_model.joblib')
        return self.model.predict(features)
#Instanciando o preditor
predictor = WeatherPredictor()

@app.route('/api/weather/current', methods=['GET'])
def get_current_weather():
    #Recebendo dados meteorologicos
    current_weather = {
        'temperature': 25 + np.random.normal(0, 5),
        'humidity': 60 + np.random.normal(0, 10),
        'wind_speed': 10 + np.random.normal(0, 5),
        'precipitation': max(0, np.random.normal(0, 0.5)),
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(current_weather)

@app.route('/api/weather/forecast', methods=['GET'])
def get_weather_forecast():
    #Gerando previsão para as proximas 24 horas
    forecasts = []
    current_time = datetime.now()
    for i in range(24):
        future_time = current_time + timedelta(hours=i)
        #Simulando previsão com alguma variação
        forecast = {
            'temperature': 25 + np.random.normal(0, 5),
            'humidity': 60 + np.random.normal(0, 10),
            'wind_speed': 10 + np.random.normal(0, 5),
            'precipitation': max(0, np.random.normal(0, 0.5)),
            'timestamp': future_time.isoformat()
        }
        forecast.append(forecast)
    return jsonify(forecasts)
