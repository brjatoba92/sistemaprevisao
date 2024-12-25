from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv
import datetime as dt

app = Flask(__name__)
CORS(app)

load_dotenv()

# Conexão com a API da Climatempo
CLIMATEMPO_API_KEY = os.getenv('API_KEY')
CLIMATEMPO_BASE_URL = "http://apiadvisor.climatempo.com.br/api/v1"

class WeatherPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler
    def prepare_data(self, data):
        features = pd.DataFrame(data)
        features['hour'] = pd.to_datetime(features.index).hour
        features['day'] = pd.to_datetime(features.index).day
        features['month'] = pd.to_datetime(features.index).month
        return features

    def train_model(self, features, target):
        self.model.fit(features, target)
        joblib.dump(self.model, 'weather_model.pkl')

    def predict(self, features):
        return self.model.predict(features)

predictor = WeatherPredictor()

def get_city_id(city_name):
    response = requests.get(
        f"{CLIMATEMPO_BASE_URL}/locale/city",
        params={'name': city_name, 'token': CLIMATEMPO_API_KEY}
    )
    if response.status_code == 200:
        cities = response.json()
        return cities[0]['id'] if cities else None
    return None

###Rotas

#Rota de retornar a pagina principal
@app.route('/')
def home():
    return render_template('index_v2.html')
#Rota de tempo atual
@app.route('/api/weather/current', methods=['GET'])
def get_current_weather():
    city = request.args.get('city', 'São Paulo')
    city_id = get_city_id(city)
    if not city_id:
        return jsonify({'error': f'Cidade nao encontrada: {city}'}), 404
    url = f"{CLIMATEMPO_BASE_URL}/weather/locale/{city_id}/current"
    params = {'token': CLIMATEMPO_API_KEY}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return jsonify({
            'temperature': data['data']['temperature'],
            'humidity': data['data']['humidity'],
            'wind_speed': data['data']['wind_velocity'],
            'precipitation': data['data']['rain'],
            'timestamp': dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            'city': city
        })
    return jsonify({'error': 'Erro ao obter dados meteorologicos'}), 500
#Rota de previsão
@app.route('/api/weather/forecast', methods=['GET'])
def get_forecast():
    city = request.args.get('city', 'São Paulo')
    city_id = get_city_id(city)
    if not city_id:
        return jsonify({'error': f'Cidade nao encontrada: {city}'}), 404
    response = requests.get(
        f"{CLIMATEMPO_BASE_URL}/forecast/locale/{city_id}/hours/72",
        params={'token': CLIMATEMPO_API_KEY}
    )
    if response.status_code == 200:
        data = response.json()
        forecasts = []
        for hour_data in data['data'][:24]: # Limita a 24 horas
            forecast = {
                'temperature': hour_data['temperature'],
                'humidity': hour_data['humidity'],
                'wind_speed': hour_data['wind_velocity'],
                'precipitation': hour_data['rain'],
                'timestamp': hour_data['date'],
                'city': city
            }
            forecasts.append(forecast)
        return jsonify(forecasts)
    return jsonify({'error': 'Erro ao obter previsao'}), 500

#Rota de historico
@app.route('/api/weather/historical', methods=['GET'])
def get_historical_data():
    city = request.args.get('city', 'São Paulo')
    days = int(request.args.get('days', 7))
    city_id = get_city_id(city)
    if not city_id:
        return jsonify({'error': f'Cidade nao encontrada: {city}'}), 404
    
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=days)

    response = requests.get(
        f"{CLIMATEMPO_BASE_URL}/forecast/locale/{city_id}/historical",
        params={
            'token': CLIMATEMPO_API_KEY,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        }
    )
    if response.status_code == 200:
        return jsonify(response.json())
    return jsonify({'error': 'Erro ao obter dados históricos'}), 500

if __name__ == '__main__':
    app.run(debug=True)   