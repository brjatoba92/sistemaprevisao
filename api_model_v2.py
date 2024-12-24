from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import logging

app = Flask(__name__)
CORS(app)

# Configuração de logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configurações da API Climatempo
CLIMATEMPO_API_KEY = "d28eb2737361a7897d75a20e4f9b2359"  # Substitua pela sua chave real
CLIMATEMPO_BASE_URL = "http://apiadvisor.climatempo.com.br/api/v1"

class WeatherPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def prepare_data(self, data):
        features = pd.DataFrame(data)
        features['hour'] = pd.to_datetime(features['timestamp']).dt.hour
        features['day'] = pd.to_datetime(features['timestamp']).dt.day
        features['month'] = pd.to_datetime(features['timestamp']).dt.month
        return features
        
    def train_model(self, features, target):
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, 'weather_model.pkl')
        
    def predict(self, features):
        if self.model is None:
            self.model = joblib.load('weather_model.pkl')
        return self.model.predict(features)

predictor = WeatherPredictor()

def register_city(city_id):
    try:
        url = f"{CLIMATEMPO_BASE_URL}/locale/city/{city_id}/register"
        params = {'token': CLIMATEMPO_API_KEY}
        logger.debug(f"Tentando registrar cidade ID {city_id} no URL {url}")
        response = requests.put(url, params=params)
        logger.debug(f"Resposta do registro: {response.status_code} - {response.text}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Erro ao registrar cidade {city_id}: {e}")
        return False
            
    except Exception as e:
        logger.error(f"Erro ao registrar cidade: {str(e)}")
        return False

def get_city_id(city_name):
    """Busca o ID da cidade na API do Climatempo e registra se necessário"""
    try:
        url = f"{CLIMATEMPO_BASE_URL}/locale/city"
        params = {
            'name': city_name,
            'token': CLIMATEMPO_API_KEY
        }
        
        #logger.debug(f"Buscando cidade: {city_name}")
        response = requests.get(url, params=params)
        logger.debug(f"Resposta completa: {response.text}")
        
        if response.status_code == 200:
            cities = response.json()
            if cities and len(cities) > 0:
                city_id = cities[0]['id']
                logger.info(f"Cidade encontrada: {city_name} (ID: {city_id})")
                
                # Tenta registrar a cidade
                if register_city(city_id):
                    return city_id
                else:
                    logger.warning(f"Não foi possível registrar a cidade {city_name}")
                    return None
            else:
                logger.warning(f"Nenhuma cidade encontrada para: {city_name}")
                return None
        else:
            logger.error(f"Erro ao buscar cidade. Status: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Erro ao buscar ID da cidade: {str(e)}")
        return None

@app.route('/')
def home():
    return render_template('index_v2.html')

@app.route('/api/weather/current', methods=['GET'])
def get_current_weather():
    """Obtém dados meteorológicos atuais para uma cidade"""
    try:
        city = request.args.get('city', 'São Paulo')
        logger.info(f"Buscando dados atuais para: {city}")
        
        city_id = get_city_id(city)
        if not city_id:
            return jsonify({'error': f'Cidade não encontrada ou não foi possível registrá-la: {city}'}), 404
            
        url = f"{CLIMATEMPO_BASE_URL}/weather/locale/{city_id}/current"
        params = {'token': CLIMATEMPO_API_KEY}
        
        response = requests.get(url, params=params)
        logger.debug(f"Status da resposta: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            logger.debug(f"Dados recebidos: {data}")
            
            if 'data' in data:
                current_weather = {
                    'temperature': data['data'].get('temperature', 0),
                    'humidity': data['data'].get('humidity', 0),
                    'wind_speed': data['data'].get('wind_velocity', 0),
                    'precipitation': data['data'].get('precipitation', 0),
                    'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                    'city': city,
                    'condition': data['data'].get('condition', '')
                }
                return jsonify(current_weather)
            else:
                return jsonify({'error': 'Dados não disponíveis'}), 500
        elif response.status_code == 400:
            return jsonify({'error': 'Cidade não registrada ou limite de requisições excedido'}), 400
        else:
            return jsonify({'error': f'Erro ao obter dados: {response.status_code}'}), response.status_code
            
    except Exception as e:
        logger.error(f"Erro ao obter dados atuais: {str(e)}")
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500

@app.route('/api/weather/forecast', methods=['GET'])
def get_forecast():
    """Obtém previsão do tempo para os próximos dias"""
    try:
        city = request.args.get('city', 'São Paulo')
        logger.info(f"Buscando previsão para: {city}")
        
        city_id = get_city_id(city)
        if not city_id:
            return jsonify({'error': f'Cidade não encontrada: {city}', 'forecasts': []}), 404
            
        url = f"{CLIMATEMPO_BASE_URL}/forecast/locale/{city_id}/hours/72"
        params = {'token': CLIMATEMPO_API_KEY}
        
        response = requests.get(url, params=params)
        logger.debug(f"Status da resposta: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            forecasts = []
            
            if 'data' in data and isinstance(data['data'], list):
                for forecast in data['data'][:24]:  # Limita a 24 horas
                    forecast_data = {
                        'temperature': forecast.get('temperature', {}).get('temperature', 0),
                        'humidity': forecast.get('humidity', {}).get('humidity', 0),
                        'wind_speed': forecast.get('wind', {}).get('velocity', 0),
                        'precipitation': forecast.get('rain', {}).get('precipitation', 0),
                        'timestamp': forecast.get('date', ''),
                        'city': city
                    }
                    forecasts.append(forecast_data)
                
            return jsonify({'forecasts': forecasts, 'error': None})
        else:
            return jsonify({'error': f'Erro ao obter previsão: {response.status_code}', 'forecasts': []}), response.status_code
            
    except Exception as e:
        logger.error(f"Erro ao obter previsão: {str(e)}")
        return jsonify({'error': f'Erro interno: {str(e)}', 'forecasts': []}), 500

@app.route('/api/weather/historical', methods=['GET'])
def get_historical_data():
    """Obtém dados históricos (simulados) para uma cidade"""
    try:
        city = request.args.get('city', 'São Paulo')
        days = int(request.args.get('days', 7))
        
        city_id = get_city_id(city)
        if not city_id:
            return jsonify({'error': f'Cidade não encontrada: {city}'}), 404
        
        # Simulando dados históricos
        historical_data = []
        current_time = datetime.now()
        
        for i in range(days * 24):
            past_time = current_time - timedelta(hours=i)
            data_point = {
                'temperature': 25 + np.random.normal(0, 3),
                'humidity': 60 + np.random.normal(0, 8),
                'wind_speed': 10 + np.random.normal(0, 2),
                'precipitation': max(0, np.random.normal(0, 1)),
                'timestamp': past_time.strftime('%d/%m/%Y %H:%M:%S'),
                'city': city
            }
            historical_data.append(data_point)
        
        return jsonify(historical_data)
        
    except Exception as e:
        logger.error(f"Erro ao obter dados históricos: {str(e)}")
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500

if __name__ == '__main__':
    print("Servidor rodando na porta 5000")
    app.run(debug=True)