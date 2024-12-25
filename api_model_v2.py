from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
from time import sleep
import joblib
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

load_dotenv()

CLIMATEMPO_API_KEY = os.getenv('CLIMATEMPO_API_KEY')
CLIMATEMPO_BASE_URL = "http://apiadvisor.climatempo.com.br/api/v1"

def make_request_with_retry(url, params=None, method='get', max_retries=3):
    """Faz requisição com retry automático respeitando o rate limit"""
    for attempt in range(max_retries):
        try:
            if method.lower() == 'get':
                response = requests.get(url, params=params)
            elif method.lower() == 'put':
                response = requests.put(url, params=params)
            
            if response.status_code == 429:
                retry_after = int(response.json().get('retry-after', '7').split()[0])
                print(f"Rate limit atingido. Aguardando {retry_after} segundos...")
                time.sleep(retry_after + 1)  # Adiciona 1 segundo extra por segurança
                continue
                
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)  # Backoff exponencial
    
    return None

def register_city(city_id):
    """Registra uma cidade para uso com o token"""
    try:
        register_url = f"{CLIMATEMPO_BASE_URL}/locale/city/{city_id}/register"
        print(f"Tentando registrar cidade ID {city_id}")
        
        response = make_request_with_retry(
            register_url,
            params={"token": CLIMATEMPO_API_KEY},
            method='put'
        )
        
        if response:
            print(f"Registro bem-sucedido para cidade {city_id}")
            return True
            
        return False
    except Exception as e:
        print(f"Erro ao registrar cidade: {str(e)}")
        return False

def get_city_id(city_name):
    """Obtém o ID da cidade e tenta registrá-la"""
    try:
        print(f"Buscando cidade: {city_name}")
        response = make_request_with_retry(
            f"{CLIMATEMPO_BASE_URL}/locale/city",
            params={
                "name": city_name,
                "token": CLIMATEMPO_API_KEY,
                "country": "BR"
            }
        )
        
        if not response:
            return None
            
        cities = response.json()
        
        if isinstance(cities, list) and cities:
            city_id = cities[0].get("id")
            if city_id:
                print(f"ID da cidade encontrado: {city_id}")
                success = register_city(city_id)
                if success:
                    print(f"Cidade {city_name} (ID: {city_id}) registrada com sucesso")
                return city_id
        return None
        
    except Exception as e:
        print(f"Erro ao buscar cidade: {str(e)}")
        return None

@app.route('/')
def home():
    return render_template('index_v2.html')

@app.route('/api/weather/current', methods=['GET'])
def get_current_weather():
    city = request.args.get('city', 'São Paulo')
    city_id = get_city_id(city)
    
    if not city_id:
        return jsonify({"error": "Cidade não encontrada"}), 404

    try:
        # Tenta obter dados atuais
        current_response = requests.get(
            f"{CLIMATEMPO_BASE_URL}/weather/locale/{city_id}/current",
            params={"token": CLIMATEMPO_API_KEY}
        )
        
        if current_response.status_code == 200:
            data = current_response.json()
            return jsonify({
                'temperature': data['data']['temperature'],
                'humidity': data['data']['humidity'],
                'wind_speed': data['data']['wind_velocity'],
                'precipitation': data['data'].get('rain', 0),
                'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M:%S')
            })
        
        # Se falhar, tenta obter a previsão do dia
        forecast_response = requests.get(
            f"{CLIMATEMPO_BASE_URL}/forecast/locale/{city_id}/days/1",
            params={"token": CLIMATEMPO_API_KEY}
        )
        
        if forecast_response.status_code == 200:
            data = forecast_response.json()
            if 'data' in data and data['data']:
                current_data = data['data'][0]
                return jsonify({
                    'temperature': current_data['temperature']['temperature'],
                    'humidity': current_data['humidity']['humidity'],
                    'wind_speed': current_data['wind']['velocity'],
                    'precipitation': current_data['rain']['precipitation'],
                    'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M:%S')
                })
        
        return jsonify({"error": "Dados não disponíveis"}), 404
        
    except Exception as e:
        print(f"Erro API current: {e}")
        return jsonify({"error": "Erro ao obter dados"}), 500

@app.route('/api/weather/forecast', methods=['GET'])
def get_forecast():
    city = request.args.get('city', 'São Paulo')
    print(f"\nIniciando previsão para cidade: {city}")
    
    city_id = get_city_id(city)
    if not city_id:
        return jsonify({"error": "Cidade não encontrada"}), 404

    try:
        forecast_url = f"{CLIMATEMPO_BASE_URL}/forecast/locale/{city_id}/days/15"
        response = make_request_with_retry(
            forecast_url,
            params={"token": CLIMATEMPO_API_KEY}
        )
        
        if response:
            data = response.json()
            forecasts = []
            
            if 'data' in data and data['data']:
                base_data = data['data'][0]
                current_hour = datetime.now().hour
                
                for hour in range(24):
                    temp_base = base_data['temperature']['temperature']
                    hour_offset = abs(hour - 14)  # Pico de temperatura às 14h
                    temp_variation = -0.3 * hour_offset
                    hour_temp = temp_base + temp_variation
                    
                    forecast_time = datetime.now().replace(hour=hour, minute=0, second=0)
                    if hour < current_hour:
                        forecast_time += timedelta(days=1)
                    
                    forecast = {
                        'temperature': round(hour_temp, 1),
                        'humidity': base_data['humidity']['humidity'],
                        'wind_speed': base_data['wind']['velocity'],
                        'precipitation': base_data['rain']['precipitation'],
                        'timestamp': forecast_time.strftime('%d/%m/%Y %H:%M:%S')
                    }
                    forecasts.append(forecast)
                
                return jsonify(sorted(forecasts, key=lambda x: x['timestamp']))
                
        return jsonify({"error": "Dados não disponíveis"}), 404
        
    except Exception as e:
        print(f"Erro na previsão: {str(e)}")
        return jsonify({"error": "Erro ao obter previsão"}), 500
    
@app.route('/api/weather/historical', methods=['GET'])
def get_historical_data():
    city = request.args.get('city', 'São Paulo')
    days = int(request.args.get('days', 7))
    city_id = get_city_id(city)
    
    if not city_id:
        return jsonify({"error": "Cidade não encontrada"}), 404

    try:
        response = requests.get(
            f"{CLIMATEMPO_BASE_URL}/climate/locale/{city_id}",
            params={"token": CLIMATEMPO_API_KEY}
        )
        
        if response.status_code == 200:
            climate_data = response.json()
            historical_data = []
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            current_date = start_date
            
            while current_date <= end_date:
                month_data = next((m for m in climate_data.get('data', []) 
                                 if m['month'] == current_date.month), None)
                
                if month_data:
                    # Adiciona variação diária aos dados mensais
                    base_temp = month_data['temperature']['mean']
                    temp_var = np.random.normal(0, 2)
                    
                    data_point = {
                        'temperature': round(base_temp + temp_var, 1),
                        'humidity': round(70 + np.random.normal(0, 5)),
                        'wind_speed': round(10 + np.random.normal(0, 2)),
                        'precipitation': month_data['rain']['mean'],
                        'timestamp': current_date.strftime('%d/%m/%Y %H:%M:%S')
                    }
                    historical_data.append(data_point)
                
                current_date += timedelta(days=1)
            
            return jsonify(historical_data)
        
        return jsonify({"error": "Dados climáticos não disponíveis"}), 404
        
    except Exception as e:
        print(f"Erro API historical: {e}")
        return jsonify({"error": "Erro ao obter dados históricos"}), 500

if __name__ == '__main__':
    app.run(debug=True)