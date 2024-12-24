from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

API_KEY = 'YOUR_API_KEY'
BASE_URL = 'https://api.openweathermap.org/data/2.5/weather'

def get_weather_data(city):
    url = f"{BASE_URL}?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None


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
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
        self.model.fit(X_train, y_train)
        #Salvando o modelo
        joblib.dump(self.model, 'weather_model.pkl')

    def predict(self, features):
        if self.model is None:
            self.model = joblib.load('weather_model.plk')
        return self.model.predict(features)

predictor = WeatherPredictor()

@app.route('/')
def home():
    #alterar o template depois
    return render_template('index.html')

@app.route('/api/weather/current', methods=['GET'])
def get_current_weather():
    city = request.args.get('city', 'Maceio')
    weather_data = get_weather_data(city)
    if weather_data:
        current_weather = {
            'temperature': weather_data['main']['temp'],
            'humidity': weather_data['main']['humidity'],
            'wind_speed': weather_data['wind']['speed'],
            'precipitation': weather_data.get('rain', {}).get('1h', 0),
            'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        }
    else:
        current_weather = {
            'temperature': 25 + np.random.normal(0, 2),
            'humidity': 60 + np.random.normal(0, 5),
            'wind_speed': 10 + np.random.normal(0, 1),
            'precipitation': max(0, np.random.normal(0, 0.5)),
            'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        }
    return jsonify(current_weather)

@app.route('/api/weather/forecast', methods=['GET'])
def get_forecast():
    city = request.args.get('city', 'Maceio')
    weather_data = get_weather_data(city)
    #Gerando previsão para as proximas 24 horas
    forecasts = []
    current_time = datetime.now()
    for i in range(24):
        future_time = current_time + timedelta(hours=i)
        #Simulando previsão com alguma variação
        forecast = {
            'temperature': weather_data['main']['temp'] + np.random.normal(0, 2) + np.sin(i / 24 * 2 * np.pi) * 5,
            'humidity': weather_data['main']['humidity'] + np.random.normal(0, 5),
            'wind_speed': weather_data['wind']['speed'] + np.random.normal(0, 1),
            'precipitation': max(0, weather_data.get('rain', {}).get('1h', 0) + np.random.normal(0, 0.5)),
            'timestamp': future_time.strftime('%d/%m/%Y %H:%M:%S')
        }
        forecasts.append(forecast)
    return jsonify(forecasts)

@app.route('/api/weather/historical', methods=['GET'])
def get_historical_data():
    days = int(request.args.get('days', 7))
    city = request.args.get('city', 'Maceio')
    weather_data = get_weather_data(city)
    historical_data = []
    current_time = datetime.now()
    for i in range(days * 24):
        past_time = current_time - timedelta(hours=i)
        data_point = {
            'temperature': weather_data['main']['temp'] + np.random.normal(0, 3),
            'humidity': weather_data['main']['humidity'] + np.random.normal(0, 8),
            'wind_speed': weather_data['wind']['speed'] + np.random.normal(0, 2),
            'precipitation': max(0, weather_data.get('rain', {}).get('1h', 0) + np.random.normal(0, 1)),        
            'timestamp': past_time.strftime('%d/%m/%Y %H:%M:%S')
        }
        historical_data.append(data_point)
    return jsonify(historical_data)

if __name__ == '__main__':
    print("Servidor rodando na porta 5000")
    app.run(debug=True)
