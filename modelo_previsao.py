import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

class WeatherPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
    
    def prepare_data(self, data):
        """
        Prepara os dados para treinamento do modelo.
        Parâmetros:
        - data: DataFrame com os dados de temperatura, umidade, pressão, velocidade do vento e precipitação.
        Retorna:
        - X: DataFrame com as features para treinamento.
        - y: Série com os valores de temperatura.
        """
        #Criando features com dados do dia anterior
        data['temp_anterior'] = data['temperature'].shift(1)
        data['umidade_anterior'] = data['humidity'].shift(1)
        data['pressao_anterior'] = data['pressure'].shift(1)
        #Removendo linhas com dados faltantes
        data = data.dropna()
        #Separando features e target
        X = data[['temp_anterior', 'umidade_anterior', 'pressao_anterior']]
        y = data['temperature']

        return X, y

    def train_model(self, X, y):
        """
        Treina o modelo com os dados de treinamento.
        Parâmetros:
        - X: DataFrame com as features para treinamento.
        - y: Série com os valores de temperatura.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #Normalizando dados
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        #Treinando modelo
        self.model.fit(X_train_scaled, y_train)
        #Avaliando o modelo
        y_pred = self.model.predict(X_test_scaled)
        #Calculando métricas
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {
            'mse': mse,
            'rmse': np.sqrt(mse), 
            'r2': r2
        }
    def predict(self, features):
        #Normalizando dados
        features_scaled = self.scaler.transform(features)
        #Fazendo previsões
        predictions = self.model.predict(features_scaled)
        return predictions

    def save_model(self, filepath):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, filepath)
    def load_model(self, filepath):
        saved_model = joblib.load(filepath)
        self.model = saved_model['model']
        self.scaler = saved_model['scaler']

#Exemplo de uso
if __name__ == '__main__':
    #Criando dados de exemplo
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = pd.DataFrame({
        'date': dates,
        'temperature': np.random.normal(loc=20, scale=5, size=len(dates)),
        'humidity': np.random.normal(loc=50, scale=10, size=len(dates)),    
        'pressure': np.random.normal(loc=1013, scale=5, size=len(dates)),
        'wind_speed': np.random.normal(loc=10, scale=5, size=len(dates)),
        'precipitation': np.random.normal(loc=0, scale=5, size=len(dates))
    })
    #Inicializar e treinar o modelo
    predictor = WeatherPredictor()
    X, y = predictor.prepare_data(data)
    metrics = predictor.train_model(X, y)
    print("Metricas do modelo: ")
    print(f'MSE: {metrics["mse"]}')
    print(f'RMSE: {metrics["rmse"]}')
    print(f'R2: {metrics["r2"]}')

    #Fazendo previsões
    new_data = X.iloc[-5:]
    predictions = predictor.predict(new_data)
    print("\nPrevisões para os proximos 5 dias:")
    for i,pred in enumerate(predictions, 1):
        print(f'Dia {i}: {pred:.1f}°C')

    #Salvando e carregando o modelo
    predictor.save_model('model.joblib')