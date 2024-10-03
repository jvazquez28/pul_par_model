import pandas as pd
import numpy as np
import os

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def create_time_features(data):
    if 'Timestamp' in data.columns:
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        data['Hour'] = data['Timestamp'].dt.hour
        data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
        data['Month'] = data['Timestamp'].dt.month
        data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
    return data

def create_distance_features(data):
    if 'Latitud' in data.columns and 'Longitud' in data.columns:
        # Ejemplo: distancia al centro de la ciudad (ajusta las coordenadas según tu ciudad)
        city_center = (40.4168, -3.7038)  # Madrid
        data['DistanceToCenter'] = np.sqrt(
            (data['Latitud'] - city_center[0])**2 + (data['Longitud'] - city_center[1])**2
        )
    return data

def build_features(data):
    data = create_time_features(data)
    data = create_distance_features(data)
    return data

if __name__ == '__main__':
    train_data = pd.read_csv(os.path.join(PROJECT_DIR, 'data', 'processed', 'train_data_preprocessed.csv'))
    test_data = pd.read_csv(os.path.join(PROJECT_DIR, 'data', 'processed', 'test_data_preprocessed.csv'))
    
    train_data = build_features(train_data)
    test_data = build_features(test_data)
    
    train_data.to_csv(os.path.join(PROJECT_DIR, 'data', 'processed', 'train_data_featured.csv'), index=False)
    test_data.to_csv(os.path.join(PROJECT_DIR, 'data', 'processed', 'test_data_featured.csv'), index=False)
    
    print("Ingeniería de características completada.")