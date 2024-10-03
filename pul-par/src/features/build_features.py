import pandas as pd
import numpy as np

def create_time_features(data):
    """
    Crea características basadas en tiempo.
    """
    data['Hour'] = data['Timestamp'].dt.hour
    data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
    data['Month'] = data['Timestamp'].dt.month
    data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Características cíclicas para la hora
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour']/24)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour']/24)
    
    return data

def create_location_features(data):
    """
    Crea características basadas en ubicación.
    """
    # Ejemplo: distancia al centro de la ciudad (ajusta las coordenadas según tu ciudad)
    city_center = (40.4168, -3.7038)  # Madrid
    data['DistanceToCenter'] = np.sqrt(
        (data['Latitud'] - city_center[0])**2 + (data['Longitud'] - city_center[1])**2
    )
    return data

def build_features(data):
    """
    Aplica todas las funciones de creación de características.
    """
    data = create_time_features(data)
    data = create_location_features(data)
    return data

if __name__ == '__main__':
    train_data = pd.read_csv('data/processed/train_data_preprocessed.csv', parse_dates=['Timestamp'])
    test_data = pd.read_csv('data/processed/test_data_preprocessed.csv', parse_dates=['Timestamp'])
    
    train_data = build_features(train_data)
    test_data = build_features(test_data)
    
    train_data.to_csv('data/processed/train_data_featured.csv', index=False)
    test_data.to_csv('data/processed/test_data_featured.csv', index=False)
    
    print("Ingeniería de características completada.")