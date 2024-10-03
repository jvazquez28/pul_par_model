import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def encode_categorical_features(data):
    """
    Codifica variables categóricas.
    """
    le = LabelEncoder()
    categorical_columns = ['Tipo de Zona', 'Barrio', 'Día de la Semana']
    for col in categorical_columns:
        data[col] = le.fit_transform(data[col])
    return data

def scale_numerical_features(data):
    """
    Escala características numéricas.
    """
    scaler = StandardScaler()
    numerical_columns = ['Latitud', 'Longitud', 'ETA (min)', 'Distancia (km)', 'Diámetro de Plazas (m)']
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    return data

def preprocess_data(data):
    """
    Aplica todas las etapas de preprocesamiento.
    """
    data = encode_categorical_features(data)
    data = scale_numerical_features(data)
    return data

if __name__ == '__main__':
    train_data = pd.read_csv('data/processed/train_data.csv', parse_dates=['Timestamp'])
    test_data = pd.read_csv('data/processed/test_data.csv', parse_dates=['Timestamp'])
    
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)
    
    train_data.to_csv('data/processed/train_data_preprocessed.csv', index=False)
    test_data.to_csv('data/processed/test_data_preprocessed.csv', index=False)
    
    print("Preprocesamiento completado.")