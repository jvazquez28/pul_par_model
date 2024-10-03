import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Definir la ruta al dataset
DATA_PATH = os.path.join('data', 'raw', 'parking_data.csv')

def load_data(filepath):
    """
    Carga los datos del CSV.
    """
    return pd.read_csv(filepath, parse_dates=['Timestamp'])

def split_data(data, test_size=0.2, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    """
    return train_test_split(data, test_size=test_size, random_state=random_state, stratify=data['Barrio'])

if __name__ == '__main__':
    # Verificar si el archivo existe
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"El archivo de datos no se encuentra en {DATA_PATH}")
    
    # Cargar datos
    data = load_data(DATA_PATH)
    
    # Dividir datos
    train_data, test_data = split_data(data)
    
    # Guardar datos procesados
    train_data.to_csv('data/processed/train_data.csv', index=False)
    test_data.to_csv('data/processed/test_data.csv', index=False)
    
    print(f"Datos cargados desde {DATA_PATH}")
    print("Datos procesados y guardados en data/processed/")