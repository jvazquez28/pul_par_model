import pandas as pd
from sklearn.model_selection import train_test_split
import os

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'raw', 'parking_data.csv')

def load_data(filepath):
    """
    Carga los datos del CSV y maneja flexiblemente la columna de tiempo.
    """
    # Primero, leemos solo la primera fila para obtener los nombres de las columnas
    df_temp = pd.read_csv(filepath, nrows=0)
    
    # Buscamos una columna que podría contener información de tiempo
    time_columns = [col for col in df_temp.columns if 'time' in col.lower() or 'date' in col.lower()]
    
    if time_columns:
        # Si encontramos una columna de tiempo, la parseamos como fecha
        return pd.read_csv(filepath, parse_dates=time_columns)
    else:
        # Si no encontramos una columna de tiempo, cargamos el CSV sin parsear fechas
        print("Advertencia: No se encontró una columna de tiempo para parsear.")
        return pd.read_csv(filepath)

def split_data(data, test_size=0.2, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    """
    # Asegúrate de que 'Barrio' está en las columnas antes de usarlo para estratificar
    if 'Barrio' in data.columns:
        return train_test_split(data, test_size=test_size, random_state=random_state, stratify=data['Barrio'])
    else:
        print("Advertencia: 'Barrio' no está en las columnas. La división no será estratificada.")
        return train_test_split(data, test_size=test_size, random_state=random_state)

if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"El archivo de datos no se encuentra en {DATA_PATH}")
    
    data = load_data(DATA_PATH)
    
    print("Columnas en el dataset:")
    print(data.columns)
    
    train_data, test_data = split_data(data)
    
    train_data.to_csv(os.path.join(PROJECT_DIR, 'data', 'processed', 'train_data.csv'), index=False)
    test_data.to_csv(os.path.join(PROJECT_DIR, 'data', 'processed', 'test_data.csv'), index=False)
    
    print(f"Datos cargados desde {DATA_PATH}")
    print("Datos procesados y guardados en data/processed/")