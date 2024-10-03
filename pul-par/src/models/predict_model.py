import pandas as pd
import joblib
import os

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_model(filepath):
    return joblib.load(filepath)

def make_predictions(model, data):
    # Asegurarse de que solo usamos las características que el modelo espera
    expected_features = model.get_booster().feature_names
    data = data.reindex(columns=expected_features, fill_value=0)
    return model.predict(data)

if __name__ == '__main__':
    model = load_model(os.path.join(PROJECT_DIR, 'models', 'saved_models', 'xgboost_model.joblib'))
    test_data = pd.read_csv(os.path.join(PROJECT_DIR, 'data', 'processed', 'test_data_featured.csv'))
    
    # Imprimir las columnas para depuración
    print("Columnas en los datos de prueba:", test_data.columns)
    print("Características esperadas por el modelo:", model.get_booster().feature_names)
    
    # Asumimos que el ID de Reporte o ID de Viaje es la primera columna, ajusta si es necesario
    id_column = test_data.columns[0]
    
    # Eliminar la columna objetivo si está presente
    features = test_data.drop(['% de Encontrar Sitio'], axis=1, errors='ignore')
    
    # Hacer predicciones
    predictions = make_predictions(model, features)
    
    results = pd.DataFrame({
        'ID': test_data[id_column],
        'Predicción % de Encontrar Sitio': predictions
    })
    
    results.to_csv(os.path.join(PROJECT_DIR, 'data', 'processed', 'predictions.csv'), index=False)
    print("Predicciones completadas y guardadas.")