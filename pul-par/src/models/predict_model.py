import pandas as pd
import joblib

def load_model(filepath):
    """
    Carga el modelo guardado.
    """
    return joblib.load(filepath)

def make_predictions(model, data):
    """
    Realiza predicciones con el modelo.
    """
    X = data.drop(['Timestamp', 'ID de Reporte'], axis=1)
    return model.predict(X)

if __name__ == '__main__':
    model = load_model('models/saved_models/xgboost_model.joblib')
    test_data = pd.read_csv('data/processed/test_data_featured.csv')
    
    predictions = make_predictions(model, test_data)
    
    results = pd.DataFrame({
        'ID de Reporte': test_data['ID de Reporte'],
        'Predicción % de Encontrar Sitio': predictions
    })
    
    results.to_csv('data/processed/predictions.csv', index=False)
    print("Predicciones completadas y guardadas.")