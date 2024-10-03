import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def prepare_data(data, target_column='% de Encontrar Sitio'):
    """
    Prepara los datos para el entrenamiento.
    """
    X = data.drop([target_column, 'Timestamp', 'ID de Reporte'], axis=1)
    y = data[target_column]
    return X, y

def train_xgboost_model(X, y):
    """
    Entrena el modelo XGBoost.
    """
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    """
    Evalúa el modelo y muestra métricas.
    """
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

if __name__ == '__main__':
    data = pd.read_csv('data/processed/train_data_featured.csv')
    
    X, y = prepare_data(data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_xgboost_model(X_train, y_train)
    
    print("Evaluación en conjunto de validación:")
    evaluate_model(model, X_val, y_val)
    
    # Guardar el modelo
    joblib.dump(model, 'models/saved_models/xgboost_model.joblib')
    
    print("Modelo entrenado y guardado.")