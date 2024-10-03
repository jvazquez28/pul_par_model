import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_feature_importance(model, X):
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Top 20 Características Más Importantes')
    plt.tight_layout()
    
    file_path = os.path.join(PROJECT_DIR, 'reports', 'figures', 'feature_importance.png')
    ensure_dir(file_path)
    plt.savefig(file_path)
    plt.close()

def plot_predictions_vs_actual(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Valor Real')
    plt.ylabel('Predicción')
    plt.title('Predicciones vs Valores Reales')
    plt.tight_layout()
    
    file_path = os.path.join(PROJECT_DIR, 'reports', 'figures', 'predictions_vs_actual.png')
    ensure_dir(file_path)
    plt.savefig(file_path)
    plt.close()

def plot_availability_by_time(data):
    if 'Hour' in data.columns and '% de Encontrar Sitio' in data.columns:
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='Hour', y='% de Encontrar Sitio', data=data)
        plt.title('Disponibilidad de Estacionamiento por Hora del Día')
        plt.xlabel('Hora del Día')
        plt.ylabel('% de Encontrar Sitio')
        plt.tight_layout()
        
        file_path = os.path.join(PROJECT_DIR, 'reports', 'figures', 'availability_by_time.png')
        ensure_dir(file_path)
        plt.savefig(file_path)
        plt.close()

if __name__ == '__main__':
    model_path = os.path.join(PROJECT_DIR, 'models', 'saved_models', 'xgboost_model.joblib')
    data_path = os.path.join(PROJECT_DIR, 'data', 'processed', 'train_data_featured.csv')

    if not os.path.exists(model_path):
        print(f"El modelo no se encuentra en {model_path}")
        exit(1)

    if not os.path.exists(data_path):
        print(f"Los datos no se encuentran en {data_path}")
        exit(1)

    model = joblib.load(model_path)
    data = pd.read_csv(data_path)
    
    target_column = '% de Encontrar Sitio'
    if target_column not in data.columns:
        print(f"La columna '{target_column}' no está en el conjunto de datos.")
        print("Columnas disponibles:", data.columns)
        exit(1)

    X = data.drop([target_column], axis=1)
    y = data[target_column]
    
    plot_feature_importance(model, X)
    
    y_pred = model.predict(X)
    plot_predictions_vs_actual(y, y_pred)
    
    plot_availability_by_time(data)
    
    print("Visualizaciones creadas en reports/figures/")