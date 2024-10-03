import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def plot_feature_importance(model, X):
    """
    Visualiza la importancia de las características.
    """
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Top 20 Características Más Importantes')
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importance.png')
    plt.close()

def plot_predictions_vs_actual(y_true, y_pred):
    """
    Compara predicciones con valores reales.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Valor Real')
    plt.ylabel('Predicción')
    plt.title('Predicciones vs Valores Reales')
    plt.tight_layout()
    plt.savefig('reports/figures/predictions_vs_actual.png')
    plt.close()

def plot_availability_by_time(data):
    """
    Visualiza la disponibilidad de estacionamiento por hora del día.
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Hour', y='% de Encontrar Sitio', data=data)
    plt.title('Disponibilidad de Estacionamiento por Hora del Día')
    plt.xlabel('Hora del Día')
    plt.ylabel('% de Encontrar Sitio')
    plt.tight_layout()
    plt.savefig('reports/figures/availability_by_time.png')
    plt.close()

if __name__ == '__main__':
    model = joblib.load('models/saved_models/xgboost_model.joblib')
    data = pd.read_csv('data/processed/train_data_featured.csv')
    
    X = data.drop(['% de Encontrar Sitio', 'Timestamp', 'ID de Reporte'], axis=1)
    y = data['% de Encontrar Sitio']
    
    plot_feature_importance(model, X)
    
    y_pred = model.predict(X)
    plot_predictions_vs_actual(y, y_pred)
    
    plot_availability_by_time(data)
    
    print("Visualizaciones creadas en reports/figures/")