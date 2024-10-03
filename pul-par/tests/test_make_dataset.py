import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def plot_feature_importance(model, X):
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importance.png')

def plot_predictions_vs_actual(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual')
    plt.tight_layout()
    plt.savefig('reports/figures/predictions_vs_actual.png')

if __name__ == '__main__':
    model = joblib.load('models/saved_models/xgboost_model.joblib')
    data = pd.read_csv('data/processed/train_data_featured.csv')
    X = data.drop('route_efficiency_score', axis=1)
    y = data['route_efficiency_score']
    
    plot_feature_importance(model, X)
    
    y_pred = model.predict(X)
    plot_predictions_vs_actual(y, y_pred)
    
    print("Visualizations created.")