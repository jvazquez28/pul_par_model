import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def encode_categorical_features(data):
    le = LabelEncoder()
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        data[col] = le.fit_transform(data[col].astype(str))
    return data

def scale_numerical_features(data):
    scaler = StandardScaler()
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    return data

def preprocess_data(data):
    data = encode_categorical_features(data)
    data = scale_numerical_features(data)
    return data

if __name__ == '__main__':
    train_data = pd.read_csv(os.path.join(PROJECT_DIR, 'data', 'processed', 'train_data.csv'))
    test_data = pd.read_csv(os.path.join(PROJECT_DIR, 'data', 'processed', 'test_data.csv'))
    
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)
    
    train_data.to_csv(os.path.join(PROJECT_DIR, 'data', 'processed', 'train_data_preprocessed.csv'), index=False)
    test_data.to_csv(os.path.join(PROJECT_DIR, 'data', 'processed', 'test_data_preprocessed.csv'), index=False)
    
    print("Preprocesamiento completado.")