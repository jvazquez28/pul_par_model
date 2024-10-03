import pandas as pd
import joblib  # Para guardar y cargar el modelo
import time

# Cargar el modelo
def load_model(file_path):
    return joblib.load(file_path)

# Hacer predicciones
def make_prediction(model, input_data):
    return model.predict(input_data)

# Función principal para ejecutar el predictor
def main():
    model_path = 'model.pkl'  # Ruta del modelo guardado
    
    # Cargar el modelo
    model = load_model(model_path)
    
    # Realizar predicciones cada minuto
    while True:
        # Simulación de nuevos datos de entrada (esto debería venir de tu app)
        new_data = {
            # Rellena con los datos necesarios
            # 'Punto de Destino': ...,
            # 'Punto de Salida': ...,
            # etc.
        }
        new_df = pd.DataFrame([new_data])  # Convertir el diccionario a DataFrame
        
        # Preprocesar los datos antes de hacer la predicción
        new_df_encoded = preprocess_data(new_df)  # Debes definir esta función
        
        # Hacer predicción
        prediction = make_prediction(model, new_df_encoded)
        print(f'Predicción: {prediction[0]}')  # 0 = No, 1 = Sí
        
        # Esperar un minuto antes de la siguiente predicción
        time.sleep(60)

if __name__ == "__main__":
    main()