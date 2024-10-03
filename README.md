Aquí tienes un README combinado, que integra elementos de ambos documentos y proporciona una visión completa del proyecto de predicción de disponibilidad de aparcamiento:

---

# Proyecto de Predicción de Disponibilidad de Aparcamiento

## Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar una aplicación que prediga la disponibilidad de aparcamiento en varios barrios de Madrid, utilizando un modelo de Machine Learning (XGBoost). La aplicación integrará datos de reportes de usuarios y estimaciones de una API de Google Maps para ofrecer predicciones en tiempo real.

## Contenido

- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Generación de Datos Sintéticos](#generación-de-datos-sintéticos)
- [Modelo de Predicción](#modelo-de-predicción)
- [Instrucciones para la Ejecución](#instrucciones-para-la-ejecución)
- [Manejo de Datos](#manejo-de-datos)
- [Evaluación del Modelo](#evaluación-del-modelo)
- [Conclusiones](#conclusiones)

## Tecnologías Utilizadas

- **Python**: Lenguaje de programación utilizado para el desarrollo del modelo.
- **Pandas**: Biblioteca para la manipulación y análisis de datos.
- **NumPy**: Biblioteca para cálculos numéricos y manipulación de matrices.
- **XGBoost**: Biblioteca de algoritmos de boosting para clasificación y regresión.
- **Scikit-learn**: Biblioteca para Machine Learning, utilizada para la división del dataset y métricas de evaluación.
- **Optuna**: Biblioteca para la optimización de hiperparámetros.
- **Google Colab**: Entorno de ejecución en la nube para notebooks de Jupyter.

## Estructura del Proyecto

```
project_directory/
│
├── data/
│   ├── synthetic_data.csv   # Datos sintéticos generados
│
├── notebooks/
│   ├── data_generation.ipynb  # Notebook para la generación de datos sintéticos
│   ├── model_training.ipynb   # Notebook para el entrenamiento del modelo
│
└── requirements.txt           # Dependencias del proyecto
└── predictor.py               # Script para realizar predicciones en tiempo real
```

## Generación de Datos Sintéticos

Dado que inicialmente no se dispone de datos sobre si los conductores han aparcado, se generó un conjunto de datos sintéticos. Este conjunto de datos simula la información sobre viajes, incluyendo variables como:

- ID de Viaje
- Punto de Destino
- Punto de Salida
- Tiempo ETA (min)
- Distancia (km)
- Fecha y Hora de Llegada
- Tipo de Zona
- Día de la Semana
- Barrio
- Evento en el barrio (Sí/No)
- Densidad Vehicular Actual en el barrio
- Número de reportes de plazas disponibles en ese barrio
- He Aparcado (Sí/No)

### Código de Generación de Datos Sintéticos

```python
import pandas as pd
import numpy as np

# Listas de barrios según los distritos
barrios = {
    # ... (Definición de barrios)
}

# Crear una lista con todos los barrios
all_barrios = [barrio for barrio_list in barrios.values() for barrio in barrio_list]

# Generar valores simulados para las variables del dataset
n = 200000

data = {
    'ID de Viaje': np.arange(n),
    'Punto de Destino': np.random.choice(all_barrios, n),
    'Punto de Salida': np.random.choice(all_barrios, n),
    'Tiempo ETA (min)': np.random.randint(5, 30, n),
    'Distancia (km)': np.random.uniform(0.5, 10.0, n),
    'Fecha y Hora de Llegada': pd.date_range(start='2024-01-01', periods=n, freq='H'),
    'Tipo de Zona': np.random.choice(['Residencial', 'Comercial'], n),
    'Día de la Semana': np.random.choice(['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'], n),
    'Barrio': np.random.choice(all_barrios, n),
    'Evento en el barrio (Sí/No)': np.random.choice(['Sí', 'No'], n),
    'Densidad Vehicular Actual en el barrio': np.random.uniform(0, 100, n),
    'Número de reportes de plazas disponibles en ese barrio': np.random.randint(0, 10, n),
    'He Aparcado (Sí/No)': np.random.choice(['Sí', 'No'], n)
}

# Crear el DataFrame
df = pd.DataFrame(data)

# Guardar el DataFrame como CSV
df.to_csv('data/synthetic_data.csv', index=False)
```

## Modelo de Predicción

El modelo se construye utilizando **XGBoost**, que es conocido por su eficacia en tareas de clasificación y regresión. Se implementa un proceso de optimización de hiperparámetros utilizando **Optuna**.

### Proceso de Entrenamiento

1. **Inspección del Dataset**: Se revisan los tipos de datos y valores nulos.
2. **Manejo de Valores Nulos**: Se eliminan filas con valores nulos.
3. **Conversión de Fechas**: Se extraen características relevantes de las columnas de fecha y hora.
4. **Codificación de Variables Categóricas**: Se utilizan variables dummy para convertir categorías en variables numéricas.
5. **División del Dataset**: Se divide el conjunto de datos en entrenamiento y prueba.
6. **Optimización de Hiperparámetros**: Se utilizan métodos de Optuna para encontrar los mejores hiperparámetros para el modelo.

### Código de Entrenamiento del Modelo

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import optuna
from optuna import create_study

# Cargar datos
df = pd.read_csv('data/synthetic_data.csv')

# Preparar datos
X = df.drop(['He Aparcado (Sí/No)', 'ID de Viaje'], axis=1)
y = df['He Aparcado (Sí/No)'].map({'Sí': 1, 'No': 0})

# Dividir el dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo final
best_params = study.best_params
final_model = XGBClassifier(**best_params)
final_model.fit(X_train, y_train)

# Predicciones finales
final_y_pred = final_model.predict(X_test)

# Evaluación del Modelo final
final_accuracy = accuracy_score(y_test, final_y_pred)
print(f'Accuracy final: {final_accuracy:.2f}')
print(classification_report(y_test, final_y_pred))
```

## Instrucciones para la Ejecución

1. **Instala las dependencias**: Asegúrate de que todas las bibliotecas necesarias están instaladas. Puedes usar el siguiente comando:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecuta el Notebook**: Abre el notebook `data_generation.ipynb` para generar datos sintéticos y luego el notebook `model_training.ipynb` para entrenar el modelo.

3. **Ejecución del script**: Para realizar predicciones en tiempo real, ejecuta el siguiente comando:
   ```bash
   python predictor.py
   ```

## Manejo de Datos

Al principio de la aplicación, se recopilarán datos de distintas personas para confirmar si han aparcado. Estos datos se almacenarán y utilizarán para reentrenar el modelo una vez que se disponga de suficiente información real.

### Simulación de Datos Sintéticos

Además de la recopilación de datos de usuarios, se emplearán datos sintéticos para entrenar el modelo inicialmente, garantizando que el modelo pueda ofrecer predicciones desde el inicio de su implementación.

## Evaluación del Modelo

El modelo se evalúa mediante métricas como la precisión y el informe de clasificación. Se utilizarán estos resultados para realizar ajustes en el modelo y mejorar su rendimiento con el tiempo.

## Conclusiones

Este proyecto busca proporcionar una solución innovadora para la predicción de la disponibilidad de aparcamiento en Madrid, combinando técnicas de Machine Learning y datos de usuario en tiempo real. A medida que se obtengan más datos reales, se espera que el modelo mejore y ofrezca predicciones más precisas.
