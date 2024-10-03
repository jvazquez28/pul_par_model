# Pul-par

## Descripción
Pul-par es un proyecto de aprendizaje automático diseñado para optimizar el uso compartido de coches y el estacionamiento en entornos urbanos. Utiliza XGBoost para predecir la disponibilidad de estacionamiento basándose en diversas características como ubicación, tiempo, y características de la zona.

## Estructura del Dataset
El dataset utilizado contiene la siguiente información:
- ID de Reporte
- Latitud
- Longitud
- Tiempo ETA (min)
- Distancia (km)
- Diámetro de Plazas (m)
- Timestamp
- Tipo de Zona
- Plazas Disponibles
- Hora del Día
- Día de la Semana
- Barrio
- % de Encontrar Sitio

## Estructura del Proyecto
```
pul-par/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── saved_models/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
│
├── notebooks/
│
├── tests/
│
├── .gitignore
├── requirements.txt
├── setup.py
└── README.md
```

## Configuración
1. Clona este repositorio.
2. Crea un entorno virtual: `python -m venv env`
3. Activa el entorno virtual:
   - Windows: `env\Scripts\activate`
   - Unix o MacOS: `source env/bin/activate`
4. Instala los paquetes requeridos: `pip install -r requirements.txt`
5. Coloca tu archivo CSV de datos en la carpeta `data/raw/` con el nombre `parking_data.csv`, o especifica la ruta al ejecutar el script.

## Uso
1. Procesa los datos:
   ```
   python src/data/make_dataset.py [--input_filepath /ruta/a/tu/dataset.csv]
   ```
   Si no se especifica `--input_filepath`, se usará `data/raw/parking_data.csv` por defecto.

2. Preprocesa los datos:
   ```
   python src/data/preprocess.py
   ```

3. Construye las características:
   ```
   python src/features/build_features.py
   ```

4. Entrena el modelo:
   ```
   python src/models/train_model.py
   ```

5. Realiza predicciones:
   ```
   python src/models/predict_model.py
   ```

6. Visualiza los resultados:
   ```
   python src/visualization/visualize.py
   ```

## Pruebas
Ejecuta las pruebas usando pytest:
```
pytest tests/
```

Enlace del proyecto: https://github.com/hackatonf5g3/pul_par_model.git