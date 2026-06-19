# MLOps_Maestria_USTA

Repositorio creado para la asignatura de MLOps de la Maestría en Ciencia de Datos de la Universidad Santo Tomás 2026.

## Contexto del proyecto

Mini flujo MLOps para la predicción de *Open Rate* (apertura de notificaciones push),
construido como Actividad Integradora 1 del módulo Fundamentos de MLOps y Gestión del Ciclo de Vida.

## Equipo

Ver [`TEAM.md`](./TEAM.md) para roles y responsabilidades de cada integrante.

## Reproducción del flujo de datos

1. **Instalar dependencias:**
```bash
   poetry install
```

2. **Descargar los datos versionados con DVC:**
```bash
   poetry run dvc pull
```

3. **Reproducir el pipeline completo:**
```bash
   poetry run dvc repro
```

   El pipeline ejecuta las siguientes etapas, en orden:

   * **`generate_data`**: genera el dataset mock (35.000 eventos, 8.000 usuarios, semilla 42).
   * **`preprocess`**: codifica variables categóricas y separa train/test (80/20).
   * **`train`**: entrena y compara dos modelos de clasificación (Regresión Logística y Random Forest), selecciona el mejor según ROC-AUC, y guarda `models/best_model.pkl` y `models/metrics.csv`.

4. **Entrenar el modelo de forma individual** (alternativa a `dvc repro` si solo se quiere correr esta etapa):
```bash
   poetry run python src/train.py
```

5. **Visualizar los experimentos registrados en MLflow:**
```bash
   poetry run mlflow ui
```
   Luego abrir [http://localhost:5000](http://localhost:5000) en el navegador para comparar las corridas, métricas (ROC-AUC, F1-score, accuracy) y los modelos registrados de cada experimento.

## Parámetros y datos

Los parámetros del flujo se controlan desde [`params.yaml`](./params.yaml).

El diccionario de datos completo está en [`data/data_dictionary.md`](./data/data_dictionary.md), 
e incluye la descripción de las 11 variables del dataset `openrate.csv` (predictoras y la 
variable objetivo `target_opened`), notas de balance de clases y la instrucción de 
reproducibilidad del dataset mock.

## Estructura del repositorio

```
actividad1-mlops-openrate-equipo01/
├── data/
├── src/
├── models/
├── outputs/
├── notebooks/
├── mlruns/
├── pyproject.toml
├── poetry.lock
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── README.md
├── TEAM.md
└── .gitignore
```
