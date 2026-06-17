# MLOps_Maestria_USTA
Repositorio creado para la asignatura de MLOps de la Maestría en ciencia de datos de la Universidad Santo Tomás 2026.

## Reproducción del flujo de datos

1. Instalar dependencias:        poetry install
2. Descargar los datos:          poetry run dvc pull
3. Reproducir el pipeline:       poetry run dvc repro
   - Etapa `generate_data`: genera el dataset mock (35.000 eventos, semilla 42).
   - Etapa `preprocess`: codifica categóricas y separa train/test (80/20 estratificado).

Los parámetros del flujo se controlan desde `params.yaml`.
El diccionario de datos está en `data/data_dictionary.md`.
