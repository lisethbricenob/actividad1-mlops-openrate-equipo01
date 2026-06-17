# Diccionario de datos — `openrate.csv`

**Caso:** predicción de *Open Rate* (apertura de notificaciones push).
**Origen:** dataset mock generado de forma reproducible con `src/generate_data.py` (semilla = 42).
**Volumen:** 35.000 registros (eventos de notificación) · 7.900 usuarios únicos · ~2,0 MB.
**Granularidad:** cada fila representa **un evento de notificación** enviado a un usuario.

Las variables siguen los lineamientos del punto 9 de la actividad.

## Variables

| # | Columna | Tipo | Descripción | Valores / Rango |
|---|---------|------|-------------|-----------------|
| 1 | `user_id` | texto | Identificador del usuario que recibe la notificación. Un mismo usuario puede aparecer en varios eventos. | `U000001` … `U008000` |
| 2 | `site` | categórica | Vertical o sitio que origina la notificación. | `news`, `ecommerce`, `sports`, `finance`, `entertainment` |
| 3 | `campaign_type` | categórica | Tipo de campaña de la notificación. | `promotional`, `transactional`, `reactivation`, `newsletter` |
| 4 | `device_os` | categórica | Sistema operativo / plataforma del dispositivo. | `android`, `ios`, `web` |
| 5 | `hour_of_day` | entero | Hora del día en que se envía la notificación. | `0`–`23` |
| 6 | `day_of_week` | entero | Día de la semana del envío. | `0`–`6` (0 = lunes … 6 = domingo) |
| 7 | `historical_open_rate` | flotante | Tasa histórica de apertura del usuario (propensión a abrir). Predictor de mayor peso. | `0.0`–`1.0` |
| 8 | `historical_push_count` | entero | Número de notificaciones que el usuario ha recibido históricamente. | entero positivo |
| 9 | `days_since_last_open` | entero | Días transcurridos desde la última apertura del usuario. | entero ≥ 0 |
| 10 | `segment` | categórica | Segmento de ciclo de vida del usuario. | `new`, `engaged`, `casual`, `dormant` |
| 11 | `target_opened` | binaria | **Variable objetivo.** Indica si el usuario abrió la notificación. | `0` (no abrió) · `1` (abrió) |

## Notas para el modelado

- **Variable objetivo:** `target_opened`. Problema de **clasificación binaria**.
- **Balance de clases:** ~21,5 % de la clase positiva (`target_opened = 1`). Desbalance moderado; no requiere remuestreo obligatorio, pero conviene reportar métricas más allá de *accuracy* (p. ej. *F1*, *ROC-AUC*).
- **Variables categóricas a codificar:** `site`, `campaign_type`, `device_os`, `segment` (p. ej. *one-hot* u *ordinal encoding*) antes de entrenar.
- **`user_id`** es un identificador, **no** una variable predictora: debe excluirse del entrenamiento.
- **Valores faltantes:** ninguno (el dataset es sintético y completo).
- **Señal incorporada (validación):** la apertura crece de forma coherente con el comportamiento esperado, p. ej. segmento `engaged` (~33 %) frente a `dormant` (~9 %), y la tasa de apertura sube monótonamente por cuartil de `historical_open_rate` (de ~9 % a ~41 %).

## Reproducibilidad

```bash
python src/generate_data.py --n-rows 35000 --seed 42 --output data/raw/openrate.csv
```

El uso de una semilla fija garantiza que el dataset sea idéntico en cada ejecución, condición necesaria para la trazabilidad del flujo MLOps.
