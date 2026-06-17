"""
generate_data.py
-----------------
Generador de un dataset mock para el problema de predicción de Open Rate
(apertura de notificaciones push).

El dataset NO es ruido aleatorio: el target `target_opened` se construye a
partir de un modelo logístico latente que combina las variables de forma
realista (p. ej. un `historical_open_rate` alto sube la probabilidad de
apertura, muchos días sin abrir la bajan, hay fatiga por exceso de pushes,
etc.). Esto garantiza que los modelos del Rol 3 encuentren señal real y no
métricas de azar.

Uso:
    python src/generate_data.py --n-rows 75000 --seed 42 --output data/raw/openrate.csv

Dependencias: numpy, pandas
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Catálogos de categorías y sus pesos (distribución no uniforme = más realista)
# ---------------------------------------------------------------------------
SITES = ["news", "ecommerce", "sports", "finance", "entertainment"]
SITE_P = [0.28, 0.26, 0.18, 0.12, 0.16]

CAMPAIGN_TYPES = ["promotional", "transactional", "reactivation", "newsletter"]
CAMPAIGN_P = [0.42, 0.23, 0.15, 0.20]

DEVICE_OS = ["android", "ios", "web"]
DEVICE_P = [0.55, 0.38, 0.07]

SEGMENTS = ["new", "engaged", "casual", "dormant"]
SEGMENT_P = [0.18, 0.30, 0.34, 0.18]

# ---------------------------------------------------------------------------
# Efectos (coeficientes) de cada categoría sobre el logit de apertura.
# Valores positivos -> más probable que el usuario abra.
# ---------------------------------------------------------------------------
CAMPAIGN_EFFECT = {
    "promotional": -0.15,   # mucha promo, fácil de ignorar
    "transactional": 0.65,  # avisos relevantes -> se abren más
    "reactivation": -0.35,  # se mandan a gente ya desenganchada
    "newsletter": 0.05,
}

SEGMENT_EFFECT = {
    "new": 0.10,
    "engaged": 0.90,
    "casual": -0.10,
    "dormant": -0.95,
}

DEVICE_EFFECT = {
    "android": 0.00,
    "ios": 0.12,
    "web": -0.20,
}


def hour_effect(hour: np.ndarray) -> np.ndarray:
    """Curva con dos picos suaves: mañana (~8h) y tarde-noche (~20h)."""
    morning = np.exp(-((hour - 8) ** 2) / 8.0)
    evening = np.exp(-((hour - 20) ** 2) / 10.0)
    return 0.6 * morning + 0.7 * evening - 0.3  # centrado alrededor de 0


def recency_effect(days: np.ndarray) -> np.ndarray:
    """Cuantos más días sin abrir, menos probable la próxima apertura."""
    return -0.45 * np.log1p(days)


def fatigue_effect(push_count: np.ndarray) -> np.ndarray:
    """Fatiga por sobre-impacto: muchos pushes históricos restan."""
    return -0.20 * np.log1p(push_count / 20.0)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate(n_rows: int, seed: int, n_users: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Usuarios y su "rasgo" latente de apertura --------------------------
    # Cada usuario tiene una propensión base; los eventos se reparten entre
    # ellos, de modo que un mismo user_id aparece en varias notificaciones.
    user_ids = np.array([f"U{idx:06d}" for idx in range(1, n_users + 1)])
    user_trait = rng.beta(2.0, 4.0, size=n_users)  # sesgada hacia abajo (~0.33)

    row_user = rng.integers(0, n_users, size=n_rows)
    user_id = user_ids[row_user]
    trait = user_trait[row_user]

    # historical_open_rate: el rasgo del usuario + ruido, recortado a [0,1]
    historical_open_rate = np.clip(trait + rng.normal(0, 0.06, n_rows), 0.0, 1.0)

    # --- Variables categóricas ---------------------------------------------
    site = rng.choice(SITES, size=n_rows, p=SITE_P)
    campaign_type = rng.choice(CAMPAIGN_TYPES, size=n_rows, p=CAMPAIGN_P)
    device_os = rng.choice(DEVICE_OS, size=n_rows, p=DEVICE_P)
    segment = rng.choice(SEGMENTS, size=n_rows, p=SEGMENT_P)

    # --- Variables temporales ----------------------------------------------
    hour_of_day = rng.integers(0, 24, size=n_rows)
    day_of_week = rng.integers(0, 7, size=n_rows)  # 0 = lunes ... 6 = domingo

    # --- Variables de comportamiento histórico -----------------------------
    historical_push_count = rng.poisson(40, size=n_rows) + 1
    # gente más enganchada tiende a haber abierto más recientemente
    days_scale = np.where(trait > 0.4, 5, 20)
    days_since_last_open = rng.poisson(days_scale).astype(int)

    # --- Construcción del logit latente ------------------------------------
    weekend = np.isin(day_of_week, [5, 6]).astype(float)

    z = (
        -0.55  # intercepto (calibra la tasa global de apertura)
        + 3.2 * (historical_open_rate - 0.30)
        + recency_effect(days_since_last_open)
        + hour_effect(hour_of_day)
        + fatigue_effect(historical_push_count)
        + np.vectorize(CAMPAIGN_EFFECT.get)(campaign_type)
        + np.vectorize(SEGMENT_EFFECT.get)(segment)
        + np.vectorize(DEVICE_EFFECT.get)(device_os)
        + 0.10 * weekend
        + rng.normal(0, 0.35, n_rows)  # ruido irreducible
    )

    p_open = sigmoid(z)
    target_opened = rng.binomial(1, p_open)

    df = pd.DataFrame(
        {
            "user_id": user_id,
            "site": site,
            "campaign_type": campaign_type,
            "device_os": device_os,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "historical_open_rate": np.round(historical_open_rate, 4),
            "historical_push_count": historical_push_count,
            "days_since_last_open": days_since_last_open,
            "segment": segment,
            "target_opened": target_opened.astype(int),
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el dataset mock de Open Rate.")
    parser.add_argument("--n-rows", type=int, default=75000, help="Número de filas (eventos de notificación).")
    parser.add_argument("--n-users", type=int, default=8000, help="Número de usuarios únicos.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad.")
    parser.add_argument("--output", type=str, default="data/raw/openrate.csv", help="Ruta del CSV de salida.")
    args = parser.parse_args()

    df = generate(args.n_rows, args.seed, args.n_users)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Resumen rápido por consola (útil como evidencia en el PR)
    rate = df["target_opened"].mean()
    print(f"[OK] Dataset generado en: {out_path}")
    print(f"     Filas: {len(df):,} | Usuarios únicos: {df['user_id'].nunique():,}")
    print(f"     Tasa de apertura (target=1): {rate:.3f}")
    print(f"     Tamaño aprox. en disco: {out_path.stat().st_size / 1_048_576:.2f} MB")


if __name__ == "__main__":
    main()
