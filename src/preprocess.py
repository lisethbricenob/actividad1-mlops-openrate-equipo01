"""
preprocess.py
--------------
Etapa 2 del pipeline DVC: preprocesamiento del dataset crudo de Open Rate.

Tareas:
  1. Carga el CSV crudo generado por `generate_data.py`.
  2. Excluye la columna identificadora (`user_id`), que no es predictora.
  3. Codifica las variables categóricas con one-hot encoding.
  4. Separa train/test de forma estratificada (preserva el balance de clases).
  5. Guarda los conjuntos procesados en `data/processed/`.

Los parámetros se leen de `params.yaml` (sección `preprocess`), de modo que
DVC detecte cambios y re-ejecute la etapa solo cuando sea necesario.

Uso (vía pipeline):
    dvc repro preprocess
Uso (directo):
    python src/preprocess.py
"""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    params = load_params()
    raw_path = Path(params["data"]["raw_path"])
    cfg = params["preprocess"]

    out_dir = Path(cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carga ----------------------------------------------------------------
    df = pd.read_csv(raw_path)
    n_inicial = len(df)

    # 2. Excluir identificador -------------------------------------------------
    df = df.drop(columns=[cfg["id_col"]])

    # 3. One-hot de categóricas -------------------------------------------------
    df = pd.get_dummies(df, columns=cfg["categorical_cols"], drop_first=False, dtype=int)

    # 4. Separación train/test estratificada -----------------------------------
    target = cfg["target_col"]
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["test_size"],
        random_state=cfg["seed"],
        stratify=y,
    )

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    # 5. Guardado ----------------------------------------------------------------
    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    # Resumen por consola (evidencia para el PR)
    print(f"[OK] Preprocesamiento completado")
    print(f"     Registros de entrada : {n_inicial:,}")
    print(f"     Features finales     : {X.shape[1]} columnas (tras one-hot)")
    print(f"     Train: {len(train):,} filas -> {train_path}")
    print(f"     Test : {len(test):,} filas -> {test_path}")
    print(f"     Balance target (train/test): {y_train.mean():.3f} / {y_test.mean():.3f}")


if __name__ == "__main__":
    main()
