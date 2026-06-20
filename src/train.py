from pathlib import Path
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import pickle


def load_params(path="params.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    params = load_params()

    target = params["preprocess"]["target_col"]
    seed = params["train"]["seed"]
    models_dir = Path(params["train"]["models_dir"])
    processed_dir = Path(params["preprocess"]["processed_dir"])

    models_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(processed_dir / "train.csv")
    test = pd.read_csv(processed_dir / "test.csv")

    X_train = train.drop(columns=[target])
    y_train = train[target]

    X_test = test.drop(columns=[target])
    y_test = test[target]


    tracking_dir = Path("mlruns").resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment("openrate-model-training")

    modelos = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=seed,
            n_jobs=-1,
        ),
    }

    resultados = []

    for nombre, modelo in modelos.items():
        with mlflow.start_run(run_name=nombre):
            modelo.fit(X_train, y_train)

            proba = modelo.predict_proba(X_test)[:, 1]
            pred = modelo.predict(X_test)

            roc_auc = roc_auc_score(y_test, proba)
            f1 = f1_score(y_test, pred)
            accuracy = accuracy_score(y_test, pred)

            mlflow.log_param("model_name", nombre)
            mlflow.log_param("seed", seed)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(modelo, artifact_path="model")

            resultados.append(
                {
                    "model": nombre,
                    "roc_auc": roc_auc,
                    "f1_score": f1,
                    "accuracy": accuracy,
                }
            )

    resultados_df = pd.DataFrame(resultados).sort_values("roc_auc", ascending=False)
    best_model_name = resultados_df.iloc[0]["model"]
    best_model = modelos[best_model_name]

    best_model.fit(X_train, y_train)

    model_path = models_dir / "best_model.pkl"
    metrics_path = models_dir / "metrics.csv"

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    resultados_df.to_csv(metrics_path, index=False)

    print("[OK] Entrenamiento completado")
    print(resultados_df)
    print(f"Mejor modelo: {best_model_name}")
    print(f"Modelo guardado en: {model_path}")
    print(f"Métricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()