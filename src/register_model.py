from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from src.config import load_config


def register_existing_model():
    config = load_config()

    model_path = Path(
        config["artifacts"]["model"]
    )

    model = joblib.load(model_path)

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    with mlflow.start_run() as run:

        mlflow.log_param(
            "model_type",
            type(model).__name__,
        )

        mlflow.log_param(
            "class_weight",
            "balanced",
        )

        mlflow.log_param(
            "threshold",
            config["model"]["threshold"],
        )

        mlflow.log_param(
            "source_artifact",
            str(model_path),
        )

        mlflow.sklearn.log_model(
            model,
            name="delivery_delay_model",
            registered_model_name=config[
                "mlflow"
            ]["registered_model_name"],
        )

        print("MLflow run:", run.info.run_id)