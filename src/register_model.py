from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from src.config import load_config


def register_existing_model():
    config = load_config()

    mlflow.set_tracking_uri(
        config["mlflow"]["tracking_uri"]
    )

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

        client = MlflowClient()

        registered_model_name = config["mlflow"]["registered_model_name"]
        model_alias = config["mlflow"]["model_alias"]

        model_versions = client.search_model_versions(
            f"name='{registered_model_name}'"
        )

        current_version = next(
            version
            for version in model_versions
            if version.run_id == run.info.run_id
        )

        client.set_registered_model_alias(
            registered_model_name,
            model_alias,
            current_version.version,
        )

        print(
            f"Model version {current_version.version} "
            f"assigned to alias '{model_alias}'"
        )

        print("MLflow run:", run.info.run_id)
