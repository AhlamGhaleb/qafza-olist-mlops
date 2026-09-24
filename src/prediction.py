import logging

import mlflow
import mlflow.sklearn
import pandas as pd

from src.config import load_config


logger = logging.getLogger(__name__)


def load_model():
    """
    Load the registered trained model from MLflow.

    The model is loaded only for prediction.
    It is never trained again during inference.
    """
    try:
        config = load_config()

        tracking_uri = config["mlflow"]["tracking_uri"]
        registered_model_name = config["mlflow"]["registered_model_name"]
        model_alias = config["mlflow"]["model_alias"]

        mlflow.set_tracking_uri(tracking_uri)

        model_uri = (
            f"models:/{registered_model_name}@{model_alias}"
        )

        model = mlflow.sklearn.load_model(model_uri)

        logger.info(
            "Model loaded successfully from MLflow Registry: %s",
            model_uri,
        )

        return model

    except Exception:
        logger.exception(
            "Failed to load model from MLflow Registry."
        )
        raise


def predict(data: pd.DataFrame):
    """
    Predict delivery delay using the registered model.

    Returns:
        prediction: 0 for on-time/early, 1 for delayed
        probability: probability of delay
    """
    try:
        config = load_config()

        threshold = config["model"]["threshold"]

        model = load_model()

        probabilities = model.predict_proba(data)[:, 1]

        predictions = (probabilities >= threshold).astype(int)

        logger.info(
            "Prediction completed successfully. "
            "Input rows: %s, threshold: %s",
            len(data),
            threshold,
        )

        return predictions, probabilities

    except Exception:
        logger.exception(
            "Prediction failed."
        )
        raise