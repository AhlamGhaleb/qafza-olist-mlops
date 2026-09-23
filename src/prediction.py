import joblib
import logging
import pandas as pd

from src.config import load_config


logger = logging.getLogger(__name__)


def load_model():
    """
    Load the trained model saved from Notebook 6.

    The model is loaded only for prediction.
    It is never trained again during inference.
    """
    try:
        config = load_config()
        model_path = config["artifacts"]["model"]

        model = joblib.load(model_path)

        logger.info(
            "Model loaded successfully from: %s",
            model_path,
        )

        return model

    except FileNotFoundError:
        logger.error(
            "Model file not found."
        )
        raise

    except Exception:
        logger.exception(
            "Failed to load the model."
        )
        raise


def predict(data: pd.DataFrame):
    """
    Predict delivery delay using the saved model.

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