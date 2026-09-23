import joblib
import logging
import pandas as pd

from src.config import load_config


logger = logging.getLogger(__name__)


def load_preprocessor():
    """
    Load the fitted preprocessor saved from Notebook 5.
    """
    try:
        config = load_config()
        preprocessor_path = config["artifacts"]["preprocessor"]

        preprocessor = joblib.load(preprocessor_path)

        logger.info(
            "Preprocessor loaded successfully from: %s",
            preprocessor_path,
        )

        return preprocessor

    except FileNotFoundError:
        logger.error(
            "Preprocessor file not found."
        )
        raise

    except Exception:
        logger.exception(
            "Failed to load the preprocessor."
        )
        raise


def transform_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw features using the fitted preprocessor.

    The preprocessor is only used for transformation.
    It is never fitted again during inference.
    """
    try:
        preprocessor = load_preprocessor()

        transformed = preprocessor.transform(data)

        feature_names = preprocessor.get_feature_names_out()

        result = pd.DataFrame(
            transformed,
            columns=feature_names,
            index=data.index,
        )

        logger.info(
            "Features transformed successfully. Input shape: %s, output shape: %s",
            data.shape,
            result.shape,
        )

        return result

    except Exception:
        logger.exception(
            "Feature transformation failed."
        )
        raise