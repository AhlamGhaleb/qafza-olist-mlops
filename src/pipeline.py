import pandas as pd

from src.features import create_time_features, select_features
from src.preprocessing import transform_features
from src.prediction import predict
from src.validation import validate_input


def run_inference(data: pd.DataFrame):
    validation_result = validate_input(data)

    if not validation_result.valid:
        raise ValueError(
            "; ".join(validation_result.errors)
        )

    data_with_features = create_time_features(data)

    raw_features = select_features(data_with_features)

    processed_features = transform_features(raw_features)

    predictions, probabilities = predict(
        processed_features
    )

    return predictions, probabilities