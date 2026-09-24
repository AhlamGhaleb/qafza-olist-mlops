import pandas as pd

from src.features import create_time_features, select_features
from src.preprocessing import transform_features
from src.prediction import predict
from src.validation import validate_input
from src.pipeline import run_inference

TEST_DATA_PATH = "data/artifacts/03_test1.parquet"


def load_raw_test_data():
    data = pd.read_parquet(TEST_DATA_PATH)
    return data.drop(columns=["is_delayed"])


def test_transform_features_matches_task2_artifact():
    data = load_raw_test_data()

    data_with_features = create_time_features(data)
    raw_features = select_features(data_with_features)

    processed_features = transform_features(raw_features)

    expected_features = pd.read_parquet(
        "data/artifacts/features/05_X_test.parquet"
    )

    assert processed_features.shape == expected_features.shape
    assert list(processed_features.columns) == list(
        expected_features.columns
    )

    pd.testing.assert_frame_equal(
        processed_features.reset_index(drop=True),
        expected_features.reset_index(drop=True),
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )

def test_predict_output():
    processed_features = pd.read_parquet(
        "data/artifacts/features/05_X_test.parquet"
    )

    predictions, probabilities = predict(processed_features)

    assert len(predictions) == len(processed_features)
    assert len(probabilities) == len(processed_features)

    assert set(predictions).issubset({0, 1})
    assert ((probabilities >= 0) & (probabilities <= 1)).all()

def test_validate_input_accepts_valid_data():
    data = load_raw_test_data().head(1)

    result = validate_input(data)

    assert result.valid is True
    assert result.errors == []


def test_validate_input_rejects_missing_column():
    data = load_raw_test_data().head(1)
    data = data.drop(columns=["total_price"])

    result = validate_input(data)

    assert result.valid is False
    assert "Missing required columns" in result.errors[0]


def test_validate_input_rejects_negative_freight():
    data = load_raw_test_data().head(1)
    data.loc[data.index[0], "total_freight"] = -1

    result = validate_input(data)

    assert result.valid is False
    assert "total_freight cannot be negative." in result.errors


def test_validate_input_rejects_invalid_num_items():
    data = load_raw_test_data().head(1)
    data.loc[data.index[0], "num_items"] = 0

    result = validate_input(data)

    assert result.valid is False
    assert "num_items must be greater than zero." in result.errors

def test_run_inference():
    data = load_raw_test_data().head(10)

    predictions, probabilities = run_inference(data)

    assert len(predictions) == 10
    assert len(probabilities) == 10

    assert set(predictions).issubset({0, 1})
    assert ((probabilities >= 0) & (probabilities <= 1)).all()