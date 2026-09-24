import pandas as pd
import pytest

from src.features import (
    RAW_FEATURES,
    create_time_features,
    select_features,
)

def test_create_time_features():
    data = pd.DataFrame(
        {
            "order_purchase_timestamp": [
                "2020-01-10 10:00:00"
            ],
            "order_estimated_delivery_date": [
                "2020-01-15 10:00:00"
            ],
        }
    )

    result = create_time_features(data)

    assert result.loc[0, "purchase_month"] == 1
    assert result.loc[0, "purchase_weekday"] == 4
    assert result.loc[0, "purchase_hour"] == 10
    assert result.loc[0, "estimated_delivery_days"] == 5

def test_select_features():
    data = pd.DataFrame(
        {
            feature: [index]
            for index, feature in enumerate(RAW_FEATURES)
        }
    )

    data["extra_column"] = [999]

    result = select_features(data)

    assert list(result.columns) == RAW_FEATURES
    assert result.shape == (1, len(RAW_FEATURES))
    assert "extra_column" not in result.columns

def test_select_features_missing_feature():
    data = pd.DataFrame(
        {
            feature: [index]
            for index, feature in enumerate(RAW_FEATURES)
            if feature != RAW_FEATURES[0]
        }
    )

    with pytest.raises(ValueError, match="Missing required features"):
        select_features(data)