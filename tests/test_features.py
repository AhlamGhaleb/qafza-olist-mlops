import pandas as pd

from src.features import (
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