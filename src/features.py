import pandas as pd


NUMERIC_FEATURES = [
    "seller_customer_distance_km",
    "total_freight",
    "total_price",
    "num_items",
    "num_unique_products",
    "num_unique_sellers",
    "total_product_weight_g",
    "total_product_volume_cm3",
    "purchase_month",
    "purchase_weekday",
    "purchase_hour",
    "estimated_delivery_days",
]

CATEGORICAL_FEATURES = [
    "customer_state",
]

RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def create_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create the time features used in Notebook 5.
    """
    data = data.copy()

    data["order_purchase_timestamp"] = pd.to_datetime(
        data["order_purchase_timestamp"],
        errors="coerce",
    )

    data["order_estimated_delivery_date"] = pd.to_datetime(
        data["order_estimated_delivery_date"],
        errors="coerce",
    )

    data["purchase_month"] = (
        data["order_purchase_timestamp"].dt.month
    )

    data["purchase_weekday"] = (
        data["order_purchase_timestamp"].dt.weekday
    )

    data["purchase_hour"] = (
        data["order_purchase_timestamp"].dt.hour
    )

    data["estimated_delivery_days"] = (
        data["order_estimated_delivery_date"]
        - data["order_purchase_timestamp"]
    ).dt.total_seconds() / (24 * 60 * 60)

    return data


def select_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Select the 13 raw features required by the saved preprocessor.
    """
    missing_features = [
        column
        for column in RAW_FEATURES
        if column not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    return data[RAW_FEATURES].copy()