from dataclasses import dataclass

import pandas as pd


REQUIRED_COLUMNS = [
    "seller_customer_distance_km",
    "total_freight",
    "total_price",
    "num_items",
    "num_unique_products",
    "num_unique_sellers",
    "total_product_weight_g",
    "total_product_volume_cm3",
    "customer_state",
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
]


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]


def validate_input(data: pd.DataFrame) -> ValidationResult:
    errors = []

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        errors.append(
            f"Missing required columns: {missing_columns}"
        )

    if errors:
        return ValidationResult(
            valid=False,
            errors=errors,
        )

    numeric_columns = [
        "seller_customer_distance_km",
        "total_freight",
        "total_price",
        "num_items",
        "num_unique_products",
        "num_unique_sellers",
        "total_product_weight_g",
        "total_product_volume_cm3",
    ]

    for column in numeric_columns:
        if data[column].isna().all():
            errors.append(
                f"{column} contains only missing values."
            )

    if (data["total_freight"] < 0).any():
        errors.append("total_freight cannot be negative.")

    if (data["total_price"] < 0).any():
        errors.append("total_price cannot be negative.")

    if (data["num_items"] <= 0).any():
        errors.append("num_items must be greater than zero.")

    if (data["num_unique_products"] <= 0).any():
        errors.append(
            "num_unique_products must be greater than zero."
        )

    if (data["num_unique_sellers"] <= 0).any():
        errors.append(
            "num_unique_sellers must be greater than zero."
        )

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
    )