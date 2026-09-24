from dataclasses import dataclass
from src.config import PROJECT_ROOT
import great_expectations as gx
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
    ge_errors = validate_with_great_expectations(data)
    errors.extend(ge_errors)

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

def validate_with_great_expectations(data: pd.DataFrame) -> list[str]:
    context = gx.get_context(
        context_root_dir=str(PROJECT_ROOT / "gx")
    )

    suite = context.suites.get("olist_inference_input_quality")

    source = context.data_sources.get("olist_inference")
    asset = source.get_asset("inference_requests")
    batch_definition = asset.get_batch_definition("inference_batch")

    ge_data = data.copy()

    date_columns = [
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
    ]

    for column in date_columns:
        ge_data[column] = pd.to_datetime(
            ge_data[column],
            errors="coerce",
        )

    batch = batch_definition.get_batch(
        batch_parameters={"dataframe": ge_data}
    )

    result = batch.validate(suite)

    if result.success:
        return []

    failed_expectations = [
        item.expectation_config.type
        for item in result.results
        if not item.success
    ]

    return [
        "Great Expectations validation failed: "
        + ", ".join(failed_expectations)
    ]