import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.pipeline import run_inference


def main():
    parser = argparse.ArgumentParser(
        description="Qafza Olist delivery delay inference CLI."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSON file.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        parser.error(
            f"Input file not found: {input_path}"
        )

    try:
        with open(
            input_path,
            "r",
            encoding="utf-8",
        ) as file:
            input_data = json.load(file)

        data = pd.DataFrame([input_data])

        predictions, probabilities = run_inference(data)

        config = load_config()

        result = {
            "prediction": int(predictions[0]),
            "probability": float(probabilities[0]),
            "model_version": config["mlflow"]["model_alias"],
        }

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

    except ValueError as exc:
        parser.error(str(exc))

    except Exception as exc:
        parser.error(
            f"Inference failed: {exc}"
        )


if __name__ == "__main__":
    main()