import time
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    OrderRequest,
    PredictionResponse,
)
from src.config import load_config
from src.logging_config import configure_logging
from src.pipeline import run_inference


config = load_config()

configure_logging(
    log_directory=config["logging"]["directory"],
    log_filename=config["logging"]["filename"],
    log_level=config["logging"]["level"],
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Qafza Olist Delivery Delay API",
    version=config["project"]["version"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    return {
        "status": "ok"
    }


@app.get(
    "/model",
    response_model=ModelInfoResponse,
)
def model_info():
    return {
        "model_name": config["mlflow"][
            "registered_model_name"
        ],
        "model_version": config["mlflow"][
            "model_alias"
        ],
        "threshold": config["model"]["threshold"],
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_order(order: OrderRequest):

    start = time.perf_counter()

    try:
        data = pd.DataFrame(
            [order.model_dump()]
        )

        predictions, probabilities = run_inference(
            data
        )

        prediction = int(predictions[0])
        probability = float(probabilities[0])

        latency = time.perf_counter() - start

        logger.info(
               "Prediction request | input=%s | output=%s | latency=%.6fs | model_version=%s",
                order.model_dump(),
                      {
                        "prediction": prediction,
                        "probability": probability,
                       },
                latency,
                config["mlflow"]["model_alias"],
        )

        return {
            "prediction": prediction,
            "probability": probability,
            "model_version": config["mlflow"][
                "model_alias"
            ],
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
)
def predict_batch(request: BatchPredictionRequest):

    start = time.perf_counter()

    try:
        data = pd.DataFrame(
            [order.model_dump() for order in request.orders]
        )

        predictions, probabilities = run_inference(
            data
        )

        results = [
            {
                "prediction": int(prediction),
                "probability": float(probability),
                "model_version": config["mlflow"][
                    "model_alias"
                ],
            }
            for prediction, probability in zip(
                predictions,
                probabilities,
            )
        ]

        latency = time.perf_counter() - start

        logger.info(
            "Batch prediction request | input=%s | output=%s | latency=%.6fs | model_version=%s",
            [order.model_dump() for order in request.orders],
            results,
            latency,
            config["mlflow"]["model_alias"],
        )

        return {
            "predictions": results,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed.",
        ) from exc