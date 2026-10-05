import time
import logging
from threading import Lock

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


# Simple in-process monitoring metrics
metrics = {
    "request_count": 0,
    "error_count": 0,
    "prediction_count": 0,
    "prediction_0_count": 0,
    "prediction_1_count": 0,
    "total_latency_seconds": 0.0,
}

metrics_lock = Lock()


def record_metrics(
    latency: float,
    predictions=None,
    error: bool = False,
):
    with metrics_lock:
        metrics["request_count"] += 1

        if error:
            metrics["error_count"] += 1

        metrics["total_latency_seconds"] += latency

        if predictions is not None:
            prediction_list = [int(value) for value in predictions]

            metrics["prediction_count"] += len(prediction_list)
            metrics["prediction_0_count"] += prediction_list.count(0)
            metrics["prediction_1_count"] += prediction_list.count(1)


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    return {"status": "ok"}


@app.get(
    "/model",
    response_model=ModelInfoResponse,
)
def model_info():
    return {
        "model_name": config["mlflow"]["registered_model_name"],
        "model_version": config["mlflow"]["model_alias"],
        "threshold": config["model"]["threshold"],
    }


@app.get("/metrics")
def get_metrics():
    with metrics_lock:
        request_count = metrics["request_count"]
        error_count = metrics["error_count"]

        average_latency = (
            metrics["total_latency_seconds"] / request_count
            if request_count > 0
            else 0.0
        )

        error_rate = error_count / request_count if request_count > 0 else 0.0

        return {
            "request_count": request_count,
            "successful_request_count": request_count - error_count,
            "error_count": error_count,
            "error_rate": error_rate,
            "prediction_count": metrics["prediction_count"],
            "prediction_distribution": {
                "0": metrics["prediction_0_count"],
                "1": metrics["prediction_1_count"],
            },
            "average_latency_seconds": average_latency,
        }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_order(order: OrderRequest):
    start = time.perf_counter()

    try:
        data = pd.DataFrame([order.model_dump()])

        predictions, probabilities = run_inference(data)

        prediction = int(predictions[0])
        probability = float(probabilities[0])

        latency = time.perf_counter() - start

        record_metrics(
            latency=latency,
            predictions=predictions,
        )

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
            "model_version": config["mlflow"]["model_alias"],
        }

    except ValueError as exc:
        latency = time.perf_counter() - start
        record_metrics(
            latency=latency,
            error=True,
        )

        logger.error(
            "Prediction validation error | error=%s | latency=%.6fs",
            str(exc),
            latency,
        )

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        latency = time.perf_counter() - start
        record_metrics(
            latency=latency,
            error=True,
        )

        logger.exception(
            "Prediction failed | latency=%.6fs",
            latency,
        )

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
        data = pd.DataFrame([order.model_dump() for order in request.orders])

        predictions, probabilities = run_inference(data)

        results = [
            {
                "prediction": int(prediction),
                "probability": float(probability),
                "model_version": config["mlflow"]["model_alias"],
            }
            for prediction, probability in zip(
                predictions,
                probabilities,
            )
        ]

        latency = time.perf_counter() - start

        record_metrics(
            latency=latency,
            predictions=predictions,
        )

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
        latency = time.perf_counter() - start
        record_metrics(
            latency=latency,
            error=True,
        )

        logger.error(
            "Batch prediction validation error | error=%s | latency=%.6fs",
            str(exc),
            latency,
        )

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        latency = time.perf_counter() - start
        record_metrics(
            latency=latency,
            error=True,
        )

        logger.exception(
            "Batch prediction failed | latency=%.6fs",
            latency,
        )

        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed.",
        ) from exc
