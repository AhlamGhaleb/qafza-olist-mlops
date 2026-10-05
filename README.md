# Qafza Olist MLOps

## Project Overview

This project provides an inference system for predicting delivery delays in the Olist dataset.

The model predicts whether an order is likely to be delivered later than the estimated delivery date.

The project focuses on reliable model inference, data validation, model management, deployment, testing, logging, and monitoring.

The trained model from the previous stage is reused. No model training or retraining is performed during inference.

## Project Architecture

The inference process follows this flow:

```text
Input Order
    ↓
Input Validation
    ↓
Create Time Features
    ↓
Select Features
    ↓
Preprocessing
    ↓
MLflow Model Registry
    ↓
Prediction
    ↓
API / CLI Response
    ↓
Logging and Monitoring
```

### Main Technologies

* **Python** — main programming language
* **FastAPI** — REST API for model inference
* **MLflow** — experiment tracking and model registry
* **PostgreSQL** — MLflow backend database
* **Docker and Docker Compose** — application and service deployment
* **DVC** — data and artifact versioning
* **Great Expectations** — input data validation
* **Pytest** — automated testing
* **GitHub Actions** — continuous integration

## Model

The project uses a Logistic Regression model trained on the Olist delivery data.

Model configuration:

```text
LogisticRegression
class_weight = balanced
max_iter = 1000
random_state = 42
```

The prediction threshold is:

```text
0.67
```

The model is registered in MLflow as:

```text
qafza-olist-delivery-delay
```

The production alias is:

```text
production
```

## Features and Preprocessing

The inference pipeline uses the same feature engineering and preprocessing process used during model development.

The pipeline creates the following time features:

* purchase month
* purchase weekday
* purchase hour
* estimated delivery days

The model uses 13 raw features.

The preprocessing step transforms these features into 39 processed features.

The saved preprocessing artifact is reused during inference.

No fitting or retraining is performed during the inference process.

## Input Validation

Input data is validated before it is sent to the model.

The validation process checks:

* required columns
* numeric values
* missing values
* non-negative freight and price
* positive item, product, and seller counts
* date values
* Great Expectations rules

Invalid input is rejected before prediction.

The API returns HTTP `422` for invalid input.

## REST API

Start the API locally with:

```bash
uvicorn app.main:app --reload --port 8000
```

### Health Check

```text
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

### Model Information

```text
GET /model
```

Example response:

```json
{
  "model_name": "qafza-olist-delivery-delay",
  "model_version": "production",
  "threshold": 0.67
}
```

### Single Prediction

```text
POST /predict
```

This endpoint receives one order and returns the prediction, probability, and model version.

Example response:

```json
{
  "prediction": 0,
  "probability": 0.4429855557718791,
  "model_version": "production"
}
```

### Batch Prediction

```text
POST /predict/batch
```

This endpoint receives multiple orders and returns a prediction for each order.

## Command Line Interface

The project also supports inference from a JSON file.

Run:

```bash
python -m app.cli --input data/sample_order.json
```

Example output:

```json
{
  "prediction": 0,
  "probability": 0.4429855557718791,
  "model_version": "production"
}
```

## MLflow

MLflow is used for experiment tracking and model registration.

The registered model is:

```text
qafza-olist-delivery-delay
```

The production alias is:

```text
production
```

When Docker Compose is running, the API connects to MLflow through:

```text
http://mlflow:5000
```

The MLflow web interface is available locally at:

```text
http://127.0.0.1:5000
```

## Docker and Docker Compose

The project uses Docker Compose to run the main services.

The stack contains:

* PostgreSQL
* MLflow
* FastAPI API

Build and start the services with:

```bash
docker compose up -d --build
```

The API is available at:

```text
http://127.0.0.1:8000
```

MLflow is available at:

```text
http://127.0.0.1:5000
```

Stop the services with:

```bash
docker compose down
```

Persistent volumes are not removed by this command.

## DVC

DVC is used to track important datasets and machine learning artifacts.

The project tracks:

* labeled data
* training data
* validation data
* test data
* feature list
* preprocessing artifact
* trained model artifact

Check the DVC status with:

```bash
dvc status
```

## Great Expectations

Great Expectations is used to validate inference input data.

The project contains the `gx/` directory with the validation configuration and expectations.

Input validation is performed before the prediction step.

## Testing

The project uses Pytest for automated testing.

Run the tests with:

```bash
python -m pytest -q
```

The latest verified result is:

```text
15 passed, 1 warning
```

The warning is a Starlette/httpx deprecation warning and does not cause test failure.

The tests cover:

* API endpoints
* feature engineering
* feature selection
* preprocessing and inference
* input validation
* batch prediction
* target leakage prevention

## Logging

The application stores inference logs in:

```text
data/artifacts/logs/inference.log
```

The logs include:

* input data
* prediction
* probability
* response latency
* model version
* validation errors
* inference errors

These logs can later be compared with the actual delivery results.

## Monitoring

The API provides a simple monitoring endpoint:

```text
GET /metrics
```

The endpoint reports:

* total request count
* successful request count
* error count
* error rate
* prediction count
* prediction distribution
* average response latency

Example:

```json
{
  "request_count": 1,
  "successful_request_count": 1,
  "error_count": 0,
  "error_rate": 0.0,
  "prediction_count": 1,
  "prediction_distribution": {
    "0": 1,
    "1": 0
  },
  "average_latency_seconds": 6.89
}
```

### Monitoring Alerts

The following situations should be investigated:

* high or increasing error rate
* high API latency
* large changes in prediction distribution
* unusual increases in delayed-order predictions
* differences between predictions and actual delivery results

The prediction logs are kept so that future delivery results can be compared with the model predictions.

## Continuous Integration

GitHub Actions is used for continuous integration.

The workflow runs when code is pushed or when a pull request is created.

The CI process:

1. Checks out the repository.
2. Sets up Python 3.10.20.
3. Installs project dependencies.
4. Runs the Pytest test suite.

The workflow file is:

```text
.github/workflows/ci.yml
```

## Project Structure

```text
Qafza_Training/
├── app/
│   ├── main.py
│   ├── schemas.py
│   └── cli.py
├── config/
│   └── config.yaml
├── data/
│   └── artifacts/
├── gx/
├── notebooks/
├── requirements/
├── src/
├── tests/
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── Dockerfile.mlflow
├── compose.yaml
├── .dockerignore
├── .dvcignore
├── .gitignore
└── README.md
```

## Reproducibility

The project uses several tools to make the inference system reproducible:

* pinned Python and package versions
* DVC for data and artifact versioning
* MLflow Model Registry
* Docker Compose
* automated tests
* configuration files

The inference pipeline uses the existing trained model and does not retrain it.
