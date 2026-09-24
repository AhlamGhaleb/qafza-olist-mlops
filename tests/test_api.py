import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app,config


client = TestClient(app)

SAMPLE_ORDER_PATH = Path("data/sample_order.json")


def load_sample_order():
    with SAMPLE_ORDER_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info():
    response = client.get("/model")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == config["mlflow"]["registered_model_name"]
    assert data["model_version"] == config["mlflow"]["model_alias"]
    assert data["threshold"] == config["model"]["threshold"]

def test_predict():
    order = load_sample_order()

    response = client.post("/predict", json=order)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0
    assert data["model_version"] == "production"


def test_predict_batch():
    order = load_sample_order()

    response = client.post(
        "/predict/batch",
        json={"orders": [order, order]},
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["predictions"]) == 2

    for result in data["predictions"]:
        assert result["prediction"] in [0, 1]
        assert 0.0 <= result["probability"] <= 1.0
        assert result["model_version"] == "production"

    assert (
        data["predictions"][0]["probability"]
        == data["predictions"][1]["probability"]
    )