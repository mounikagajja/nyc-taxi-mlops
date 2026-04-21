from fastapi.testclient import TestClient
import sys
import os
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

mock_model = mock.MagicMock()
mock_model.predict.return_value = [900.0]

with mock.patch("mlflow.pyfunc.load_model", return_value=mock_model):
    from src.serving.app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict():
    payload = {
        "distance_km": 3.2,
        "hour_of_day": 8,
        "day_of_week": 2,
        "is_weekend": 0,
        "is_rush_hour": 1,
        "is_night": 0,
        "passenger_count": 1,
        "PULocationID": 161,
        "DOLocationID": 236
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_duration_seconds" in response.json()

def test_invalid_input():
    response = client.post("/predict", json={"distance_km": -1})
    assert response.status_code == 422