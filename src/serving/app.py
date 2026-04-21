from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.pyfunc
import pandas as pd
import os

app = FastAPI(
    title="NYC Taxi Duration Predictor",
    description="Predicts trip duration in seconds",
    version="1.0"
)

MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/workspace.default.nyc-taxi-duration@champion"
)
model = mlflow.pyfunc.load_model(MODEL_URI)

class TripRequest(BaseModel):
    distance_km: float = Field(..., gt=0, example=3.2)
    hour_of_day: int = Field(..., ge=0, le=23, example=8)
    day_of_week: int = Field(..., ge=1, le=7, example=2)
    is_weekend: int = Field(..., ge=0, le=1, example=0)
    is_rush_hour: int = Field(..., ge=0, le=1, example=1)
    is_night: int = Field(..., ge=0, le=1, example=0)
    passenger_count: int = Field(..., ge=1, le=6, example=1)
    PULocationID: int = Field(..., example=161)
    DOLocationID: int = Field(..., example=236)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
def predict(trip: TripRequest):
    try:
        df = pd.DataFrame([trip.model_dump()])
        pred = float(model.predict(df)[0])
        return {
            "predicted_duration_seconds": round(pred, 1),
            "predicted_duration_minutes": round(pred / 60, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))