# src/app.py
"""
FastAPI application.

Defines the HTTP API surface:
- /health
- /predict
- /predict/batch
- /stats (for Grafana)
"""

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .predict import predict_batch, predict_single
from .monitoring import build_stats_payload


class PredictRequest(BaseModel):
    text: str = Field(..., description="Raw social media text to analyze.")


class PredictBatchRequest(BaseModel):
    texts: List[str] = Field(
        ..., description="List of texts to analyze in a single request."
    )


class PredictResponse(BaseModel):
    label: str
    score: float
    probabilities: dict


class PredictBatchResponse(BaseModel):
    results: List[PredictResponse]


app = FastAPI(
    title="MachineInnovators – Online Reputation Monitoring",
    description=(
        "Sentiment analysis API using a pretrained RoBERTa model from Hugging Face. "
        "Designed for MLOps workflows with monitoring and future retraining."
    ),
    version="1.0.0",
)


@app.get("/health")
def healthcheck() -> dict:
    """Basic health endpoint used for liveness checks."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict sentiment for a single piece of text."""
    result = predict_single(request.text)
    return PredictResponse(**result)


@app.post("/predict/batch", response_model=PredictBatchResponse)
def predict_batch_endpoint(request: PredictBatchRequest) -> PredictBatchResponse:
    """Predict sentiment for a list of texts."""
    results = predict_batch(request.texts)
    return PredictBatchResponse(results=[PredictResponse(**r) for r in results])


@app.get("/stats")
def stats() -> dict:
    """Return monitoring statistics for Grafana Infinity."""
    return build_stats_payload()
