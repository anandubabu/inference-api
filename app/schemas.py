"""Request/response schemas for the anomaly inference API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


MAX_BATCH = 256
MAX_FEATURES = 128  # hard cap; actual required length comes from the model


def _reject_nonfinite(values: list[float], *, where: str) -> list[float]:
    cleaned: list[float] = []
    for i, v in enumerate(values):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{where}[{i}] must be a finite number, got {type(v).__name__}")
        fv = float(v)
        if fv != fv:  # NaN
            raise ValueError(f"{where}[{i}] is NaN; finite floats required")
        if fv == float("inf") or fv == float("-inf"):
            raise ValueError(f"{where}[{i}] is Inf; finite floats required")
        # Reject values that overflow float32
        if abs(fv) > 3.4028234663852886e38:
            raise ValueError(f"{where}[{i}] overflows float32 range")
        cleaned.append(fv)
    return cleaned


class PredictRequest(BaseModel):
    """Single feature vector for anomaly scoring."""

    features: list[float] = Field(..., min_length=1, max_length=MAX_FEATURES)

    @field_validator("features", mode="before")
    @classmethod
    def _validate_features(cls, v: Any) -> list[float]:
        if not isinstance(v, list):
            raise ValueError("features must be a list of numbers")
        if len(v) == 0:
            raise ValueError("features must not be empty")
        if len(v) > MAX_FEATURES:
            raise ValueError(f"features length must be <= {MAX_FEATURES}")
        return _reject_nonfinite(v, where="features")


class PredictResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    n_features: int
    latency_ms: float
    score_polarity: str = "higher_more_anomalous"


class BatchPredictRequest(BaseModel):
    """Batch of feature vectors; one ONNX session.run for the whole batch."""

    instances: list[list[float]] = Field(..., min_length=1, max_length=MAX_BATCH)

    @model_validator(mode="after")
    def _validate_instances(self) -> BatchPredictRequest:
        if not self.instances:
            raise ValueError("instances must not be empty")
        if len(self.instances) > MAX_BATCH:
            raise ValueError(f"batch size must be <= {MAX_BATCH}")
        widths = {len(row) for row in self.instances}
        if len(widths) != 1:
            raise ValueError(
                f"all rows must have the same length; got lengths {sorted(widths)}"
            )
        cleaned: list[list[float]] = []
        for i, row in enumerate(self.instances):
            if len(row) == 0:
                raise ValueError(f"instances[{i}] must not be empty")
            if len(row) > MAX_FEATURES:
                raise ValueError(f"instances[{i}] length must be <= {MAX_FEATURES}")
            cleaned.append(_reject_nonfinite(row, where=f"instances[{i}]"))
        self.instances = cleaned
        return self


class BatchPredictionItem(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    n_features: int


class BatchPredictResponse(BaseModel):
    predictions: list[BatchPredictionItem]
    batch_size: int
    latency_ms: float  # total server-side inference time for the batch
    score_polarity: str = "higher_more_anomalous"


class HealthResponse(BaseModel):
    status: str  # "ok" | "unavailable"
    model_loaded: bool
    n_features: int | None = None
    model_path: str | None = None
