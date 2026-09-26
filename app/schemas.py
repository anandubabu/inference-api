from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    # Flat feature vector — swap length/meaning when you change the model
    features: list[float] = Field(..., min_length=1, max_length=128)


class PredictResponse(BaseModel):
    label: int
    probabilities: list[float] | None = None
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
