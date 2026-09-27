"""FastAPI anomaly inference service."""
from __future__ import annotations

import logging
import math
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.model import OnnxAnomalyModel
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    BatchPredictionItem,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("inference-api")

model = OnnxAnomalyModel(os.getenv("MODEL_PATH") or None)


def _sanitize_for_json(obj):
    """Make validation/error payloads JSON-safe (no NaN/Inf/Exception objects)."""
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, BaseException):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Structured request logging: method, path, status, duration_ms. No feature payloads."""

    async def dispatch(self, request: Request, call_next):
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        except Exception:
            status = 500
            raise
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "request method=%s path=%s status=%s duration_ms=%.2f",
                request.method,
                request.url.path,
                status,
                duration_ms,
            )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        model.load()
    except Exception as exc:  # noqa: BLE001
        # Stay up so /health can return 503; Docker healthcheck fails on non-200.
        logger.error("Model unavailable at startup: %s", exc)
    yield


app = FastAPI(
    title="Real-Time ML Inference API",
    description=(
        "Containerized ONNX anomaly detection (IsolationForest on synthetic "
        "sensor-style features). Scores are not probabilities."
    ),
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(RequestLogMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    content = _sanitize_for_json({"detail": exc.errors()})
    return JSONResponse(status_code=422, content=content)


@app.exception_handler(Exception)
async def unhandled(_request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", response_model=HealthResponse)
def health():
    if model.ready:
        return HealthResponse(
            status="ok",
            model_loaded=True,
            n_features=model.n_features,
            model_path=str(model.model_path),
        )
    # Explicit non-200 so Docker HEALTHCHECK / compose healthcheck fail
    return JSONResponse(
        status_code=503,
        content={
            "status": "unavailable",
            "model_loaded": False,
            "n_features": model.n_features,
            "model_path": str(model.model_path),
            "detail": model.load_error or "Model not loaded",
        },
    )


def _require_model() -> None:
    if not model.ready:
        raise HTTPException(
            status_code=503,
            detail=model.load_error or "Model not loaded",
        )


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    _require_model()
    started = time.perf_counter()
    try:
        result = model.predict_one(body.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed") from None
    latency_ms = (time.perf_counter() - started) * 1000
    return PredictResponse(
        **result,
        latency_ms=round(latency_ms, 3),
        score_polarity=model.score_polarity,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(body: BatchPredictRequest):
    _require_model()
    started = time.perf_counter()
    try:
        results = model.predict_batch(body.instances)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception("Batch inference failed")
        raise HTTPException(status_code=500, detail="Inference failed") from None
    latency_ms = (time.perf_counter() - started) * 1000
    return BatchPredictResponse(
        predictions=[BatchPredictionItem(**r) for r in results],
        batch_size=len(results),
        latency_ms=round(latency_ms, 3),
        score_polarity=model.score_polarity,
    )
