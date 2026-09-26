import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.model import OnnxClassifier
from app.schemas import HealthResponse, PredictRequest, PredictResponse

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("inference-api")

classifier = OnnxClassifier(os.getenv("MODEL_PATH") or None)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        classifier.load()
    except FileNotFoundError as exc:
        # Allow /health to stay up so Docker healthchecks can report the miss
        logger.error("%s", exc)
    yield


app = FastAPI(
    title="Real-Time ML Inference API",
    description="Containerized ONNX model serving with FastAPI",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if classifier.ready else "degraded",
        model_loaded=classifier.ready,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    if not classifier.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    started = time.perf_counter()
    try:
        label, probs = classifier.predict(body.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed") from None

    latency_ms = (time.perf_counter() - started) * 1000
    logger.info("predict label=%s latency_ms=%.2f", label, latency_ms)
    return PredictResponse(label=label, probabilities=probs, latency_ms=round(latency_ms, 3))


@app.exception_handler(Exception)
async def unhandled(_request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
