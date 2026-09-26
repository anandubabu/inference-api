# Real-Time ML Inference API — Containerized Model Serving

A small, production-shaped service that loads an ONNX model once at startup and serves predictions over HTTP. Built to match the portfolio project of the same name: real-time inference, request validation, structured logging, and a reproducible Docker image you can run anywhere.

The checked-in model is a generic multi-class classifier (4 float features) so you can clone, run, and swap in a real artifact later without rewriting the API.

## What it does

- Loads `models/model.onnx` with **ONNX Runtime** during app startup
- Exposes **`GET /health`** for liveness and model-ready status
- Exposes **`POST /predict`** for a single feature vector (or extend to batch later)
- Validates payloads with **Pydantic**
- Returns the predicted label, optional class probabilities, and server-side `latency_ms`
- Logs each prediction at INFO for easy debugging under load

## Tech stack

| Piece | Choice |
|--------|--------|
| API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Inference | ONNX Runtime (CPU) |
| Packaging | Docker + docker-compose |
| Sample export | scikit-learn → ONNX via `skl2onnx` (`scripts/export_model.py`) |

## Project layout

```
app/
  main.py            # routes, lifespan, logging
  model.py           # ONNX Runtime wrapper
  schemas.py         # request / response models
models/
  model.onnx         # served artifact (sample classifier)
scripts/
  export_model.py    # rebuild the sample ONNX file
Dockerfile
docker-compose.yml
requirements.txt
```

## Setup

Requires Python 3.11+ (local) or Docker.

```bash
git clone https://github.com/anandubabu/inference-api.git
cd inference-api

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The sample model is already in `models/model.onnx`. To rebuild it:

```bash
pip install scikit-learn skl2onnx onnx
python scripts/export_model.py
```

## Run locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open docs at http://localhost:8000/docs

## Run with Docker

```bash
docker compose up --build
```

The API listens on http://localhost:8000. Compose includes a healthcheck against `/health`.

## API examples

### Health

```bash
curl -s http://localhost:8000/health
```

```json
{"status":"ok","model_loaded":true}
```

If the model file is missing, status is `"degraded"` and `model_loaded` is `false`.

### Predict

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[5.1, 3.5, 1.4, 0.2]}'
```

```json
{
  "label": 0,
  "probabilities": [0.98, 0.02, 0.0],
  "latency_ms": 0.61
}
```

Wrong feature length returns HTTP 422. Model not loaded returns HTTP 503.

## Swapping in your own model

1. Export your model to ONNX (sklearn via `scripts/export_model.py`, or PyTorch/TensorFlow export).
2. Place it at `models/model.onnx` (or set `MODEL_PATH`).
3. Keep input as `float32` with shape `[batch, n_features]`, or adjust `app/model.py` to match your graph.
4. Restart Uvicorn or rebuild the image.

## Latency notes

Each `/predict` response includes measured inference time in milliseconds. For a quick local sample:

```bash
for i in $(seq 1 100); do
  curl -s -o /dev/null -w "%{time_total}\n" -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"features":[5.1, 3.5, 1.4, 0.2]}'
done
```

Use a proper load tool (`hey`, `wrk`, or k6) when you need p50/p95 latency and requests-per-second for a write-up. ONNX Runtime quantization is a natural next step for tighter latency or memory budgets.

## License

MIT — portfolio / interview demo.
