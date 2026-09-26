# Real-Time ML Inference API — Containerized Model Serving

Production-style sketch of the CV project: a trained anomaly/classification model served as a real-time REST API with FastAPI, validated inputs via Pydantic, ONNX Runtime for fast inference, and Docker for reproducible deployment.

The sample model is a generic sklearn classifier exported to ONNX (Iris-shaped, 4 features). Swap `models/model.onnx` for your own model without changing the API shape.

## Stack

- Python
- FastAPI
- ONNX Runtime
- Pydantic
- Docker / docker-compose
- scikit-learn + skl2onnx (export script only)

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness + whether the model loaded |
| `POST` | `/predict` | Run inference on a feature vector |

### Example

```bash
curl -s http://localhost:8000/health

curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features":[5.1, 3.5, 1.4, 0.2]}'
```

Response shape:

```json
{
  "label": 0,
  "probabilities": [0.97, 0.02, 0.01],
  "latency_ms": 0.42
}
```

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install scikit-learn skl2onnx onnx   # only if you need to rebuild the model
python scripts/export_model.py
uvicorn app.main:app --reload --port 8000
```

## Docker

```bash
# rebuild the sample model first if models/model.onnx is missing
python scripts/export_model.py

docker compose up --build
```

API listens on `http://localhost:8000`.

## Exporting your own model

1. Train whatever you like in sklearn (or export from PyTorch/TF to ONNX).
2. For sklearn: adapt `scripts/export_model.py` and write `models/model.onnx`.
3. Keep the input name/`features` tensor as `float32 [batch, n_features]`, or update `app/model.py` to match your graph.
4. Rebuild the image.

## Logging and latency notes

- Each `/predict` call logs `label` and `latency_ms` at INFO.
- Response includes measured server-side inference latency (pre/post ONNX session).
- For real-time validation, benchmark with something like:

```bash
# rough local check — not a substitute for load testing
for i in $(seq 1 100); do
  curl -s -o /dev/null -w '%{time_total}\n' -X POST http://localhost:8000/predict \
    -H 'Content-Type: application/json' \
    -d '{"features":[5.1, 3.5, 1.4, 0.2]}'
done
```

Capture p50/p95 of those times and requests-per-second under concurrent load (e.g. `hey` or `wrk`) when you need numbers for a write-up. ONNX Runtime quantization can cut latency and memory further for edge or high-QPS deployments — leave that as a follow-up once you have a real model.

## Layout

```
app/
  main.py       # FastAPI routes + lifespan model load
  model.py      # ONNX Runtime wrapper
  schemas.py    # Pydantic request/response models
models/
  model.onnx    # served artifact
scripts/
  export_model.py
Dockerfile
docker-compose.yml
```

## License

MIT (personal portfolio / interview demo).
