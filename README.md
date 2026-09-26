# Real-Time ML Inference API — Containerized Model Serving

FastAPI service that serves **tabular / sensor ONNX models** for edge-style anomaly detection and signal pipelines — the same shape as the portfolio project on sensor preprocessing, Isolation Forest / tree baselines, and low-latency inference.

This is **not** an image-classifier demo. The request body is a flat feature vector (windowed sensor stats, FFT bins, scaled channels, etc.). Swap in your own `.onnx` from an embedded or edge training run and keep the same `/predict` contract.

## What it does

- Loads `models/model.onnx` once at startup with **ONNX Runtime** (CPU)
- **`GET /health`** — liveness + whether the model is ready
- **`POST /predict`** — infer on a feature vector; returns label, probabilities (when available), and `latency_ms`
- **Pydantic** validation on inputs
- **Docker / docker-compose** for reproducible deploys
- Structured INFO logs per prediction

Ideal fit for CV talking points: real-time sensor streams, anomaly flags, and portable edge serving without rewriting the API when the model changes.

## Tech stack

| Piece | Choice |
|--------|--------|
| API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Inference | ONNX Runtime (CPU) |
| Packaging | Docker + docker-compose |
| Sample export | scikit-learn → ONNX (`scripts/export_model.py`) |

## Quick start

```bash
git clone https://github.com/anandubabu/inference-api.git
cd inference-api
make install
make run
```

API: http://localhost:8000 — docs at `/docs`.

### Docker

```bash
docker compose up --build
```

## Bring your own ONNX (one command)

Drop any compatible classifier/regressor ONNX file (input: `float32 [batch, n_features]`):

```bash
make use-model MODEL=/path/to/your_anomaly_model.onnx
make run
# or: docker compose up --build
```

Under the hood this runs `scripts/use_model.sh`, which copies your file to `models/model.onnx`. Restart the process (or rebuild the image) so the new weights load.

Feature length must match the model. If you trained on 32 window features, send 32 floats in `features`.

## API examples

### Health

```bash
curl -s http://localhost:8000/health
# {"status":"ok","model_loaded":true}
```

### Predict (sensor-style feature vector)

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[5.1, 3.5, 1.4, 0.2]}'
```

```json
{
  "label": 0,
  "probabilities": [0.98, 0.02, 0.0],
  "latency_ms": 0.05
}
```

Wrong length → HTTP 422. Model missing → HTTP 503 / health `degraded`.

## Latency benchmarks (measured)

Host: Linux x86_64 CPU VM used to build this repo (single client, warmed up). Sample model: 4-feature sklearn → ONNX classifier. **n = 200** after 20 warmup calls.

| Metric | p50 | p95 |
|--------|-----|-----|
| End-to-end HTTP `/predict` (ms) | **0.69** | **1.24** |
| Server-side ONNX `latency_ms` (ms) | **0.05** | **0.08** |

Rough single-client throughput from e2e p50: ~1400 req/s on this host. These numbers are for a tiny tabular model on a quiet CPU box — not a cloud free-tier SLA and not an image network. Re-run on your machine (or free-tier host) with the commands below and replace the table.

### Reproduce benchmarks

```bash
make run          # terminal 1
make bench        # terminal 2 — defaults to 200 requests
# custom feature width after swapping models:
. .venv/bin/activate && python scripts/bench.py --features 0.1,0.2,...,0.N --n 500
```

Or a raw loop:

```bash
for i in $(seq 1 200); do
  curl -s -o /dev/null -w "%{time_total}\n" -X POST http://127.0.0.1:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"features":[5.1,3.5,1.4,0.2]}'
done
```

For concurrent p50/p95 and RPS under load, use `hey` or `k6` against `/predict`.

## Layout

```
app/                 # FastAPI app, ONNX wrapper, schemas
models/model.onnx    # served artifact (sample or your swap-in)
scripts/
  export_model.py    # rebuild sample ONNX
  use_model.sh       # bring-your-own ONNX
  bench.py           # p50 / p95 helper
Makefile
Dockerfile
docker-compose.yml
```

## License

MIT — portfolio / interview demo for embedded and edge ML serving.
