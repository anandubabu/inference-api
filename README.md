# Real-Time ML Inference API — Containerized Model Serving

FastAPI service that serves an **IsolationForest anomaly detector** exported to ONNX. The model was trained on **clearly labelled synthetic** multi-feature sensor-style data (not hardware logs, not internship/employer data).

This rebuild focuses on honest API design: input validation, batch inference, health checks that fail when the model is missing, structured request logging (no public log endpoint), and measured latency/RPS on this host.

## What it does

- Loads `models/model.onnx` once at startup with **ONNX Runtime** (CPU)
- **`GET /health`** — HTTP 200 when ready, **HTTP 503** when the model is unavailable
- **`POST /predict`** — score one feature vector
- **`POST /predict/batch`** — score many vectors in one `session.run`
- Returns `is_anomaly`, `anomaly_score`, `threshold`, `latency_ms` (not class probabilities)
- Pydantic validation (feature count, NaN/Inf, batch size caps)
- Docker / docker-compose with a healthcheck that treats non-200 as unhealthy
- Structured request logs: method, path, status, duration_ms (no raw features)

### Response format change (vs older demo)

Older versions returned `label` / `probabilities` for a tiny Iris classifier. This project is an **anomaly** API:

| Field | Meaning |
|-------|---------|
| `anomaly_score` | Higher = more anomalous (sklearn `score_samples` negated) |
| `is_anomaly` | `anomaly_score >= threshold` |
| `threshold` | Calibrated on held-out synthetic normals (99th percentile) |
| `latency_ms` | Server-side inference time for that call / batch |

**Scores are not probabilities.**

## Features (8)

| Name | Meaning (synthetic) |
|------|---------------------|
| `mean_ax` / `std_ax` | Window mean/std of accel X |
| `mean_ay` / `std_ay` | Window mean/std of accel Y |
| `mean_az` / `std_az` | Window mean/std of accel Z (~9.8 in normals) |
| `rms_vibration` | RMS vibration magnitude |
| `temp_c` | Temperature °C |

## Tech stack

Python, FastAPI, Uvicorn, Pydantic v2, ONNX Runtime, scikit-learn, skl2onnx, Docker.

## Quick start

```bash
git clone https://github.com/anandubabu/inference-api.git
cd inference-api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # or requirements.txt for API-only
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API: http://localhost:8000 — OpenAPI at `/docs`.

### Docker

```bash
docker compose up --build
```

## API examples

### Health

```bash
curl -s http://localhost:8000/health
# {"status":"ok","model_loaded":true,"n_features":8,"model_path":"..."}
```

### Single predict

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[0.05,0.15,-0.02,0.18,9.75,0.12,0.8,24.5]}'
```

```json
{
  "is_anomaly": false,
  "anomaly_score": 0.45,
  "threshold": 0.566,
  "n_features": 8,
  "latency_ms": 2.1,
  "score_polarity": "higher_more_anomalous"
}
```

### Batch predict

```bash
curl -s -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"instances":[[0.05,0.15,-0.02,0.18,9.75,0.12,0.8,24.5],[2.0,2.5,-1.5,2.0,12.0,2.5,8.0,55.0]]}'
```

Wrong length / NaN / Inf / empty / oversized batch (cap 256) → HTTP **422**.  
Model missing → `/health` and predict routes → HTTP **503**.

## Train / export / quantize

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/train_export.py    # synthetic data → model_fp32.onnx + meta
python scripts/quantize_onnx.py   # INT8 attempt + models/QUANTIZATION.md
```

Committed artifacts under `models/` so Docker/CI work without retraining.

### ONNX contract (replacement models)

- Input: `float32` tensor named `features`, shape `[N, 8]`
- Score output used by the API: `score_samples` (lower raw = more anomalous; API negates)
- Also present: `label`, `scores` (decision-function-like); serving uses `score_samples`
- Keep `models/threshold.json` / `model_meta.json` in sync when swapping models

## Quantization (honest)

Dynamic INT8 via ONNX Runtime was evaluated on this IsolationForest graph:

- Tree ensemble ops were **not** meaningfully quantized (op types unchanged; INT8 file was **larger**)
- No clear single-row latency win on this host
- Default served artifact remains **FP32** (`model.onnx` = copy of `model_fp32.onnx`)

See `models/QUANTIZATION.md` and `models/quantization_results.json`. Prefer CV wording like “evaluated ORT dynamic INT8; kept FP32 after measuring agreement and latency,” not “cut latency and memory.”

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Covers health 200/503, single/batch predict, validation 422s, FP32↔sklearn agreement, INT8 tolerances.

## Benchmarks (this host only)

Measured on the Linux x86_64 box that built this commit. Warmup excluded.  
**RPS = completed_requests / total_elapsed_seconds** (not `1000/median`).

| Run | n | e2e p50 (ms) | e2e p95 (ms) | server p50 (ms) | RPS |
|-----|--:|-------------:|-------------:|----------------:|----:|
| `/predict` sequential | 200 | 3.48 | 4.69 | 2.14 | 272.7 |
| `/predict` concurrent (8) | 200 | 8.22 | 11.84 | 3.50 | 905.3 |
| `/predict/batch` sequential (batch=16) | 50 | 4.56 | 5.49 | 2.86 | 212.5 |

Raw JSON: `docs/benchmarks/results_latest.json`. Reproduce:

```bash
# terminal 1
uvicorn app.main:app --host 127.0.0.1 --port 8000
# terminal 2
python scripts/bench.py --n 200 --warmup 20 --concurrency 8 --batch-size 16
```

These are lab numbers on a quiet CPU VM — not hard real-time guarantees and not a production SLA.

## Layout

```
app/                  # FastAPI app, OnnxAnomalyModel, schemas
models/               # ONNX + threshold/meta + QUANTIZATION.md
scripts/
  train_export.py     # synthetic train + ONNX export
  quantize_onnx.py    # INT8 evaluation
  bench.py            # p50/p95 + RPS
tests/
docs/benchmarks/      # measured results
Dockerfile
docker-compose.yml
.github/workflows/ci.yml
```

## Limitations

- Synthetic data only; not a claim of field sensor accuracy
- CPU ONNX Runtime; no GPU path in this repo
- Dynamic INT8 did not help this tree model on this host
- Structured logs are process stdout — there is **no** `/logs` HTTP endpoint
- Not presented as a production deployment history

## License

MIT — see `LICENSE`.
