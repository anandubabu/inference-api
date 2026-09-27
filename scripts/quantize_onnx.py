#!/usr/bin/env python3
"""Dynamically quantize FP32 ONNX → INT8 and compare honestly.

Documents:
  - op-type changes (graph actually quantized?)
  - prediction quality / FP32 vs INT8 agreement
  - single vs batch score stability (dynamic INT8 can depend on batch contents)
  - file sizes; optional process RSS — never claim memory savings from file size alone
  - latency on this host (not a blanket "faster" claim)

Writes models/model_int8.onnx and models/QUANTIZATION.md. Default served model
stays FP32 unless INT8 clearly wins on this box.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
FP32 = MODELS / "model_fp32.onnx"
INT8 = MODELS / "model_int8.onnx"
DEFAULT = MODELS / "model.onnx"
META = MODELS / "model_meta.json"
EVAL = MODELS / "eval_data.npz"
REPORT = MODELS / "QUANTIZATION.md"


def op_types(path: Path) -> Counter:
    model = onnx.load(str(path))
    return Counter(n.op_type for n in model.graph.node)


def load_meta() -> dict:
    return json.loads(META.read_text())


def anomaly_scores(sess: ort.InferenceSession, X: np.ndarray, meta: dict) -> np.ndarray:
    in_name = sess.get_inputs()[0].name
    outs = sess.run(None, {in_name: np.asarray(X, dtype=np.float32)})
    names = [o.name for o in sess.get_outputs()]
    tensor = meta["onnx_score_tensor"]
    raw = np.asarray(outs[names.index(tensor)], dtype=np.float64).ravel()
    if meta.get("serving_negate_raw_score", True):
        return -raw
    return raw


def timed_scores(sess, X, meta, repeats: int = 30) -> tuple[np.ndarray, float]:
    # warmup
    for _ in range(5):
        anomaly_scores(sess, X, meta)
    times = []
    last = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = anomaly_scores(sess, X, meta)
        times.append((time.perf_counter() - t0) * 1000)
    return last, float(np.median(times))


def main() -> None:
    if not FP32.exists():
        raise SystemExit(f"Missing {FP32}; run scripts/train_export.py first")
    meta = load_meta()
    data = np.load(EVAL)
    X = data["X"]
    y = data["y"]
    threshold = float(meta["threshold"])

    print(f"Quantizing {FP32} -> {INT8}")
    # weight-only dynamic quantization (MatMul/Gemm etc. where applicable)
    quantize_dynamic(
        model_input=str(FP32),
        model_output=str(INT8),
        weight_type=QuantType.QInt8,
    )

    ops_fp32 = op_types(FP32)
    ops_int8 = op_types(INT8)
    print("FP32 op types:", dict(ops_fp32))
    print("INT8 op types:", dict(ops_int8))
    new_ops = sorted(set(ops_int8) - set(ops_fp32))
    print("New op types after quant:", new_ops)

    sess_fp32 = ort.InferenceSession(str(FP32), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(INT8), providers=["CPUExecutionProvider"])

    s_fp32 = anomaly_scores(sess_fp32, X, meta)
    s_int8 = anomaly_scores(sess_int8, X, meta)
    abs_diff = np.abs(s_fp32 - s_int8)
    pred_fp32 = s_fp32 >= threshold
    pred_int8 = s_int8 >= threshold
    agreement = float(np.mean(pred_fp32 == pred_int8))

    # quality vs synthetic labels
    def prf(pred):
        from sklearn.metrics import f1_score, precision_score, recall_score

        return {
            "precision": float(precision_score(y, pred.astype(int), zero_division=0)),
            "recall": float(recall_score(y, pred.astype(int), zero_division=0)),
            "f1": float(f1_score(y, pred.astype(int), zero_division=0)),
        }

    metrics_fp32 = prf(pred_fp32)
    metrics_int8 = prf(pred_int8)

    # single vs batch: score of row i alone vs inside a batch
    idx = list(range(0, min(32, len(X))))
    single = np.array([anomaly_scores(sess_int8, X[i : i + 1], meta)[0] for i in idx])
    batched = anomaly_scores(sess_int8, X[idx], meta)
    batch_vs_single_max = float(np.max(np.abs(single - batched)))
    single_fp = np.array([anomaly_scores(sess_fp32, X[i : i + 1], meta)[0] for i in idx])
    batched_fp = anomaly_scores(sess_fp32, X[idx], meta)
    batch_vs_single_fp = float(np.max(np.abs(single_fp - batched_fp)))

    # latency: single row and batch=32
    _, lat_fp32_1 = timed_scores(sess_fp32, X[:1], meta)
    _, lat_int8_1 = timed_scores(sess_int8, X[:1], meta)
    _, lat_fp32_32 = timed_scores(sess_fp32, X[:32], meta)
    _, lat_int8_32 = timed_scores(sess_int8, X[:32], meta)

    size_fp32 = FP32.stat().st_size
    size_int8 = INT8.stat().st_size

    rss_note = "psutil not available"
    try:
        import os

        import psutil

        proc = psutil.Process(os.getpid())
        # Rough: create sessions in isolation is hard mid-process; report current RSS only
        rss_note = (
            f"Current process RSS after loading both sessions: "
            f"{proc.memory_info().rss / (1024 * 1024):.1f} MiB. "
            "This is NOT a controlled A/B of FP32-only vs INT8-only memory; "
            "do not claim memory reduction from file size alone."
        )
    except Exception as exc:  # noqa: BLE001
        rss_note = f"Could not sample RSS ({exc})"

    # Decide default: keep FP32 unless INT8 is clearly better on latency AND agreement high
    recommend_default = "FP32"
    reasons = []
    if not new_ops and ops_fp32 == ops_int8:
        reasons.append("Graph op types did not change in an obvious quantized way.")
    if agreement < 0.99:
        reasons.append(f"Label agreement with FP32 is only {agreement:.4f}.")
    if lat_int8_1 >= lat_fp32_1 * 0.95:
        reasons.append(
            f"Single-row median latency INT8 ({lat_int8_1:.3f} ms) is not clearly "
            f"faster than FP32 ({lat_fp32_1:.3f} ms)."
        )
    if batch_vs_single_max > 1e-5:
        reasons.append(
            f"Dynamic INT8 scores depend on batch contents "
            f"(max |single-batch|={batch_vs_single_max:.3e})."
        )
    if metrics_int8["f1"] + 1e-9 < metrics_fp32["f1"] - 0.01:
        reasons.append(
            f"INT8 F1 ({metrics_int8['f1']:.4f}) is worse than FP32 ({metrics_fp32['f1']:.4f})."
        )
    if not reasons:
        # only switch if INT8 is faster and agreement is excellent
        if lat_int8_1 < lat_fp32_1 * 0.9 and agreement >= 0.999:
            recommend_default = "INT8"
            reasons.append("INT8 is faster with near-perfect FP32 agreement on this host.")
        else:
            reasons.append("No strong reason to leave FP32 as default.")

    # Keep default as FP32 copy unless we recommend INT8
    if recommend_default == "INT8":
        DEFAULT.write_bytes(INT8.read_bytes())
        default_note = "Default model.onnx replaced with INT8 copy."
    else:
        DEFAULT.write_bytes(FP32.read_bytes())
        default_note = "Default model.onnx kept as FP32 copy."

    findings = {
        "file_sizes_bytes": {"fp32": size_fp32, "int8": size_int8},
        "file_size_ratio_int8_over_fp32": size_int8 / size_fp32,
        "new_op_types": new_ops,
        "ops_fp32": dict(ops_fp32),
        "ops_int8": dict(ops_int8),
        "score_diff_fp32_vs_int8": {
            "max_abs": float(abs_diff.max()),
            "mean_abs": float(abs_diff.mean()),
            "p95_abs": float(np.quantile(abs_diff, 0.95)),
        },
        "label_agreement": agreement,
        "metrics_fp32": metrics_fp32,
        "metrics_int8": metrics_int8,
        "batch_vs_single_max_abs_int8": batch_vs_single_max,
        "batch_vs_single_max_abs_fp32": batch_vs_single_fp,
        "latency_ms_median": {
            "fp32_batch1": lat_fp32_1,
            "int8_batch1": lat_int8_1,
            "fp32_batch32": lat_fp32_32,
            "int8_batch32": lat_int8_32,
        },
        "recommend_default": recommend_default,
        "reasons": reasons,
        "rss_note": rss_note,
        "default_note": default_note,
    }
    (MODELS / "quantization_results.json").write_text(json.dumps(findings, indent=2) + "\n")

    md = f"""# ONNX quantization findings (this repo)

Generated by `scripts/quantize_onnx.py` on the machine that built these artifacts.
**Synthetic eval data only.** Do not treat these as production SLOs.

## Method

- Source: `model_fp32.onnx` (sklearn Pipeline → skl2onnx)
- Technique: ONNX Runtime **dynamic** quantization (`quantize_dynamic`, `QuantType.QInt8`)
- Scores: `anomaly_score = -score_samples` (higher = more anomalous); threshold from `model_meta.json`

## Artifacts

| File | Size (bytes) |
|------|-------------:|
| `model_fp32.onnx` | {size_fp32} |
| `model_int8.onnx` | {size_int8} |
| ratio INT8/FP32 | {size_int8 / size_fp32:.3f} |

{default_note}

## Did the graph actually change?

- New op types after quantization: `{new_ops}`
- FP32 op counts: `{dict(ops_fp32)}`
- INT8 op counts: `{dict(ops_int8)}`

## Prediction quality (synthetic held-out)

| Variant | precision | recall | F1 | label agreement w/ FP32 |
|---------|----------:|-------:|---:|------------------------:|
| FP32 | {metrics_fp32['precision']:.4f} | {metrics_fp32['recall']:.4f} | {metrics_fp32['f1']:.4f} | 1.0000 |
| INT8 | {metrics_int8['precision']:.4f} | {metrics_int8['recall']:.4f} | {metrics_int8['f1']:.4f} | {agreement:.4f} |

Score |FP32 − INT8|: max={abs_diff.max():.4e}, mean={abs_diff.mean():.4e}, p95={np.quantile(abs_diff, 0.95):.4e}

## Single vs batch score stability

Dynamic quantization can make activations depend on the batch. Measured max |score(row alone) − score(row in batch)|:

- INT8: **{batch_vs_single_max:.6e}**
- FP32: **{batch_vs_single_fp:.6e}**

## Latency (median over timed session.run on this host)

| | batch=1 (ms) | batch=32 (ms) |
|--|-------------:|--------------:|
| FP32 | {lat_fp32_1:.4f} | {lat_fp32_32:.4f} |
| INT8 | {lat_int8_1:.4f} | {lat_int8_32:.4f} |

## Memory

{rss_note}

File size reduction ≠ runtime memory reduction. Do **not** claim “cut memory” from the INT8 file size alone.

## Recommendation

**Default served model: {recommend_default}**

Reasons:
"""
    for r in reasons:
        md += f"- {r}\n"
    md += """
### CV wording to avoid / correct

- Do **not** claim quantization “cut latency and memory” unless your own measurements show both.
- Do **not** imply INT8 was deployed to production.
- Prefer: “evaluated ONNX Runtime dynamic INT8 quantization; kept FP32 as default after measuring agreement, batch stability, and latency on this host.”
"""
    REPORT.write_text(md)
    print(json.dumps(findings, indent=2))
    print(f"Wrote {REPORT}")
    print(f"Recommend default: {recommend_default}")


if __name__ == "__main__":
    main()
