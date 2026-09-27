#!/usr/bin/env python3
"""Train IsolationForest on clearly labelled SYNTHETIC sensor-style data and export ONNX.

Data is generated with a fixed RNG seed. It does NOT come from hardware, an
internship, or an employer — it is a reproducible demo for portfolio / local use.

Features (8):
  mean_ax, std_ax, mean_ay, std_ay, mean_az, std_az  — window stats of 3-axis accel
  rms_vibration                                       — RMS magnitude of vibration
  temp_c                                              — temperature in Celsius

Score convention (documented in models/model_meta.json):
  sklearn IsolationForest score_samples: lower = more anomalous.
  We export / serve anomaly_score = -score_samples so that HIGHER = more anomalous.
  is_anomaly = (anomaly_score >= threshold).
  These are NOT class probabilities.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

FEATURE_NAMES = [
    "mean_ax",
    "std_ax",
    "mean_ay",
    "std_ay",
    "mean_az",
    "std_az",
    "rms_vibration",
    "temp_c",
]
N_FEATURES = len(FEATURE_NAMES)
RNG_SEED = 47


def generate_synthetic(
    n_normal: int,
    n_anomaly: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return X (float64) and y (0=normal, 1=anomaly). Completely synthetic."""
    normal = np.column_stack(
        [
            rng.normal(0.0, 0.4, n_normal),
            rng.uniform(0.05, 0.35, n_normal),
            rng.normal(0.0, 0.4, n_normal),
            rng.uniform(0.05, 0.35, n_normal),
            rng.normal(9.8, 0.3, n_normal),
            rng.uniform(0.05, 0.35, n_normal),
            rng.uniform(0.2, 1.5, n_normal),
            rng.normal(25.0, 2.0, n_normal),
        ]
    ).astype(np.float64)

    anomaly = np.column_stack(
        [
            rng.normal(0.0, 2.5, n_anomaly),
            rng.uniform(0.8, 3.0, n_anomaly),
            rng.normal(0.0, 2.5, n_anomaly),
            rng.uniform(0.8, 3.0, n_anomaly),
            rng.normal(9.8, 2.0, n_anomaly),
            rng.uniform(0.8, 3.0, n_anomaly),
            rng.uniform(3.0, 12.0, n_anomaly),
            rng.normal(45.0, 8.0, n_anomaly),
        ]
    ).astype(np.float64)

    X = np.vstack([normal, anomaly])
    y = np.array([0] * n_normal + [1] * n_anomaly, dtype=np.int32)
    return X, y


def score_anomaly_sklearn(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    """Higher = more anomalous (negated sklearn score_samples)."""
    raw = pipe.named_steps["iforest"].score_samples(
        pipe.named_steps["scale"].transform(X)
    )
    return -np.asarray(raw, dtype=np.float64)


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    X_train, _ = generate_synthetic(n_normal=4000, n_anomaly=0, rng=rng)
    X_cal, _ = generate_synthetic(n_normal=1000, n_anomaly=0, rng=rng)
    X_eval, y_eval = generate_synthetic(n_normal=800, n_anomaly=200, rng=rng)

    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "iforest",
                IsolationForest(
                    n_estimators=200,
                    contamination="auto",
                    max_samples=256,
                    random_state=RNG_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(X_train)

    cal_scores = score_anomaly_sklearn(pipe, X_cal)
    threshold_skl = float(np.quantile(cal_scores, 0.99))

    eval_scores = score_anomaly_sklearn(pipe, X_eval)
    y_pred = (eval_scores >= threshold_skl).astype(np.int32)
    precision = float(precision_score(y_eval, y_pred, zero_division=0))
    recall = float(recall_score(y_eval, y_pred, zero_division=0))
    f1 = float(f1_score(y_eval, y_pred, zero_division=0))

    print("=== Evaluation (synthetic held-out, sklearn path) ===")
    print(f"threshold (higher=more anomalous): {threshold_skl:.6f}")
    print(f"precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}")
    print(
        f"score normals mean={eval_scores[y_eval == 0].mean():.4f}  "
        f"anomalies mean={eval_scores[y_eval == 1].mean():.4f}"
    )

    MODELS.mkdir(parents=True, exist_ok=True)

    skl_path = MODELS / "sklearn_pipeline.pkl"
    with skl_path.open("wb") as f:
        pickle.dump(pipe, f)

    # Prefer explicit score_samples output from skl2onnx
    onnx_model = convert_sklearn(
        pipe,
        initial_types=[("features", FloatTensorType([None, N_FEATURES]))],
        target_opset={"": 12, "ai.onnx.ml": 3},
        options={IsolationForest: {"score_samples": True}},
    )

    fp32_path = MODELS / "model_fp32.onnx"
    default_path = MODELS / "model.onnx"
    raw = onnx_model.SerializeToString()
    fp32_path.write_bytes(raw)
    default_path.write_bytes(raw)
    print(f"Wrote {fp32_path} ({fp32_path.stat().st_size} bytes)")
    print(f"Wrote {default_path} (default = FP32 copy)")

    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    out_meta = [(o.name, list(o.shape), str(o.type)) for o in sess.get_outputs()]
    print("ONNX outputs:", out_meta)

    if "score_samples" not in out_names:
        raise RuntimeError(
            f"Expected 'score_samples' ONNX output, got {out_names}. "
            "Re-export with IsolationForest score_samples option."
        )
    score_tensor_name = "score_samples"
    # sklearn score_samples: lower = more anomalous → negate for serving
    serving_negate = True
    onnx_raw_polarity = "lower_more_anomalous_like_score_samples"
    score_polarity = "higher_more_anomalous"

    def onnx_anomaly_scores(X: np.ndarray) -> np.ndarray:
        outs = sess.run(None, {in_name: np.asarray(X, dtype=np.float32)})
        raw_s = np.asarray(outs[out_names.index(score_tensor_name)], dtype=np.float64).ravel()
        return -raw_s if serving_negate else raw_s

    sample = X_eval[:8]
    skl_s = score_anomaly_sklearn(pipe, sample)
    onnx_s = onnx_anomaly_scores(sample)
    max_abs_sample = float(np.max(np.abs(onnx_s - skl_s)))
    print(f"sample max|onnx-skl| anomaly_score = {max_abs_sample:.6e}")

    cal_onnx = onnx_anomaly_scores(X_cal)
    threshold = float(np.quantile(cal_onnx, 0.99))
    eval_onnx = onnx_anomaly_scores(X_eval)
    y_pred_onnx = (eval_onnx >= threshold).astype(np.int32)
    precision_o = float(precision_score(y_eval, y_pred_onnx, zero_division=0))
    recall_o = float(recall_score(y_eval, y_pred_onnx, zero_division=0))
    f1_o = float(f1_score(y_eval, y_pred_onnx, zero_division=0))
    max_abs_diff = float(np.max(np.abs(eval_onnx - eval_scores)))

    print("=== ONNX-path evaluation ===")
    print(f"score_tensor={score_tensor_name!r} serving_negate={serving_negate}")
    print(f"threshold={threshold:.6f}  max|onnx-skl|={max_abs_diff:.6e}")
    print(f"precision={precision_o:.4f}  recall={recall_o:.4f}  f1={f1_o:.4f}")

    np.savez_compressed(
        MODELS / "eval_data.npz",
        X=X_eval.astype(np.float32),
        y=y_eval,
        X_cal=X_cal.astype(np.float32),
    )

    meta = {
        "model_type": "IsolationForest",
        "pipeline": ["StandardScaler", "IsolationForest"],
        "n_features": N_FEATURES,
        "feature_names": FEATURE_NAMES,
        "feature_descriptions": {
            "mean_ax": "Mean acceleration on X over a synthetic window",
            "std_ax": "Std of acceleration on X",
            "mean_ay": "Mean acceleration on Y",
            "std_ay": "Std of acceleration on Y",
            "mean_az": "Mean acceleration on Z (gravity-ish ~9.8 in normals)",
            "std_az": "Std of acceleration on Z",
            "rms_vibration": "RMS vibration magnitude",
            "temp_c": "Temperature in Celsius",
        },
        "data_source": "synthetic_only",
        "data_disclaimer": (
            "All training/calibration/eval data is synthetically generated with a "
            "fixed RNG seed. It is not from hardware, sensors in the field, an "
            "internship, or an employer."
        ),
        "rng_seed": RNG_SEED,
        "train_size_normal_only": int(X_train.shape[0]),
        "calibration_size_normal_only": int(X_cal.shape[0]),
        "eval_size": {"normal": 800, "anomaly": 200},
        "threshold": threshold,
        "threshold_quantile_on_calibration_normals": 0.99,
        "score_polarity": score_polarity,
        "score_definition": (
            "anomaly_score is higher when more anomalous. "
            f"Serving reads ONNX tensor '{score_tensor_name}' and applies "
            f"anomaly_score = -raw (negate={serving_negate}). "
            "is_anomaly = (anomaly_score >= threshold). "
            "These values are NOT class probabilities."
        ),
        "onnx_input_name": in_name,
        "onnx_input_shape": [None, N_FEATURES],
        "onnx_input_dtype": "float32",
        "onnx_outputs": [{"name": n, "shape": s, "type": t} for n, s, t in out_meta],
        "onnx_score_tensor": score_tensor_name,
        "onnx_raw_score_polarity": onnx_raw_polarity,
        "serving_negate_raw_score": serving_negate,
        "default_artifact": "model.onnx",
        "artifacts": {
            "model.onnx": "default served model (FP32 copy unless switched)",
            "model_fp32.onnx": "FP32 ONNX export",
            "model_int8.onnx": "INT8 dynamic-quantized (see QUANTIZATION.md)",
            "sklearn_pipeline.pkl": "sklearn Pipeline for agreement tests",
            "eval_data.npz": "synthetic eval + calibration arrays",
            "threshold.json": "threshold + score polarity for serving",
            "model_meta.json": "full training / ONNX contract metadata",
        },
        "metrics_sklearn_path": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "metrics_onnx_path": {
            "precision": precision_o,
            "recall": recall_o,
            "f1": f1_o,
            "max_abs_score_diff_vs_sklearn": max_abs_diff,
        },
        "isolation_forest_params": {
            "n_estimators": 200,
            "contamination": "auto",
            "max_samples": 256,
            "random_state": RNG_SEED,
        },
    }

    meta_path = MODELS / "model_meta.json"
    thr_path = MODELS / "threshold.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    thr_path.write_text(
        json.dumps(
            {
                "threshold": threshold,
                "score_polarity": score_polarity,
                "serving_negate_raw_score": serving_negate,
                "onnx_score_tensor": score_tensor_name,
                "n_features": N_FEATURES,
                "feature_names": FEATURE_NAMES,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {meta_path}")
    print(f"Wrote {thr_path}")


if __name__ == "__main__":
    main()
