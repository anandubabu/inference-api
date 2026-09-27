from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


@pytest.fixture(scope="module")
def meta():
    return json.loads((MODELS / "model_meta.json").read_text())


@pytest.fixture(scope="module")
def eval_data():
    data = np.load(MODELS / "eval_data.npz")
    return data["X"], data["y"]


def _scores(sess, X, meta):
    in_name = sess.get_inputs()[0].name
    outs = sess.run(None, {in_name: np.asarray(X, dtype=np.float32)})
    names = [o.name for o in sess.get_outputs()]
    raw = np.asarray(outs[names.index(meta["onnx_score_tensor"])], dtype=np.float64).ravel()
    return -raw if meta.get("serving_negate_raw_score", True) else raw


def test_fp32_onnx_vs_sklearn_agreement(meta, eval_data):
    X, _ = eval_data
    with open(MODELS / "sklearn_pipeline.pkl", "rb") as f:
        pipe = pickle.load(f)
    skl = -pipe.named_steps["iforest"].score_samples(
        pipe.named_steps["scale"].transform(X)
    )
    sess = ort.InferenceSession(
        str(MODELS / "model_fp32.onnx"), providers=["CPUExecutionProvider"]
    )
    onnx_scores = _scores(sess, X, meta)
    max_abs = float(np.max(np.abs(skl - onnx_scores)))
    assert max_abs < 1e-5, f"FP32 ONNX vs sklearn max abs diff {max_abs}"


def test_default_model_matches_fp32(meta, eval_data):
    X, _ = eval_data
    s_fp = ort.InferenceSession(
        str(MODELS / "model_fp32.onnx"), providers=["CPUExecutionProvider"]
    )
    s_def = ort.InferenceSession(
        str(MODELS / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    a = _scores(s_fp, X[:50], meta)
    b = _scores(s_def, X[:50], meta)
    assert np.allclose(a, b, atol=1e-6)


@pytest.mark.skipif(
    not (MODELS / "model_int8.onnx").exists(),
    reason="INT8 artifact not present",
)
def test_int8_vs_fp32_documented_tolerance(meta, eval_data):
    """INT8 may differ; allow documented tolerance. Also check batch vs single."""
    X, y = eval_data
    s_fp = ort.InferenceSession(
        str(MODELS / "model_fp32.onnx"), providers=["CPUExecutionProvider"]
    )
    s_i8 = ort.InferenceSession(
        str(MODELS / "model_int8.onnx"), providers=["CPUExecutionProvider"]
    )
    fp = _scores(s_fp, X, meta)
    i8 = _scores(s_i8, X, meta)
    # Tree ensembles often unchanged by dynamic quant → diffs may be ~0
    max_abs = float(np.max(np.abs(fp - i8)))
    # Generous ceiling; real findings live in QUANTIZATION.md
    assert max_abs < 0.5, f"Unexpectedly large INT8 drift: {max_abs}"

    thr = float(meta["threshold"])
    agree = float(np.mean((fp >= thr) == (i8 >= thr)))
    assert agree >= 0.95

    # Batch vs single for INT8 (dynamic quant caveat)
    idx = list(range(16))
    single = np.array([_scores(s_i8, X[i : i + 1], meta)[0] for i in idx])
    batched = _scores(s_i8, X[idx], meta)
    # Documented: may be nonzero for some graphs; IsolationForest trees usually 0
    assert float(np.max(np.abs(single - batched))) < 0.1
