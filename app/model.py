"""ONNX Runtime wrapper for IsolationForest anomaly scores."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "model.onnx"
DEFAULT_META_PATH = ROOT / "models" / "model_meta.json"
DEFAULT_THRESHOLD_PATH = ROOT / "models" / "threshold.json"


class OnnxAnomalyModel:
    """Load once; score float32 [N, F] → anomaly_score (higher = more anomalous)."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        meta_path: Path | str | None = None,
        threshold_path: Path | str | None = None,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.meta_path = Path(meta_path) if meta_path else DEFAULT_META_PATH
        self.threshold_path = (
            Path(threshold_path) if threshold_path else DEFAULT_THRESHOLD_PATH
        )
        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.n_features: int | None = None
        self.score_tensor: str | None = None
        self.serving_negate: bool = True
        self.threshold: float | None = None
        self.score_polarity: str = "higher_more_anomalous"
        self.feature_names: list[str] = []
        self._load_error: str | None = None

    @property
    def ready(self) -> bool:
        return self.session is not None and self.threshold is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> None:
        """Load ONNX + metadata. Raises on failure; leaves ready=False if caller catches."""
        self.session = None
        self._load_error = None
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {self.model_path}. "
                    "Run: python scripts/train_export.py"
                )

            meta: dict = {}
            if self.meta_path.exists():
                meta = json.loads(self.meta_path.read_text())
            thr: dict = {}
            if self.threshold_path.exists():
                thr = json.loads(self.threshold_path.read_text())

            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
            inp = self.session.get_inputs()[0]
            self.input_name = inp.name
            shape = inp.shape
            if len(shape) < 2:
                raise ValueError(f"Expected input rank >= 2, got shape {shape}")
            n_from_onnx = shape[1] if isinstance(shape[1], int) else None
            n_from_meta = meta.get("n_features") or thr.get("n_features")
            if n_from_onnx is not None:
                self.n_features = int(n_from_onnx)
            elif n_from_meta is not None:
                self.n_features = int(n_from_meta)
            else:
                raise ValueError("Cannot determine n_features from ONNX or metadata")

            out_names = [o.name for o in self.session.get_outputs()]
            self.score_tensor = (
                thr.get("onnx_score_tensor")
                or meta.get("onnx_score_tensor")
                or ("score_samples" if "score_samples" in out_names else None)
            )
            if self.score_tensor is None or self.score_tensor not in out_names:
                raise ValueError(
                    f"Score tensor {self.score_tensor!r} not in ONNX outputs {out_names}"
                )

            self.serving_negate = bool(
                thr.get(
                    "serving_negate_raw_score",
                    meta.get("serving_negate_raw_score", True),
                )
            )
            if "threshold" not in thr and "threshold" not in meta:
                raise ValueError("threshold missing from threshold.json / model_meta.json")
            self.threshold = float(thr.get("threshold", meta["threshold"]))
            self.score_polarity = thr.get(
                "score_polarity", meta.get("score_polarity", "higher_more_anomalous")
            )
            self.feature_names = list(
                thr.get("feature_names") or meta.get("feature_names") or []
            )

            # Smoke-run to validate graph
            dummy = np.zeros((1, self.n_features), dtype=np.float32)
            self._run(dummy)
            logger.info(
                "Loaded anomaly model from %s (n_features=%s threshold=%.6f score=%s)",
                self.model_path,
                self.n_features,
                self.threshold,
                self.score_tensor,
            )
        except Exception as exc:
            self.session = None
            self._load_error = str(exc)
            logger.error("Failed to load model: %s", exc)
            raise

    def _run(self, x: np.ndarray) -> np.ndarray:
        assert self.session is not None and self.input_name and self.score_tensor
        outs = self.session.run(None, {self.input_name: x})
        names = [o.name for o in self.session.get_outputs()]
        raw = np.asarray(outs[names.index(self.score_tensor)], dtype=np.float64).ravel()
        return -raw if self.serving_negate else raw

    def _validate_matrix(self, rows: list[list[float]]) -> np.ndarray:
        if self.session is None or self.n_features is None or self.threshold is None:
            raise RuntimeError("Model is not loaded")
        if not rows:
            raise ValueError("empty batch")
        for i, row in enumerate(rows):
            if len(row) != self.n_features:
                raise ValueError(
                    f"Expected {self.n_features} features, got {len(row)} (row {i})"
                )
        x = np.asarray(rows, dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError("features contain NaN or Inf")
        return x

    def predict_one(self, features: list[float]) -> dict:
        scores = self.predict_batch([features])
        return scores[0]

    def predict_batch(self, instances: list[list[float]]) -> list[dict]:
        x = self._validate_matrix(instances)
        scores = self._run(x)
        assert self.threshold is not None and self.n_features is not None
        results = []
        for s in scores:
            sf = float(s)
            results.append(
                {
                    "is_anomaly": bool(sf >= self.threshold),
                    "anomaly_score": sf,
                    "threshold": float(self.threshold),
                    "n_features": int(self.n_features),
                }
            )
        return results
