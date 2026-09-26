import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.onnx"


class OnnxClassifier:
    """Thin wrapper around an ONNX Runtime session."""

    def __init__(self, model_path: Path | None = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.n_features: int | None = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Run: python scripts/export_model.py"
            )
        # CPU EP is fine for demos; add CUDAExecutionProvider when a GPU is available
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        # shape is typically [batch, n_features]
        self.n_features = int(shape[1]) if isinstance(shape[1], int) else None
        logger.info("Loaded ONNX model from %s", self.model_path)

    @property
    def ready(self) -> bool:
        return self.session is not None

    def predict(self, features: list[float]) -> tuple[int, list[float] | None]:
        if self.session is None or self.input_name is None:
            raise RuntimeError("Model is not loaded")

        if self.n_features is not None and len(features) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {len(features)}"
            )

        x = np.asarray([features], dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: x})

        # skl2onnx classifiers usually return (label, probabilities)
        label = int(np.asarray(outputs[0]).ravel()[0])
        probs = None
        if len(outputs) > 1:
            raw = outputs[1]
            # sometimes a list of dicts per class; sometimes a 2D array
            if isinstance(raw, list) and raw and isinstance(raw[0], dict):
                probs = [float(v) for _, v in sorted(raw[0].items())]
            else:
                probs = np.asarray(raw).ravel().astype(float).tolist()

        return label, probs
