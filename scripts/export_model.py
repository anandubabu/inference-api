"""Train a tiny sklearn classifier and export it to ONNX.

Swap this for your real training pipeline when you have one.
The sample model expects 4 float features (Iris-shaped).
"""

from pathlib import Path

import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models" / "model.onnx"


def main() -> None:
    X, y = load_iris(return_X_y=True)
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    pipe.fit(X, y)

    # Fixed batch=1 keeps the ONNX graph simple for serving demos
    onnx_model = convert_sklearn(
        pipe,
        initial_types=[("features", FloatTensorType([None, X.shape[1]]))],
        target_opset=12,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(onnx_model.SerializeToString())
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")

    # Sanity check with onnxruntime if available
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        sample = np.asarray([X[0]], dtype=np.float32)
        pred = sess.run(None, {name: sample})
        print("sample prediction:", pred[0])
    except ImportError:
        pass


if __name__ == "__main__":
    main()
