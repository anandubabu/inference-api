from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


@pytest.fixture(scope="session")
def meta() -> dict:
    return json.loads((MODELS / "model_meta.json").read_text())


@pytest.fixture(scope="session")
def threshold_info() -> dict:
    return json.loads((MODELS / "threshold.json").read_text())


@pytest.fixture(scope="session")
def normal_features(meta) -> list[float]:
    return [0.05, 0.15, -0.02, 0.18, 9.75, 0.12, 0.8, 24.5]


@pytest.fixture(scope="session")
def anomaly_features() -> list[float]:
    return [2.0, 2.5, -1.5, 2.0, 12.0, 2.5, 8.0, 55.0]


@pytest.fixture(scope="session")
def client():
    """Session-scoped app with default model loaded once."""
    from app.model import OnnxAnomalyModel
    import app.main as main_mod

    main_mod.model = OnnxAnomalyModel()
    main_mod.model.load()
    with TestClient(main_mod.app) as c:
        yield c


@pytest.fixture()
def client_no_model(tmp_path):
    """Temporary unavailable model; restores previous model afterward."""
    from app.model import OnnxAnomalyModel
    import app.main as main_mod

    missing = tmp_path / "missing.onnx"
    previous = main_mod.model
    main_mod.model = OnnxAnomalyModel(model_path=missing)
    try:
        main_mod.model.load()
    except Exception:
        pass
    with TestClient(main_mod.app) as c:
        yield c
    main_mod.model = previous
