from __future__ import annotations

import json


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_features"] == 8


def test_predict_normal(client, normal_features):
    r = client.post("/predict", json={"features": normal_features})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {
        "is_anomaly",
        "anomaly_score",
        "threshold",
        "latency_ms",
        "n_features",
        "score_polarity",
    }
    assert "probabilities" not in body
    assert "label" not in body
    assert body["n_features"] == 8
    assert body["score_polarity"] == "higher_more_anomalous"
    assert body["is_anomaly"] is False
    assert body["anomaly_score"] < body["threshold"]


def test_predict_anomaly(client, anomaly_features):
    r = client.post("/predict", json={"features": anomaly_features})
    assert r.status_code == 200
    body = r.json()
    assert body["is_anomaly"] is True
    assert body["anomaly_score"] >= body["threshold"]


def test_predict_batch(client, normal_features, anomaly_features):
    r = client.post(
        "/predict/batch",
        json={"instances": [normal_features, anomaly_features, normal_features]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["batch_size"] == 3
    assert len(body["predictions"]) == 3
    assert body["predictions"][0]["is_anomaly"] is False
    assert body["predictions"][1]["is_anomaly"] is True
    assert "latency_ms" in body


def test_wrong_feature_count_422(client):
    r = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert r.status_code == 422


def test_empty_features_422(client):
    r = client.post("/predict", json={"features": []})
    assert r.status_code == 422
    assert r.headers["content-type"].startswith("application/json")
    _ = r.json()


def test_nan_422(client, normal_features):
    # Send raw JSON with NaN (allow_nan) so the server must reject non-finite values
    feats = list(normal_features)
    feats[0] = float("nan")
    payload = json.dumps({"features": feats}, allow_nan=True)
    r = client.post(
        "/predict",
        content=payload.encode(),
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert "detail" in body


def test_inf_422(client, normal_features):
    feats = list(normal_features)
    feats[1] = float("inf")
    payload = json.dumps({"features": feats}, allow_nan=True)
    r = client.post(
        "/predict",
        content=payload.encode(),
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422
    assert r.headers["content-type"].startswith("application/json")
    _ = r.json()


def test_batch_inconsistent_lengths_422(client, normal_features):
    r = client.post(
        "/predict/batch",
        json={"instances": [normal_features, normal_features[:4]]},
    )
    assert r.status_code == 422
    _ = r.json()


def test_batch_oversized_422(client, normal_features):
    r = client.post(
        "/predict/batch",
        json={"instances": [normal_features] * 300},
    )
    assert r.status_code == 422


def test_invalid_type_422(client):
    r = client.post("/predict", json={"features": ["x", 1, 2, 3, 4, 5, 6, 7]})
    assert r.status_code == 422
    _ = r.json()


def test_health_503_when_unavailable(client_no_model):
    r = client_no_model.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "unavailable"
    assert body["model_loaded"] is False


def test_predict_503_when_unavailable(client_no_model, normal_features):
    r = client_no_model.post("/predict", json={"features": normal_features})
    assert r.status_code == 503


def test_no_public_logs_endpoint(client):
    r = client.get("/logs")
    assert r.status_code == 404
