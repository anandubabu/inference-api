"""Measure /predict p50 and p95 (end-to-end and server latency_ms)."""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--features",
        default="5.1,3.5,1.4,0.2",
        help="Comma-separated floats matching model input size",
    )
    args = parser.parse_args()
    features = [float(x) for x in args.features.split(",")]
    body = json.dumps({"features": features}).encode()

    def once() -> tuple[float, float]:
        t0 = time.perf_counter()
        req = urllib.request.Request(
            args.url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
        e2e = (time.perf_counter() - t0) * 1000
        return e2e, float(data["latency_ms"])

    for _ in range(args.warmup):
        once()

    e2e_ms: list[float] = []
    server_ms: list[float] = []
    for _ in range(args.n):
        e2e, srv = once()
        e2e_ms.append(e2e)
        server_ms.append(srv)

    print(f"n={args.n} url={args.url}")
    print(
        f"e2e_ms      p50={percentile(e2e_ms, 50):.3f}  p95={percentile(e2e_ms, 95):.3f}"
    )
    print(
        f"server_ms   p50={percentile(server_ms, 50):.3f}  p95={percentile(server_ms, 95):.3f}"
    )
    print(f"approx_rps  {1000 / percentile(e2e_ms, 50):.1f}  (from e2e p50, single client)")


if __name__ == "__main__":
    main()
