#!/usr/bin/env python3
"""Benchmark /predict and /predict/batch with honest p50/p95 and RPS.

RPS = completed_requests / total_elapsed_seconds  (NOT 1000/median).
Reports HTTP e2e latency and server-side latency_ms from the response.
Excludes warmup. Saves JSON under docs/benchmarks/.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "benchmarks"

# Default: a clearly normal-ish synthetic vector matching feature order
DEFAULT_FEATURES = [0.05, 0.15, -0.02, 0.18, 9.75, 0.12, 0.8, 24.5]


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def dep_versions() -> dict:
    vers = {}
    for mod in ("fastapi", "onnxruntime", "pydantic", "numpy", "sklearn", "uvicorn"):
        try:
            m = __import__(mod if mod != "sklearn" else "sklearn")
            vers[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            vers[mod] = "not-imported"
    return vers


def once(
    url: str,
    body: bytes,
    timeout: float,
) -> tuple[float, float | None, int, str | None]:
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
        e2e = (time.perf_counter() - t0) * 1000
        data = json.loads(raw)
        server = data.get("latency_ms")
        return e2e, float(server) if server is not None else None, status, None
    except Exception as exc:  # noqa: BLE001
        e2e = (time.perf_counter() - t0) * 1000
        return e2e, None, 0, f"{type(exc).__name__}: {exc}"


def run_sequential(url, body, n, warmup, timeout):
    for _ in range(warmup):
        once(url, body, timeout)
    e2e, server, failures = [], [], []
    t_wall0 = time.perf_counter()
    for _ in range(n):
        e, s, status, err = once(url, body, timeout)
        if err or status != 200:
            failures.append(err or f"status={status}")
        else:
            e2e.append(e)
            if s is not None:
                server.append(s)
    wall = time.perf_counter() - t_wall0
    return e2e, server, failures, wall


def run_concurrent(url, body, n, warmup, timeout, concurrency):
    for _ in range(warmup):
        once(url, body, timeout)
    e2e, server, failures = [], [], []
    t_wall0 = time.perf_counter()

    def job(_):
        return once(url, body, timeout)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        for e, s, status, err in ex.map(job, range(n)):
            if err or status != 200:
                failures.append(err or f"status={status}")
            else:
                e2e.append(e)
                if s is not None:
                    server.append(s)
    wall = time.perf_counter() - t_wall0
    return e2e, server, failures, wall


def summarize(name, e2e, server, failures, wall, n, extra: dict) -> dict:
    completed = len(e2e)
    rps = completed / wall if wall > 0 else 0.0
    summary = {
        "name": name,
        "requested": n,
        "completed_ok": completed,
        "failures": len(failures),
        "failure_samples": failures[:5],
        "total_elapsed_seconds": wall,
        "rps": rps,
        "e2e_ms": {
            "p50": percentile(e2e, 50) if e2e else None,
            "p95": percentile(e2e, 95) if e2e else None,
            "mean": statistics.fmean(e2e) if e2e else None,
        },
        "server_latency_ms": {
            "p50": percentile(server, 50) if server else None,
            "p95": percentile(server, 95) if server else None,
            "mean": statistics.fmean(server) if server else None,
        },
        **extra,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--features",
        default=",".join(str(x) for x in DEFAULT_FEATURES),
        help="Comma-separated floats for single /predict",
    )
    parser.add_argument("--skip-concurrent", action="store_true")
    parser.add_argument("--skip-batch", action="store_true")
    args = parser.parse_args()

    features = [float(x) for x in args.features.split(",")]
    single_body = json.dumps({"features": features}).encode()
    batch_body = json.dumps({"instances": [features] * args.batch_size}).encode()

    uname = platform.uname()
    try:
        uname_a = subprocess.check_output(["uname", "-a"], text=True).strip()
    except Exception:
        uname_a = " ".join(uname)

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "uname": uname_a,
            "processor": uname.processor or platform.processor(),
            "machine": uname.machine,
            "system": uname.system,
            "python": sys.version,
            "cpu_count": os.cpu_count(),
        },
        "dependency_versions": dep_versions(),
        "logging": {"LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO")},
        "notes": [
            "Warmup requests excluded from latency samples and RPS denominator.",
            "RPS = completed_ok / total_elapsed_seconds over the measured window.",
            "Not a hard real-time guarantee; single-host lab measurement only.",
            "Sequential and concurrent runs are reported separately.",
        ],
        "config": {
            "n": args.n,
            "warmup": args.warmup,
            "timeout_s": args.timeout,
            "concurrency": args.concurrency,
            "batch_size": args.batch_size,
            "n_features": len(features),
            "base": args.base,
        },
    }

    results = {"meta": meta, "runs": []}

    # Health check first
    try:
        with urllib.request.urlopen(args.base + "/health", timeout=args.timeout) as r:
            health = json.loads(r.read())
            if r.status != 200:
                raise SystemExit(f"Health not OK: {r.status} {health}")
    except Exception as exc:
        raise SystemExit(f"Cannot reach {args.base}/health: {exc}") from exc

    e2e, server, failures, wall = run_sequential(
        args.base + "/predict", single_body, args.n, args.warmup, args.timeout
    )
    results["runs"].append(
        summarize(
            "predict_sequential",
            e2e,
            server,
            failures,
            wall,
            args.n,
            {"mode": "sequential", "endpoint": "/predict", "batch_size": 1},
        )
    )

    if not args.skip_concurrent:
        e2e, server, failures, wall = run_concurrent(
            args.base + "/predict",
            single_body,
            args.n,
            args.warmup,
            args.timeout,
            args.concurrency,
        )
        results["runs"].append(
            summarize(
                "predict_concurrent",
                e2e,
                server,
                failures,
                wall,
                args.n,
                {
                    "mode": "concurrent",
                    "endpoint": "/predict",
                    "batch_size": 1,
                    "concurrency": args.concurrency,
                },
            )
        )

    if not args.skip_batch:
        e2e, server, failures, wall = run_sequential(
            args.base + "/predict/batch",
            batch_body,
            max(50, args.n // 4),
            max(5, args.warmup // 2),
            args.timeout,
        )
        results["runs"].append(
            summarize(
                "predict_batch_sequential",
                e2e,
                server,
                failures,
                wall,
                max(50, args.n // 4),
                {
                    "mode": "sequential",
                    "endpoint": "/predict/batch",
                    "batch_size": args.batch_size,
                },
            )
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT_DIR / f"results_{stamp}.json"
    latest = OUT_DIR / "results_latest.json"
    text = json.dumps(results, indent=2) + "\n"
    out_path.write_text(text)
    latest.write_text(text)

    # Human summary
    print(f"Saved {out_path}")
    print(f"Hardware: {uname_a}")
    for run in results["runs"]:
        print(
            f"{run['name']}: completed={run['completed_ok']}/{run['requested']} "
            f"failures={run['failures']} "
            f"e2e_p50={run['e2e_ms']['p50']:.3f}ms e2e_p95={run['e2e_ms']['p95']:.3f}ms "
            f"server_p50={run['server_latency_ms']['p50']} "
            f"RPS={run['rps']:.1f} (completed/total_elapsed)"
        )


if __name__ == "__main__":
    main()
